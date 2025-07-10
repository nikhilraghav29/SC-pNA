#!/usr/bin/python3
"""This recipe implements diarization system using deep embedding extraction followed by spectral clustering.

To run this recipe:
> python experiment.py hparams/<your_hyperparams_file.yaml>
 e.g., python experiment.py hparams/ecapa_tdnn.yaml

Condition: Oracle VAD (speech regions taken from the groundtruth).

Note: There are multiple ways to write this recipe. We iterate over individual recordings.
 This approach is less GPU memory demanding and also makes code easy to understand.

Citation: This recipe is based on the following paper,
 N. Dawalatabad, M. Ravanelli, F. Grondin, J. Thienpondt, B. Desplanques, H. Na,
 "ECAPA-TDNN Embeddings for Speaker Diarization," arXiv:2104.01466, 2021.

Authors
 * Nauman Dawalatabad 2020


"""

import os
import sys
import csv
import torch
import logging
import pickle
import json
import glob
import shutil
import numpy as np
# Add the desired path to the beginning of sys.path
sys.path.insert(0, '/home/nikhil/Documents/speechbrain/')
import speechbrain as sb
import pandas as pd
from tqdm.contrib import tqdm
from hyperpyyaml import load_hyperpyyaml

from speechbrain.utils.distributed import run_on_main
from speechbrain.processing.PLDA_LDA import StatObject_SB

from speechbrain.processing import dihard3_diarization as diar #CHANGE FOR DIHARD3
#import dihard3_diarization as diar
from speechbrain.utils.DER import DER
from speechbrain.dataio.dataio import read_audio
from speechbrain.dataio.dataio import read_audio_multichannel
import warnings
warnings.filterwarnings("ignore")

from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
)


np.random.seed(1234)

# Logger setup
logger = logging.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))


SAMPLERATE = 16000

try:
    import sklearn  # noqa F401
except ImportError:
    err_msg = "Cannot import optional dependency `scikit-learn` (sklearn) used in this module.\n"
    err_msg += "Please follow the below instructions\n"
    err_msg += "=============================\n"
    err_msg += "Using pip:\n"
    err_msg += "pip install scikit-learn\n"
    err_msg += "================================ \n"
    err_msg += "Using conda:\n"
    err_msg += "conda install scikit-learn"
    raise ImportError(err_msg)


def compute_embeddings(wavs, lens):
    """Definition of the steps for computation of embeddings from the waveforms."""
    with torch.no_grad():
        # Ensure the tensor is on the specified device
        wavs = wavs.to(run_opts["device"])
        # Save waveforms as .npy file
        #np.save('wavs_b4.npy', wavs.cpu().numpy())
        
        # Compute features
        feats = params["compute_features"](wavs)
        # Save features as .npy file
        #np.save('feats1_b4.npy', feats.cpu().numpy())
        
        # Normalize features
        feats = params["mean_var_norm"](feats, lens)
        # Save normalized features as .npy file
        #np.save('feats2_b4.npy', feats.cpu().numpy())
        
        # Compute embeddings
        emb = params["embedding_model"](feats, lens)
        # Save embeddings as .npy file
        #np.save('emb1_b4.npy', emb.cpu().numpy())
        
        # Normalize embeddings
        emb = params["mean_var_norm_emb"](emb, torch.ones(emb.shape[0], device=run_opts["device"]))
        # Save normalized embeddings as .npy file
        #np.save('emb2_b4.npy', emb.cpu().numpy())
        
    return emb


def embedding_computation_loop(split, set_loader, stat_file):
    """Extracts embeddings for a given dataset loader."""

    # Note: We use speechbrain.processing.PLDA_LDA.StatObject_SB type to store embeddings.
    # Extract embeddings (skip if already done).
    if not os.path.isfile(stat_file):

        logger.debug("Extracting deep embeddings and diarizing")
        embeddings = np.empty(shape=[0, params["emb_dim"]], dtype=np.float64)
        modelset = []
        segset = []

        # Different data may have different statistics.
        params["mean_var_norm_emb"].count = 0

        for batch in set_loader:

            ids = batch.id
            wavs, lens = batch.sig
            # Check if the audio length in samples is greater than 3200.
            if wavs.shape[1] > 500:                                     # Set this condition due to the issue of padding. In case of 'maptask' and 'socio_field' domain the condition should be replaced with 700 samples for reproducibility of the results.
                #print("The type of wavs is: ",type(wavs))
                #print("The shape of wavs is: ",wavs.shape)

                
                mod = [x for x in ids]
                seg = [x for x in ids]
                modelset = modelset + mod
                segset = segset + seg
                
                # Printing wavs, and lens
                #print("wavs is:",wavs)
                #print(wavs.shape)
                #print("lens is:",lens)
                # Embedding computation.
                emb = (
                    compute_embeddings(wavs, lens)
                    .contiguous()
                    .squeeze(1)
                    .cpu()
                    .numpy()
                )

                embeddings = np.concatenate((embeddings, emb), axis=0)
                #print("The type of emb is",type(embeddings))            
        
        

        modelset = np.array(modelset, dtype="|O")
        segset = np.array(segset, dtype="|O")

        # Intialize variables for start, stop and stat0.
        s = np.array([None] * embeddings.shape[0])
        b = np.array([[1.0]] * embeddings.shape[0])

        stat_obj = StatObject_SB(
            modelset=modelset,
            segset=segset,
            start=s,
            stop=s,
            stat0=b,
            stat1=embeddings,
        )
        logger.debug("Saving Embeddings...")
        stat_obj.save_stat_object(stat_file)

    else:
        logger.debug("Skipping embedding extraction (as already present).")
        logger.debug("Loading previously saved embeddings.")

        with open(stat_file, "rb") as in_file:
            stat_obj = pickle.load(in_file)

    return stat_obj


def prepare_subset_json(full_meta_data, rec_id, out_meta_file):
    """Prepares metadata for a given recording ID.

    Arguments
    ---------
    full_meta_data : json
        Full meta (json) containing all the recordings
    rec_id : str
        The recording ID for which meta (json) has to be prepared
    out_meta_file : str
        Path of the output meta (json) file.
    """

    subset = {}
    for key in full_meta_data:
        k = str(key)
        if k.startswith(rec_id):
            subset[key] = full_meta_data[key]

    with open(out_meta_file, mode="w") as json_f:
        json.dump(subset, json_f, indent=2)


def diarize_dataset(full_meta, split_type, n_lambdas, pval, n_neighbors=10):
    """This function diarizes all the recordings in a given dataset. It performs
    computation of embedding and clusters them using spectral clustering (or other backends).
    The output speaker boundary file is stored in the RTTM format.
    """

    # Prepare `spkr_info` only once when Oracle num of speakers is selected.
    # spkr_info is essential to obtain number of speakers from groundtruth.
    if params["oracle_n_spkrs"] is True:
        full_ref_rttm_file = (
            params["ref_rttm_dir"] + "/fullref_dihard_" + split_type + ".rttm"
        )

        rttm = diar.read_rttm(full_ref_rttm_file)

        spkr_info = list(  # noqa F841
            filter(lambda x: x.startswith("SPKR-INFO"), rttm)
        )
        print(spkr_info)

    # Get all the recording IDs in this dataset.
    all_keys = full_meta.keys()
    A = ['_'.join(line.split('_')[:3]) for line in all_keys]
    all_rec_ids = list(set(A[1:]))
    all_rec_ids.sort()

    split = "dihard_" + split_type
    i = 1

    # Setting eval modality.
    params["embedding_model"].eval()
    msg = "Diarizing " + split_type + " set"
    logger.info(msg)

    if len(all_rec_ids) <= 0:
        msg = "No recording IDs found! Please check if meta_data json file is properly generated."
        logger.error(msg)
        sys.exit()

    # Diarizing different recordings in a dataset.
    for rec_id in tqdm(all_rec_ids):
        # This tag will be displayed in the log.
        

        
        tag = (
            "["
            + str(split_type)
            + ": "
            + str(i)
            + "/"
            + str(len(all_rec_ids))
            + "]"
        )
        i = i + 1

        # Log message.
        msg = "Diarizing %s : %s " % (tag, rec_id)
        logger.debug(msg)

        # Embedding directory.
        if not os.path.exists(os.path.join(params["embedding_dir"], split)):
            os.makedirs(os.path.join(params["embedding_dir"], split))

        # File to store embeddings.
        emb_file_name = rec_id + ".emb_stat.pkl"
        diary_stat_emb_file = os.path.join(
            params["embedding_dir"], split, emb_file_name
        )

        # Prepare a metadata (json) for one recording. This is basically a subset of full_meta.
        # Lets keep this meta-info in embedding directory itself.
        json_file_name = rec_id + ".json"
        

        meta_per_rec_file = os.path.join(
            params["embedding_dir"], split, json_file_name
        )


        
        # Write subset (meta for one recording) json metadata.
        prepare_subset_json(full_meta, rec_id, meta_per_rec_file)

        # Prepare data loader.
        diary_set_loader = dataio_prep(params, split_type, meta_per_rec_file)
        
        

        # Putting modules on the device.
        params["compute_features"].to(run_opts["device"])
        params["mean_var_norm"].to(run_opts["device"])
        params["embedding_model"].to(run_opts["device"])
        params["mean_var_norm_emb"].to(run_opts["device"])

        # Compute Embeddings.
        diary_obj = embedding_computation_loop(
            "diary", diary_set_loader, diary_stat_emb_file
        )

        # Adding tag for directory path.
        type_of_num_spkr = "oracle" if params["oracle_n_spkrs"] else "est"
        tag = (
            type_of_num_spkr
            + "_"
            + str(params["affinity"])
            + "_"
            + params["backend"]
        )
        out_rttm_dir = os.path.join(
            params["sys_rttm_dir"], split, tag
        )
        if not os.path.exists(out_rttm_dir):
            os.makedirs(out_rttm_dir)
        out_rttm_file = out_rttm_dir + "/" + rec_id + ".rttm"

        # Processing starts from here.
        if params["oracle_n_spkrs"] is True:
            # Oracle num of speakers.
            num_spkrs = diar.get_oracle_num_spkrs(rec_id, spkr_info)
        else:
            if params["affinity"] == "nn":
                # Num of speakers tunned on dev set (only for nn affinity).
                num_spkrs = n_lambdas
            else:
                # Num of speakers will be estimated using max eigen gap for cos based affinity.
                # So adding None here. Will use this None later-on.
                num_spkrs = None

        if params["backend"] == "kmeans":
            diar.do_kmeans_clustering(
                diary_obj, out_rttm_file, rec_id, num_spkrs, pval,
            )

        if params["backend"] == "SC":
            # Go for Spectral Clustering (SC).
            diar.do_spec_clustering(
                diary_obj,
                out_rttm_file,
                rec_id,
                num_spkrs,
                pval,
                params["affinity"],
                n_neighbors,
            )


        if params["backend"] == "ASC":
            # Go for Spectral Clustering (SC).
            diar.do_spec_clustering_asc(
                diary_obj,
                out_rttm_file,
                rec_id,
                num_spkrs,
                pval,
                params["affinity"],
                n_neighbors,
            )
            
        # Can used for AHC later. Likewise one can add different backends here.
        if params["backend"] == "AHC":
            # call AHC
            threshold = pval  # pval for AHC is nothing but threshold.
            diar.do_AHC(diary_obj, out_rttm_file, rec_id, num_spkrs, threshold)

    # Once all RTTM outputs are generated, concatenate individual RTTM files to obtain single RTTM file.
    # This is not needed but just staying with the standards.
    concate_rttm_file = out_rttm_dir + "/sys_output.rttm"
    logger.debug("Concatenating individual RTTM files...")
    with open(concate_rttm_file, "w") as cat_file:
        for f in glob.glob(out_rttm_dir + "/*.rttm"):
            if f == concate_rttm_file:
                continue
            with open(f, "r") as indi_rttm_file:
                shutil.copyfileobj(indi_rttm_file, cat_file)

    msg = "The system generated RTTM file for %s set : %s" % (
        split_type,
        concate_rttm_file,
    )
    logger.debug(msg)

    return concate_rttm_file


def dev_pval_tuner(full_meta, split_type):
    """Tuning p_value for affinity matrix.
    The p_value used so that only p% of the values in each row is retained.
    """

    DER_list = []
    #prange = np.arange(0.001, 0.020, 0.001)
    #prange = np.arange(0.002, 0.03, 0.001) #np.arange(0.000, 0.030, 0.005)
    prange = np.arange(0.002, 0.003, 0.001)
    

    n_lambdas = None  # using it as flag later.
    for p_v in prange:
        # Process whole dataset for value of p_v.
        
        concate_rttm_file = diarize_dataset(
            full_meta, split_type, n_lambdas, p_v
        )

        ref_rttm = os.path.join(params["ref_rttm_dir"], "fullref_dihard_dev.rttm")
        sys_rttm = concate_rttm_file
        
        #print(ref_rttm)
        #print(sys_rttm)
        
        
        [MS, FA, SER, DER_] = DER(
            ref_rttm,
            sys_rttm,
            params["ignore_overlap"],
            params["forgiveness_collar"],
        )

        DER_list.append(DER_)


        #print(f"The DER at p_value {p_v:.3f} is {DER_:.2f}.")
        
        if params["oracle_n_spkrs"] is True and params["backend"] == "kmeans":
            # no need of p_val search. Note p_val is needed for SC for both oracle and est num of speakers.
            # p_val is needed in oracle_n_spkr=False when using kmeans backend.
            break
            
    #csv_file = "broadcast_interview_DER_list.csv"        
    # Write the list to a CSV file
    #with open(csv_file, 'w', newline='') as file:
     #   writer = csv.writer(file)
      #  writer.writerow(["Values"])  # Writing header if needed
       # writer.writerows(map(lambda x: [x], DER_list))
    
    # Take p_val that gave minmum DER on Dev dataset.
    tuned_p_val = prange[DER_list.index(min(DER_list))]
    min_der = min(DER_list)
    print("The best pval is:",tuned_p_val)
    return tuned_p_val, min_der


def dev_ahc_threshold_tuner(full_meta, split_type):
    """Tuning threshold for affinity matrix. This function is called when AHC is used as backend."""

    DER_list = []
    prange = np.arange(0.0, 1.0, 0.1)

    n_lambdas = None  # using it as flag later.

    # Note: p_val is threshold in case of AHC.
    for p_v in prange:
        # Process whole dataset for value of p_v.
        concate_rttm_file = diarize_dataset(
            full_meta, split_type, n_lambdas, p_v
        )

        ref_rttm = os.path.join(params["ref_rttm_dir"], "fullref_dihard_dev.rttm")
        sys_rttm = concate_rttm_file
        [MS, FA, SER, DER_] = DER(
            ref_rttm,
            sys_rttm,
            params["ignore_overlap"],
            params["forgiveness_collar"],
        )

        DER_list.append(DER_)

        if params["oracle_n_spkrs"] is True:
            break  # no need of threshold search.

    # Take p_val that gave minmum DER on Dev dataset.
    tuned_p_val = prange[DER_list.index(min(DER_list))]

    return tuned_p_val


def dev_nn_tuner(full_meta, split_type):
    """Tuning n_neighbors on dev set. Assuming oracle num of speakers.
    This is used when nn based affinity is selected.
    """

    DER_list = []
    pval = None

    # Now assumming oracle num of speakers.
    n_lambdas = 4

    for nn in range(5, 15):

        # Process whole dataset for value of n_lambdas.
        concate_rttm_file = diarize_dataset(
            full_meta, split_type, n_lambdas, pval, nn
        )

        ref_rttm = os.path.join(params["ref_rttm_dir"], "fullref_dihard_dev.rttm")
        sys_rttm = concate_rttm_file
        [MS, FA, SER, DER_] = DER(
            ref_rttm,
            sys_rttm,
            params["ignore_overlap"],
            params["forgiveness_collar"],
        )

        DER_list.append([nn, DER_])

        if params["oracle_n_spkrs"] is True and params["backend"] == "kmeans":
            break

    DER_list.sort(key=lambda x: x[1])
    tunned_nn = DER_list[0]

    return tunned_nn[0]


def dev_tuner(full_meta, split_type):
    """Tuning n_components on dev set. Used for nn based affinity matrix.
    Note: This is a very basic tunning for nn based affinity.
    This is work in progress till we find a better way.
    """

    DER_list = []
    pval = None
    for n_lambdas in range(1, params["max_num_spkrs"] + 1):

        # Process whole dataset for value of n_lambdas.
        concate_rttm_file = diarize_dataset(
            full_meta, split_type, n_lambdas, pval
        )

        ref_rttm = os.path.join(params["ref_rttm_dir"], "fullref_dihard_dev.rttm")
        sys_rttm = concate_rttm_file
        [MS, FA, SER, DER_] = DER(
            ref_rttm,
            sys_rttm,
            params["ignore_overlap"],
            params["forgiveness_collar"],
        )

        DER_list.append(DER_)

    # Take n_lambdas with minmum DER.
    tuned_n_lambdas = DER_list.index(min(DER_list)) + 1

    return tuned_n_lambdas


def dataio_prep(hparams, split_type, json_file):
    """Creates the datasets and their data processing pipelines.
    This is used for multi-mic processing.
    """

    # 1. Datasets
    
    if split_type == "dev":
        data_folder = hparams["data_folder_dev"]

    if split_type == "eval":
        data_folder = hparams["data_folder_eval"]
    
    dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=json_file, replacements={"data_root": data_folder},
    )
    #print("The type of dataset is:",type(dataset))

    # Single microphone
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item([dataset], audio_pipeline)


    # 3. Set output:
    sb.dataio.dataset.set_output_keys([dataset], ["id", "sig"])


    # 4. Create dataloader:
    dataloader = sb.dataio.dataloader.make_dataloader(
        dataset, **params["dataloader_opts"]
    )

    return dataloader



def get_subsegments(merged_segs, max_subseg_dur=3.0, overlap=1.5):
    """Divides bigger segments into smaller sub-segments
    """

    shift = max_subseg_dur - overlap
    subsegments = []

    # These rows are in RTTM format
    for row in merged_segs:
        seg_dur = float(row[4])
        rec_id = row[1]

        if seg_dur > max_subseg_dur:
            num_subsegs = int(seg_dur / shift)
            # Taking 0.01 sec as small step
            seg_start = float(row[3])
            seg_end = seg_start + seg_dur

            # Now divide this segment (new_row) in smaller subsegments
            for i in range(num_subsegs):
                subseg_start = seg_start + i * shift
                subseg_end = min(subseg_start + max_subseg_dur - 0.01, seg_end)
                subseg_dur = subseg_end - subseg_start

                new_row = [
                    "SPEAKER",
                    rec_id,
                    "1", #CHANGE FOR DIHARD3
                    str(round(float(subseg_start), 4)),
                    str(round(float(subseg_dur), 4)),
                    "<NA>",
                    "<NA>",
                    row[7],
                    "<NA>",
                    "<NA>",
                ]

                subsegments.append(new_row)

                # Break if exceeding the boundary
                if subseg_end >= seg_end:
                    break
        else:
            subsegments.append(row)

    return subsegments

def is_overlapped(end1, start2):
    """Returns True if the two segments overlap

    Arguments
    ---------
    end1 : float
        End time of the first segment.
    start2 : float
        Start time of the second segment.
    """

    if start2 > end1:
        return False
    else:
        return True
        
def merge_rttm_intervals(rttm_segs):
    """Merges adjacent segments in rttm if they overlap.
    """
    # For one recording
    # rec_id = rttm_segs[0][1]
    rttm_segs.sort(key=lambda x: float(x[3]))

    # first_seg = rttm_segs[0] # first interval.. as it is
    merged_segs = [rttm_segs[0]]
    strt = float(rttm_segs[0][3])
    end = float(rttm_segs[0][3]) + float(rttm_segs[0][4])

    for row in rttm_segs[1:]:
        s = float(row[3])
        e = float(row[3]) + float(row[4])

        if is_overlapped(end, s):
            # Update only end. The strt will be same as in last segment
            # Just update last row in the merged_segs
            end = max(end, e)
            merged_segs[-1][3] = str(round(strt, 4))
            merged_segs[-1][4] = str(round((end - strt), 4))
            merged_segs[-1][7] = "overlap"  # previous_row[7] + '-'+ row[7]
        else:
            # Add a new disjoint segment
            strt = s
            end = e
            merged_segs.append(row)  # this will have 1 spkr ID

    return merged_segs

def prepare_segs_for_RTTM(
    list_ids, out_rttm_file, audio_dir, annot_dir, split_type
):

    RTTM = []  # Stores all RTTMs clubbed together for a given dataset split

    for main_meet_id in list_ids:
    
        file_path = audio_dir + 'data/rttm/' + main_meet_id + '.rttm'
        
        with open(file_path, "r") as file:
            all_lines = file.readlines()
        
        rttm_per_rec = all_lines
        RTTM = RTTM + rttm_per_rec

    # Write one RTTM as groundtruth. For example, "fullref_eval.rttm"
    with open(out_rttm_file, "w") as f:
        for item in RTTM:
            f.write("%s" % item)
            


def prepare_metadata(
    rttm_file, save_dir, data_dir, filename, max_subseg_dur, overlap
):
    # Read RTTM, get unique meeting_IDs (from RTTM headers)
    # For each MeetingID. select that meetID -> merge -> subsegment -> json -> append

    # Read RTTM
    RTTM = []
    with open(rttm_file, "r") as f:
        for line in f:
            entry = line[:-1]
            RTTM.append(entry)

    rec_ids = set(item.split()[1] for item in RTTM)
    rec_ids = list(rec_ids)
    rec_ids.sort()
    

    # For each recording merge segments and then perform subsegmentation
    MERGED_SEGMENTS = []
    SUBSEGMENTS = []
    for rec_id in rec_ids:
        segs_iter = filter(
            lambda x: x.startswith("SPEAKER " + str(rec_id)), RTTM
        )
        gt_rttm_segs = [row.split(" ") for row in segs_iter]

        # Merge, subsegment and then convert to json format.
        merged_segs = merge_rttm_intervals(
            gt_rttm_segs
        )  # We lose speaker_ID after merging
        MERGED_SEGMENTS = MERGED_SEGMENTS + merged_segs

        # Divide segments into smaller sub-segments
        subsegs = get_subsegments(merged_segs, max_subseg_dur, overlap)
        SUBSEGMENTS = SUBSEGMENTS + subsegs

    # Write segment AND sub-segments (in RTTM format)
    segs_file = save_dir + "/" + filename + ".segments.rttm"
    subsegment_file = save_dir + "/" + filename + ".subsegments.rttm"

    with open(segs_file, "w") as f:
        for row in MERGED_SEGMENTS:
            line_str = " ".join(row)
            f.write("%s\n" % line_str)

    with open(subsegment_file, "w") as f:
        for row in SUBSEGMENTS:
            line_str = " ".join(row)
            f.write("%s\n" % line_str)

    # Create JSON from subsegments
    json_dict = {}
    for row in SUBSEGMENTS:
        rec_id = row[1]
        strt = str(round(float(row[3]), 4))
        end = str(round((float(row[3]) + float(row[4])), 4))
        subsegment_ID = rec_id + "_" + strt + "_" + end
        dur = row[4]
        start_sample = int(float(strt) * SAMPLERATE)
        end_sample = int(float(end) * SAMPLERATE)

        # Single mic audio
        wav_file_path = (
            data_dir
            + "data/flac/"
            + rec_id
            + ".flac"
        )

        # Note: key "file" without 's' is used for single-mic
        json_dict[subsegment_ID] = {
            "wav": {
                "file": wav_file_path,
                "duration": float(dur),
                "start": int(start_sample),
                "stop": int(end_sample),
            },
        }

    out_json_file = save_dir + "/" + filename + ".subsegs.json"
    with open(out_json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    msg = "%s JSON prepared" % (out_json_file)
    logger.debug(msg)


            
# Begin experiment!
if __name__ == "__main__":  # noqa: C901

    # Load hyperparameters file with command-line overrides.
    params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[1:])

    with open(params_file) as fin:
        params = load_hyperpyyaml(fin, overrides)

    data_folder_dev = params["data_folder_dev"]
    data_folder_eval = params["data_folder_eval"]
    save_folder = params["save_folder"]
    ref_rttm_dir = params["ref_rttm_dir"]
    meta_data_dir = params["meta_data_dir"]
    manual_annot_folder = params["manual_annot_folder"]
    subset_id_dev = params["subset_id_dev"]
    subset_id_eval = params["subset_id_eval"]

    vad_type = params["vad_type"]
    max_subseg_dur = params["max_subseg_dur"]
    overlap = params["overlap"]
    process_subset = params["process_subset"]



    if process_subset:
        dev_list_path = data_folder_dev + 'docs/recordings.tbl'
        df = pd.read_csv(dev_list_path, sep='\t')
        filtered_df = df[df.iloc[:, 3] == subset_id_dev]
        dev_list = filtered_df.iloc[:, 0].tolist()
    
    
        eval_list_path = data_folder_eval + 'docs/recordings.tbl'
        df = pd.read_csv(eval_list_path, sep='\t')
        filtered_df = df[df.iloc[:, 3] == subset_id_eval]
        eval_list = filtered_df.iloc[:, 0].tolist()
        
    else:
        dev_list_path = data_folder_dev + 'docs/recordings.tbl'
        df = pd.read_csv(dev_list_path, sep='\t')
        filtered_df = df
        dev_list = filtered_df.iloc[:, 0].tolist()
    
    
        eval_list_path = data_folder_eval + 'docs/recordings.tbl'
        df = pd.read_csv(eval_list_path, sep='\t')
        filtered_df = df
        eval_list = filtered_df.iloc[:, 0].tolist()
    
    # Meta files
    meta_files = [
        os.path.join(meta_data_dir, "dihard_dev.subsegs.json"),
        os.path.join(meta_data_dir, "dihard_eval.subsegs.json"),
    ]
    
    # Create configuration for easily skipping data_preparation stage
    conf = {
        "data_folder_dev": data_folder_dev,
        "data_folder_eval": data_folder_eval,
        "save_folder": save_folder,
        "ref_rttm_dir": ref_rttm_dir,
        "meta_data_dir": meta_data_dir,
        "process_subset": process_subset,
        "subset_id_dev": subset_id_dev,
        "subset_id_eval": subset_id_eval,
        "vad": vad_type,
        "max_subseg_dur": max_subseg_dur,
        "overlap": overlap,
        "meta_files": meta_files,
    }
    

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting output option files.
    opt_file = "opt_dihard_prepare.pkl"




    msg = "\tCreating meta-data file for the DIHARD3 Dataset.."
    logger.debug(msg)
    
    # Prepare RTTM from XML(manual annot) and store are groundtruth
    # Create ref_RTTM directory
    if not os.path.exists(ref_rttm_dir):
        os.makedirs(ref_rttm_dir)

    # Create reference RTTM files
    splits = ["dev", "eval"]
    for i in splits:
        rttm_file = ref_rttm_dir + "/fullref_dihard_" + i + ".rttm"
        if i == "dev":
            prepare_segs_for_RTTM(
                dev_list,
                rttm_file,
                data_folder_dev,
                manual_annot_folder,
                i,
            )
        if i == "eval":
            prepare_segs_for_RTTM(
                eval_list,
                rttm_file,
                data_folder_eval,
                manual_annot_folder,
                i,
            )

    # Create meta_files for splits
    meta_data_dir = meta_data_dir
    if not os.path.exists(meta_data_dir):
        os.makedirs(meta_data_dir)



    for i in splits:
        rttm_file = ref_rttm_dir + "/fullref_dihard_" + i + ".rttm"
        meta_filename_prefix = "dihard_" + i
        if i == "dev":
            prepare_metadata(
                rttm_file,
                meta_data_dir,
                data_folder_dev,
                meta_filename_prefix,
                max_subseg_dur,
                overlap,
            )
        if i == "eval":
            prepare_metadata(
                rttm_file,
                meta_data_dir,
                data_folder_eval,
                meta_filename_prefix,
                max_subseg_dur,
                overlap,
            )

    save_opt_file = os.path.join(save_folder, opt_file)
    save_pkl(conf, save_opt_file)
    
            
    # Create experiment directory.
    sb.core.create_experiment_directory(
        experiment_directory=params["output_folder"],
        hyperparams_to_save=params_file,
        overrides=overrides,
    )

    # Few more experiment directories inside results/ (to maintain cleaner structure).
    exp_dirs = [
        params["embedding_dir"],
        params["sys_rttm_dir"],
        params["der_dir"],
    ]
    for dir_ in exp_dirs:
        if not os.path.exists(dir_):
            os.makedirs(dir_)

    # We download the pretrained Model from HuggingFace (or elsewhere depending on
    # the path given in the YAML file).
    run_on_main(params["pretrainer"].collect_files)
    params["pretrainer"].load_collected(device=run_opts["device"])
    params["embedding_model"].eval()
    params["embedding_model"].to(run_opts["device"])

    # DIHARD3 Dev Set: Tune hyperparams on dev set.
    # Read the meta-data file for dev set generated during data_prep
    dev_meta_file = params["dev_meta_file"]
    with open(dev_meta_file, "r") as f:
        meta_dev = json.load(f)

    full_meta = meta_dev

    # Processing starts from here
    # Following few lines selects option for different backend and affinity matrices. Finds best values for hyperameters using dev set.
    best_nn = None
    if params["affinity"] == "nn":
        logger.info("Tuning for nn (Multiple iterations over DIHARD3 Dev set)")
        best_nn = dev_nn_tuner(full_meta, "dev")

    n_lambdas = None
    best_pval = None

    if params["affinity"] == "cos" and (
        params["backend"] == "SC" or params["backend"] == "kmeans" or params["backend"] == "ASC"
    ):
        if params["backend"] == "SC" or params["backend"] == "kmeans":
        
            # oracle num_spkrs or not, doesn't matter for kmeans and SC backends
            # cos: Tune for the best pval for SC /kmeans (for unknown num of spkrs)
            logger.info(
                "Tuning for p-value for SC (Multiple iterations over DIHARD3 Dev set)"
            )
            [best_pval, min_der] = dev_pval_tuner(full_meta, "dev")
            print(f"the best threshold is {best_pval:0.3f}")
            print(f"the DER at best threshold is {min_der:0.2f}")
            
        elif params["backend"] == "ASC":
        
            best_pval = None #dummy number
            

    elif params["backend"] == "AHC":
        logger.info("Tuning for threshold-value for AHC")
        best_threshold = dev_ahc_threshold_tuner(full_meta, "dev")
        best_pval = best_threshold
        
    else:
        # NN for unknown num of speakers (can be used in future)
        if params["oracle_n_spkrs"] is False:
            # nn: Tune num of number of components (to be updated later)
            logger.info(
                "Tuning for number of eigen components for NN (Multiple iterations over AMI Dev set)"
            )
            # dev_tuner used for tuning num of components in NN. Can be used in future.
            n_lambdas = dev_tuner(full_meta, "dev")

    # Load 'dev' and 'eval' metadata files.
    full_meta_dev = full_meta  # current full_meta is for 'dev'
    eval_meta_file = params["eval_meta_file"]
    with open(eval_meta_file, "r") as f:
        full_meta_eval = json.load(f)

    # Tag to be appended to final output DER files. Writing DER for individual files.
    type_of_num_spkr = "oracle" if params["oracle_n_spkrs"] else "est"
    tag = (
        type_of_num_spkr
        + "_"
        + str(params["affinity"])
    )

    # Perform final diarization on 'dev' and 'eval' with best hyperparams.
    final_DERs = {}
    for split_type in ["dev", "eval"]:
        if split_type == "dev":
            full_meta = full_meta_dev
        else:
            full_meta = full_meta_eval

        # Performing diarization.
        msg = "Diarizing using best hyperparams: " + split_type + " set"
        logger.info(msg)
        out_boundaries = diarize_dataset(
            full_meta,
            split_type,
            n_lambdas=n_lambdas,
            pval=best_pval,
            n_neighbors=best_nn,
        )

        # Computing DER.
        msg = "Computing DERs for " + split_type + " set"
        logger.info(msg)
        ref_rttm = os.path.join(
            params["ref_rttm_dir"], "fullref_dihard_" + split_type + ".rttm"
        )
        sys_rttm = out_boundaries
        [MS, FA, SER, DER_vals] = DER(
            ref_rttm,
            sys_rttm,
            params["ignore_overlap"],
            params["forgiveness_collar"],
            individual_file_scores=True,
        )

        # Writing DER values to a file. Append tag.
        der_file_name = split_type + "_DER_" + tag
        out_der_file = os.path.join(params["der_dir"], der_file_name)
        msg = "Writing DER file to: " + out_der_file
        logger.info(msg)
        diar.write_ders_file(ref_rttm, DER_vals, out_der_file)

        msg = (
            "DIHARD3 "
            + split_type
            + " set DER = %s %%\n" % (str(round(DER_vals[-1], 2)))
        )
        logger.info(msg)
        final_DERs[split_type] = round(DER_vals[-1], 2)

    # Final print DERs
    msg = (
        "Final Diarization Error Rate (%%) on DIHARD3 corpus: Dev = %.2f %% | Eval = %.2f %%\n"
        % ((final_DERs["dev"]), (final_DERs["eval"]))
    )
    logger.info(msg)
