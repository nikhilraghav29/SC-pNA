"""
This script contains basic functions used for speaker diarization.
This script has an optional dependency on open source scikit-learn (sklearn) library.
A few scikit-learn functions are modified in this script as per requirement.

Reference
---------
This code is written using the following:

- Von Luxburg, U. A tutorial on spectral clustering. Stat Comput 17, 395–416 (2007).
  https://doi.org/10.1007/s11222-007-9033-z

- https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/cluster/_spectral.py

- https://github.com/tango4j/Auto-Tuning-Spectral-Clustering/blob/master/spectral_opt.py

Authors
 * Nauman Dawalatabad 2020
"""

import csv
import numbers
import warnings
import scipy
import pytest
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.io import savemat

from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from sklearn.metrics import pairwise_distances

# The following function is defined for performing the MeanShift clustering algorithm.
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture

np.random.seed(1234)
pytest.importorskip("sklearn")

try:
    import sklearn
    from sklearn.neighbors import kneighbors_graph
    from sklearn.cluster import SpectralClustering
    from sklearn.cluster._kmeans import k_means
except ImportError:
    err_msg = "The optional dependency scikit-learn (sklearn) is used in this module\n"
    err_msg += "Cannot import scikit-learn. \n"
    err_msg += "Please follow the below instructions\n"
    err_msg += "=============================\n"
    err_msg += "Using pip:\n"
    err_msg += "pip install scikit-learn\n"
    err_msg += "================================ \n"
    err_msg += "Using conda:\n"
    err_msg += "conda install scikit-learn"
    raise ImportError(err_msg)


def read_rttm(rttm_file_path):
    """Reads and returns RTTM in list format.

    Arguments
    ---------
    rttm_file_path : str
        Path to the RTTM file to be read.

    Returns
    -------
    rttm : list
        List containing rows of RTTM file.
    """

    rttm = []
    with open(rttm_file_path, "r") as f:
        for line in f:
            entry = line[:-1]
            rttm.append(entry)
    return rttm


def write_ders_file(ref_rttm, DER, out_der_file):
    """Write the final DERs for individual recording.

    Arguments
    ---------
    ref_rttm : str
        Reference RTTM file.
    DER : array
        Array containing DER values of each recording.
    out_der_file : str
        File to write the DERs.
    """

    rttm = read_rttm(ref_rttm)
    #spkr_info = list(filter(lambda x: x.startswith("SPKR-INFO"), rttm))
    spkr_info = list(filter(lambda x: x.startswith("SPEAKER"), rttm))

    rec_id_list = []
    count = 0

    with open(out_der_file, "w") as f:
        for row in spkr_info:
            a = row.split(" ")
            rec_id = a[1]
            if rec_id not in rec_id_list:
                r = [rec_id, str(round(DER[count], 2))]
                rec_id_list.append(rec_id)
                line_str = " ".join(r)
                f.write("%s\n" % line_str)
                count += 1
        r = ["OVERALL ", str(round(DER[count], 2))]
        line_str = " ".join(r)
        f.write("%s\n" % line_str)


def prepare_subset_csv(full_diary_csv, rec_id, out_csv_file):
    """Prepares csv for a given recording ID.

    Arguments
    ---------
    full_diary_csv : csv
        Full csv containing all the recordings
    rec_id : str
        The recording ID for which csv has to be prepared
    out_csv_file : str
        Path of the output csv file.
    """

    out_csv_head = [full_diary_csv[0]]
    entry = []
    for row in full_diary_csv:
        if row[0].startswith(rec_id):
            entry.append(row)

    out_csv = out_csv_head + entry

    with open(out_csv_file, mode="w") as csv_file:
        csv_writer = csv.writer(
            csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for r in out_csv:
            csv_writer.writerow(r)


def is_overlapped(end1, start2):
    """Returns True if segments are overlapping.

    Arguments
    ---------
    end1 : float
        End time of the first segment.
    start2 : float
        Start time of the second segment.

    Returns
    -------
    overlapped : bool
        True of segments overlapped else False.

    Example
    -------
    >>> from speechbrain.processing import diarization as diar
    >>> diar.is_overlapped(5.5, 3.4)
    True
    >>> diar.is_overlapped(5.5, 6.4)
    False
    """

    if start2 > end1:
        return False
    else:
        return True


def merge_ssegs_same_speaker(lol):
    """Merge adjacent sub-segs from the same speaker.

    Arguments
    ---------
    lol : list of list
        Each list contains [rec_id, sseg_start, sseg_end, spkr_id].

    Returns
    -------
    new_lol : list of list
        new_lol contains adjacent segments merged from the same speaker ID.

    Example
    -------
    >>> from speechbrain.processing import diarization as diar
    >>> lol=[['r1', 5.5, 7.0, 's1'],
    ... ['r1', 6.5, 9.0, 's1'],
    ... ['r1', 8.0, 11.0, 's1'],
    ... ['r1', 11.5, 13.0, 's2'],
    ... ['r1', 14.0, 15.0, 's2'],
    ... ['r1', 14.5, 15.0, 's1']]
    >>> diar.merge_ssegs_same_speaker(lol)
    [['r1', 5.5, 11.0, 's1'], ['r1', 11.5, 13.0, 's2'], ['r1', 14.0, 15.0, 's2'], ['r1', 14.5, 15.0, 's1']]
    """

    new_lol = []

    # Start from the first sub-seg
    sseg = lol[0]
    flag = False
    for i in range(1, len(lol)):
        next_sseg = lol[i]

        # IF sub-segments overlap AND has same speaker THEN merge
        if is_overlapped(sseg[2], next_sseg[1]) and sseg[3] == next_sseg[3]:
            sseg[2] = next_sseg[2]  # just update the end time
            # This is important. For the last sseg, if it is the same speaker the merge
            # Make sure we don't append the last segment once more. Hence, set FLAG=True
            if i == len(lol) - 1:
                flag = True
                new_lol.append(sseg)
        else:
            new_lol.append(sseg)
            sseg = next_sseg

    # Add last segment only when it was skipped earlier.
    if flag is False:
        new_lol.append(lol[-1])

    return new_lol


def distribute_overlap(lol):
    """Distributes the overlapped speech equally among the adjacent segments
    with different speakers.

    Arguments
    ---------
    lol : list of list
        It has each list structure as [rec_id, sseg_start, sseg_end, spkr_id].

    Returns
    -------
    new_lol : list of list
        It contains the overlapped part equally divided among the adjacent
        segments with different speaker IDs.

    Example
    -------
    >>> from speechbrain.processing import diarization as diar
    >>> lol = [['r1', 5.5, 9.0, 's1'],
    ... ['r1', 8.0, 11.0, 's2'],
    ... ['r1', 11.5, 13.0, 's2'],
    ... ['r1', 12.0, 15.0, 's1']]
    >>> diar.distribute_overlap(lol)
    [['r1', 5.5, 8.5, 's1'], ['r1', 8.5, 11.0, 's2'], ['r1', 11.5, 12.5, 's2'], ['r1', 12.5, 15.0, 's1']]
    """

    new_lol = []
    sseg = lol[0]

    # Add first sub-segment here to avoid error at: "if new_lol[-1] != sseg:" when new_lol is empty
    # new_lol.append(sseg)

    for i in range(1, len(lol)):
        next_sseg = lol[i]
        # No need to check if they are different speakers.
        # Because if segments are overlapped then they always have different speakers.
        # This is because similar speaker's adjacent sub-segments are already merged by "merge_ssegs_same_speaker()"

        if is_overlapped(sseg[2], next_sseg[1]):

            # Get overlap duration.
            # Now this overlap will be divided equally between adjacent segments.
            overlap = sseg[2] - next_sseg[1]

            # Update end time of old seg
            sseg[2] = sseg[2] - (overlap / 2.0)

            # Update start time of next seg
            next_sseg[1] = next_sseg[1] + (overlap / 2.0)

            if len(new_lol) == 0:
                # For first sub-segment entry
                new_lol.append(sseg)
            else:
                # To avoid duplicate entries
                if new_lol[-1] != sseg:
                    new_lol.append(sseg)

            # Current sub-segment is next sub-segment
            sseg = next_sseg

        else:
            # For the first sseg
            if len(new_lol) == 0:
                new_lol.append(sseg)
            else:
                # To avoid duplicate entries
                if new_lol[-1] != sseg:
                    new_lol.append(sseg)

            # Update the current sub-segment
            sseg = next_sseg

    # Add the remaining last sub-segment
    new_lol.append(next_sseg)

    return new_lol


def write_rttm(segs_list, out_rttm_file):
    """Writes the segment list in RTTM format (A standard NIST format).

    Arguments
    ---------
    segs_list : list of list
        Each list contains [rec_id, sseg_start, sseg_end, spkr_id].
    out_rttm_file : str
        Path of the output RTTM file.
    """

    rttm = []
    rec_id = segs_list[0][0]

    for seg in segs_list:
        new_row = [
            "SPEAKER",
            rec_id,
            "1",
            str(round(seg[1], 4)),
            str(round(seg[2] - seg[1], 4)),
            "<NA>",
            "<NA>",
            seg[3],
            "<NA>",
            "<NA>",
        ]
        rttm.append(new_row)

    with open(out_rttm_file, "w") as f:
        for row in rttm:
            line_str = " ".join(row)
            f.write("%s\n" % line_str)


#######################################


def _graph_connected_component(graph, node_id):
    """Find the largest graph connected components that contains one
    given node.

    Arguments
    ---------
    graph : array-like, shape: (n_samples, n_samples)
        Adjacency matrix of the graph, non-zero weight means an edge
        between the nodes.
    node_id : int
        The index of the query node of the graph.

    Returns
    -------
    connected_components_matrix : array-like
        shape - (n_samples,).
        An array of bool value indicating the indexes of the nodes belonging
        to the largest connected components of the given query node.
    """

    n_node = graph.shape[0]
    if sparse.issparse(graph):
        # speed up row-wise access to boolean connection mask
        graph = graph.tocsr()
    connected_nodes = np.zeros(n_node, dtype=bool)
    nodes_to_explore = np.zeros(n_node, dtype=bool)
    nodes_to_explore[node_id] = True
    for _ in range(n_node):
        last_num_component = connected_nodes.sum()
        np.logical_or(connected_nodes, nodes_to_explore, out=connected_nodes)
        if last_num_component >= connected_nodes.sum():
            break
        indices = np.where(nodes_to_explore)[0]
        nodes_to_explore.fill(False)
        for i in indices:
            if sparse.issparse(graph):
                neighbors = graph[i].toarray().ravel()
            else:
                neighbors = graph[i]
            np.logical_or(nodes_to_explore, neighbors, out=nodes_to_explore)
    return connected_nodes


def _graph_is_connected(graph):
    """Return whether the graph is connected (True) or Not (False)

    Arguments
    ---------
    graph : array-like or sparse matrix, shape: (n_samples, n_samples)
        Adjacency matrix of the graph, non-zero weight means an edge between the nodes.

    Returns
    -------
    is_connected : bool
        True means the graph is fully connected and False means not.
    """

    if sparse.isspmatrix(graph):
        # sparse graph, find all the connected components
        n_connected_components, _ = connected_components(graph)
        return n_connected_components == 1
    else:
        # dense graph, find all connected components start from node 0
        return _graph_connected_component(graph, 0).sum() == graph.shape[0]


def _set_diag(laplacian, value, norm_laplacian):
    """
    Set the diagonal of the laplacian matrix and convert it to a sparse
    format well suited for eigenvalue decomposition.

    Arguments
    ---------
    laplacian : array or sparse matrix
        The graph laplacian.
    value : float
        The value of the diagonal.
    norm_laplacian : bool
        Whether the value of the diagonal should be changed or not.

    Returns
    -------
    laplacian : array or sparse matrix
        An array of matrix in a form that is well suited to fast eigenvalue
        decomposition, depending on the bandwidth of the matrix.
    """

    n_nodes = laplacian.shape[0]
    # We need all entries in the diagonal to values
    if not sparse.isspmatrix(laplacian):
        if norm_laplacian:
            laplacian.flat[:: n_nodes + 1] = value
    else:
        laplacian = laplacian.tocoo()
        if norm_laplacian:
            diag_idx = laplacian.row == laplacian.col
            laplacian.data[diag_idx] = value
        # If the matrix has a small number of diagonals (as in the
        # case of structured matrices coming from images), the
        # dia format might be best suited for matvec products:
        n_diags = np.unique(laplacian.row - laplacian.col).size
        if n_diags <= 7:
            # 3 or less outer diagonals on each side
            laplacian = laplacian.todia()
        else:
            # csr has the fastest matvec and is thus best suited to
            # arpack
            laplacian = laplacian.tocsr()
    return laplacian


def _deterministic_vector_sign_flip(u):
    """Modify the sign of vectors for reproducibility. Flips the sign of
    elements of all the vectors (rows of u) such that the absolute
    maximum element of each vector is positive.

    Arguments
    ---------
    u : ndarray
        Array with vectors as its rows.

    Returns
    -------
    u_flipped : ndarray
        Array with the sign flipped vectors as its rows. The same shape as `u`.
    """

    max_abs_rows = np.argmax(np.abs(u), axis=1)
    signs = np.sign(u[range(u.shape[0]), max_abs_rows])
    u *= signs[:, np.newaxis]
    return u


def _check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Arguments
    ---------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """

    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a np.random.RandomState" " instance" % seed
    )


#####################


def get_oracle_num_spkrs(rec_id, spkr_info):
    """
    Returns actual number of speakers in a recording from the ground-truth.
    This can be used when the condition is oracle number of speakers.

    Arguments
    ---------
    rec_id : str
        Recording ID for which the number of speakers have to be obtained.
    spkr_info : list
        Header of the RTTM file. Starting with `SPKR-INFO`.

    Example
    -------
    >>> from speechbrain.processing import diarization as diar
    >>> spkr_info = ['SPKR-INFO ES2011a 0 <NA> <NA> <NA> unknown ES2011a.A <NA> <NA>',
    ... 'SPKR-INFO ES2011a 0 <NA> <NA> <NA> unknown ES2011a.B <NA> <NA>',
    ... 'SPKR-INFO ES2011a 0 <NA> <NA> <NA> unknown ES2011a.C <NA> <NA>',
    ... 'SPKR-INFO ES2011a 0 <NA> <NA> <NA> unknown ES2011a.D <NA> <NA>',
    ... 'SPKR-INFO ES2011b 0 <NA> <NA> <NA> unknown ES2011b.A <NA> <NA>',
    ... 'SPKR-INFO ES2011b 0 <NA> <NA> <NA> unknown ES2011b.B <NA> <NA>',
    ... 'SPKR-INFO ES2011b 0 <NA> <NA> <NA> unknown ES2011b.C <NA> <NA>']
    >>> diar.get_oracle_num_spkrs('ES2011a', spkr_info)
    4
    >>> diar.get_oracle_num_spkrs('ES2011b', spkr_info)
    3
    """

    num_spkrs = 0
    for line in spkr_info:
        if rec_id in line:
            # Since rec_id is prefix for each speaker
            num_spkrs += 1

    return num_spkrs


def spectral_embedding_sb(
    adjacency, n_components=8, norm_laplacian=True, drop_first=True,
):
    """Returns spectral embeddings.

    Arguments
    ---------
    adjacency : array-like or sparse graph
        shape - (n_samples, n_samples)
        The adjacency matrix of the graph to embed.
    n_components : int
        The dimension of the projection subspace.
    norm_laplacian : bool
        If True, then compute normalized Laplacian.
    drop_first : bool
        Whether to drop the first eigenvector.

    Returns
    -------
    embedding : array
        Spectral embeddings for each sample.

    Example
    -------
    >>> import numpy as np
    >>> from speechbrain.processing import diarization as diar
    >>> affinity = np.array([[1, 1, 1, 0.5, 0, 0, 0, 0, 0, 0.5],
    ... [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    ... [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    ... [0.5, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    ... [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    ... [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    ... [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    ... [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    ... [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    ... [0.5, 0, 0, 0, 0, 0, 1, 1, 1, 1]])
    >>> embs = diar.spectral_embedding_sb(affinity, 3)
    >>> # Notice similar embeddings
    >>> print(np.around(embs , decimals=3))
    [[ 0.075  0.244  0.285]
     [ 0.083  0.356 -0.203]
     [ 0.083  0.356 -0.203]
     [ 0.26  -0.149  0.154]
     [ 0.29  -0.218 -0.11 ]
     [ 0.29  -0.218 -0.11 ]
     [-0.198 -0.084 -0.122]
     [-0.198 -0.084 -0.122]
     [-0.198 -0.084 -0.122]
     [-0.167 -0.044  0.316]]
    """

    # Whether to drop the first eigenvector
    if drop_first:
        n_components = n_components + 1

    if not _graph_is_connected(adjacency):
        warnings.warn(
            "Graph is not fully connected, spectral embedding"
            " may not work as expected."
        )

    laplacian, dd = csgraph_laplacian(
        adjacency, normed=norm_laplacian, return_diag=True
    )

    laplacian = _set_diag(laplacian, 1, norm_laplacian)

    laplacian *= -1

    vals, diffusion_map = eigsh(
        laplacian, k=n_components, sigma=1.0, which="LM",
    )

    embedding = diffusion_map.T[n_components::-1]

    if norm_laplacian:
        embedding = embedding / dd

    embedding = _deterministic_vector_sign_flip(embedding)
    if drop_first:
        return embedding[1:n_components].T
    else:
        return embedding[:n_components].T


def spectral_clustering_sb(
    affinity, n_clusters=8, n_components=None, random_state=None, n_init=10,
):
    """Performs spectral clustering.

    Arguments
    ---------
    affinity : matrix
        Affinity matrix.
    n_clusters : int
        Number of clusters for kmeans.
    n_components : int
        Number of components to retain while estimating spectral embeddings.
    random_state : int
        A pseudo random number generator used by kmeans.
     n_init : int
        Number of time the k-means algorithm will be run with different centroid seeds.

    Returns
    -------
    labels : array
        Cluster label for each sample.

    Example
    -------
    >>> import numpy as np
    >>> from speechbrain.processing import diarization as diar
    >>> affinity = np.array([[1, 1, 1, 0.5, 0, 0, 0, 0, 0, 0.5],
    ... [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    ... [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    ... [0.5, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    ... [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    ... [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    ... [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    ... [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    ... [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    ... [0.5, 0, 0, 0, 0, 0, 1, 1, 1, 1]])
    >>> labs = diar.spectral_clustering_sb(affinity, 3)
    >>> # print (labs) # [2 2 2 1 1 1 0 0 0 0]
    """

    random_state = _check_random_state(random_state)
    n_components = n_clusters if n_components is None else n_components

    maps = spectral_embedding_sb(
        affinity, n_components=n_components, drop_first=False,
    )

    _, labels, _ = k_means(
        maps, n_clusters, random_state=random_state, n_init=n_init
    )

    return labels


class Spec_Cluster(SpectralClustering):
    """Performs spectral clustering using sklearn on embeddings."""

    def perform_sc(self, X, n_neighbors=10):
        """
        Performs spectral clustering using sklearn on embeddings.

        Arguments
        ---------
        X : array (n_samples, n_features)
            Embeddings to be clustered.
        n_neighbors : int
            Number of neighbors in estimating affinity matrix.

        Reference
        ---------
        https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/cluster/_spectral.py
        """

        # Computation of affinity matrix
        connectivity = kneighbors_graph(
            X, n_neighbors=n_neighbors, include_self=True,
        )
        self.affinity_matrix_ = 0.5 * (connectivity + connectivity.T)

        # Perform spectral clustering on affinity matrix
        self.labels_ = spectral_clustering_sb(
            self.affinity_matrix_, n_clusters=self.n_clusters,
        )
        return self


#####################


class Spec_Clust_unorm:
    """
    This class implements the spectral clustering with unnormalized affinity matrix.
    Useful when affinity matrix is based on cosine similarities.

    Reference
    ---------
    Von Luxburg, U. A tutorial on spectral clustering. Stat Comput 17, 395–416 (2007).
    https://doi.org/10.1007/s11222-007-9033-z

    Example
    -------
    >>> from speechbrain.processing import diarization as diar
    >>> clust = diar.Spec_Clust_unorm(min_num_spkrs=2, max_num_spkrs=10)
    >>> emb = [[ 2.1, 3.1, 4.1, 4.2, 3.1],
    ... [ 2.2, 3.1, 4.2, 4.2, 3.2],
    ... [ 2.0, 3.0, 4.0, 4.1, 3.0],
    ... [ 8.0, 7.0, 7.0, 8.1, 9.0],
    ... [ 8.1, 7.1, 7.2, 8.1, 9.2],
    ... [ 8.3, 7.4, 7.0, 8.4, 9.0],
    ... [ 0.3, 0.4, 0.4, 0.5, 0.8],
    ... [ 0.4, 0.3, 0.6, 0.7, 0.8],
    ... [ 0.2, 0.3, 0.2, 0.3, 0.7],
    ... [ 0.3, 0.4, 0.4, 0.4, 0.7],]
    >>> # Estimating similarity matrix
    >>> sim_mat = clust.get_sim_mat(emb)
    >>> print (np.around(sim_mat[5:,5:], decimals=3))
    [[1.    0.957 0.961 0.904 0.966]
     [0.957 1.    0.977 0.982 0.997]
     [0.961 0.977 1.    0.928 0.972]
     [0.904 0.982 0.928 1.    0.976]
     [0.966 0.997 0.972 0.976 1.   ]]
    >>> # Prunning
    >>> pruned_sim_mat = clust.p_pruning(sim_mat, 0.3)
    >>> print (np.around(pruned_sim_mat[5:,5:], decimals=3))
    [[1.    0.    0.    0.    0.   ]
     [0.    1.    0.    0.982 0.997]
     [0.    0.977 1.    0.    0.972]
     [0.    0.982 0.    1.    0.976]
     [0.    0.997 0.    0.976 1.   ]]
    >>> # Symmetrization
    >>> sym_pruned_sim_mat = 0.5 * (pruned_sim_mat + pruned_sim_mat.T)
    >>> print (np.around(sym_pruned_sim_mat[5:,5:], decimals=3))
    [[1.    0.    0.    0.    0.   ]
     [0.    1.    0.489 0.982 0.997]
     [0.    0.489 1.    0.    0.486]
     [0.    0.982 0.    1.    0.976]
     [0.    0.997 0.486 0.976 1.   ]]
    >>> # Laplacian
    >>> laplacian = clust.get_laplacian(sym_pruned_sim_mat)
    >>> print (np.around(laplacian[5:,5:], decimals=3))
    [[ 1.999  0.     0.     0.     0.   ]
     [ 0.     2.468 -0.489 -0.982 -0.997]
     [ 0.    -0.489  0.975  0.    -0.486]
     [ 0.    -0.982  0.     1.958 -0.976]
     [ 0.    -0.997 -0.486 -0.976  2.458]]
    >>> # Spectral Embeddings
    >>> spec_emb, num_of_spk = clust.get_spec_embs(laplacian, 3)
    >>> print(num_of_spk)
    3
    >>> # Clustering
    >>> clust.cluster_embs(spec_emb, num_of_spk)
    >>> # print (clust.labels_) # [0 0 0 2 2 2 1 1 1 1]
    >>> # Complete spectral clustering
    >>> clust.do_spec_clust(emb, k_oracle=3, p_val=0.3)
    >>> # print(clust.labels_) # [0 0 0 2 2 2 1 1 1 1]
    """

    def __init__(self, min_num_spkrs=2, max_num_spkrs=10):

        self.min_num_spkrs = min_num_spkrs
        self.max_num_spkrs = max_num_spkrs
        
    def lower_triangular_to_vector(self, X):
        n = len(X)
        lower_triangular = []

        for i in range(n):
            for j in range(i):
                lower_triangular.append(X[i][j])

        return lower_triangular
    
    def do_spec_clust(self, X, k_oracle, p_val):
        """Function for spectral clustering.

        Arguments
        ---------
        X : array
            (n_samples, n_features).
            Embeddings extracted from the model.
        k_oracle : int
            Number of speakers (when oracle number of speakers).
        p_val : float
            p percent value to prune the affinity matrix.
        """
        
        #data = X

        #mean = np.mean(data, axis=0)
        #std_dev = np.std(data, axis=0)
        #standardized_data = (data - mean) / std_dev

        # Step 2: Apply PCA
        #pca = PCA()
        #pca.fit(standardized_data)

        # Step 3: Determine the number of components
        # You can use scree plot, explained variance ratio, or cumulative explained variance to decide the number of components to keep.
        # For example, you can plot pca.explained_variance_ratio_ to see the variance explained by each component.

        # Step 4: Transform the data onto the new feature space
        # Choose the number of components you want to keep based on the analysis in step 3
        #num_components = 10
        #X = pca.transform(standardized_data)[:, :num_components]
        # Similarity matrix computation
        sim_mat = self.get_sim_mat(X)
        
        #vector = self.lower_triangular_to_vector(sim_mat)
        #vector = np.array(vector)
        #vector = vector.reshape(-1, 1)
        #print(X.shape)
        
        #savemat('vector.mat', {'vector': vector,'X': X,'sim_mat': sim_mat})
        #kmeans = KMeans(n_clusters=2, random_state=0).fit(vector)
        
        #thres = np.max(kmeans.cluster_centers_)
        #print(thres)
        

        # Refining similarity matrix with p_val
        
        #sim_mat[sim_mat > thres] = 0
        #sim_mat[sim_mat > thres] = 1
        
        # I am commenting this
        pruned_sim_mat = self.p_pruning(sim_mat, p_val)
        #threshold = 0.5
        #pruned_sim_mat = self.p_pruning_score_row_wise(sim_mat, threshold)
        
        
        #savemat('vector.mat', {'sim_mat': sim_mat,'pruned_sim_mat': pruned_sim_mat,'X':X})
        #print(pruned_sim_mat)
        # Symmetrization
        sym_pruned_sim_mat = 0.5 * (pruned_sim_mat + pruned_sim_mat.T)

        # Laplacian calculation
        laplacian = self.get_laplacian(sym_pruned_sim_mat)

        # Get Spectral Embeddings
        emb, num_of_spk = self.get_spec_embs(laplacian, k_oracle)

        # Perform clustering
        #print("The predicted num of speakers:",num_of_spk)
        self.cluster_embs(emb, num_of_spk)
        # Using GMM instead of k-means at the clustering stage inside spectral clustering
        #self.do_GMM(emb,num_of_spk)

    def do_GMM(self, X,num_of_spk):
        """Function for GMM clustering.

        Arguments
        ---------
        X : array
            (n_samples, n_features).
            Embeddings extracted from the model.
        """
        # Define the Gaussian Mixture Model with 3 components
        gmm = GaussianMixture(n_components=num_of_spk, covariance_type= 'tied',max_iter=1,init_params="k-means++",random_state=42)
        
        # Fit the model to the data
        gmm.fit(X)   

        # Predict cluster labels for each data point
        predicted_labels = gmm.predict(X)     
        #Get the cluster labels assigned to each segment
        self.labels_ = predicted_labels

    def do_Mean_Shift(self, X):
        """Function for MeanShift clustering.

        Arguments
        ---------
        X : array
            (n_samples, n_features).
            Embeddings extracted from the model.
        """
        # Step 1: Normalize the speaker embeddings
        speaker_embeddings = X / np.linalg.norm(X, axis=1)[:, None]

        n_samples = max(int(X.shape[0]) // 10, 100)
        # Step 2: Estimate optimal bandwidth using estimate_bandwidth
        #bandwidth = estimate_bandwidth(speaker_embeddings, quantile=0.30, n_samples= (int(X.shape[0])//10))
        bandwidth = estimate_bandwidth(speaker_embeddings, quantile=0.25, n_samples= n_samples)

        #print("The estimated bandwidth is:", bandwidth)

        # Step 3: Apply MeanShift with the estimated bandwidth
        mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=False)
        mean_shift.fit(speaker_embeddings)

        # Step 4: Get the cluster labels assigned to each segment
        self.labels_ = mean_shift.labels_

    
    def do_Mean_Shift_old(self, X):
        """Function for spectral clustering.

        Arguments
        ---------
        X : array
            (n_samples, n_features).
            Embeddings extracted from the model.
        """
        # Assume that X holds the speaker embeddings in a numpy array
        speaker_embeddings = X/np.linalg.norm(X, axis = 1)[:, None]
       
        # Normalize the Embeddings
        # scaler = StandardScaler()
 
        # embeddings_normalized = scaler.fit_transform(speaker_embeddings)
        # Apply Mean Shift
        mean_shift = MeanShift(bandwidth=1.0)  # You can tune the bandwidth parameter
        mean_shift.fit(speaker_embeddings)
        # Get the cluster labels assigned to each segment
        #cluster_labels = mean_shift.labels_
        self.labels_ = mean_shift.labels_
        #print("Cluster labels for each segment:", cluster_labels)
        #return cluster_labels
        """

        # If one wants to use adapt the euclidean distance to the cosine similarity distance
        # Normalize the embeddings to unit vectors (L2 norm)
        normalized_embeddings = normalize(speaker_embeddings, norm='l2')    
        # Apply Mean Shift to normalized embeddings
        mean_shift = MeanShift(bandwidth=bandwidth)
        mean_shift.fit(normalized_embeddings)

        cluster_labels = mean_shift.labels_
        print("Cluster labels for each segment:", cluster_labels)
        """
        #return cluster_labels 
    
    # The following function return the upper off diagonal elements of a matrix.
    def upper_off_diag(self, matrix):
        # Here I assume that matrix is a list of shape n by n.
        upper_off = []  # Initialize an empty list to store the upper off-diagonal elements.
        
        # Iterate over the rows
        for i in range(matrix.shape[0]):  # Use matrix.shape[0] to get the number of rows
            # Iterate over the columns starting with index i+1
            for j in range(i + 1, matrix.shape[1]):  # Use matrix.shape[1] to get the number of columns
                upper_off.append(matrix[i, j])  # Append the upper off-diagonal element to the list
        
        return upper_off   
# EER-Delta
    def eer_delta(self, affinity_matrix):
        n = len(affinity_matrix)
        print("The number of embeddings are:",n)
        thresholds = []
        #top_40_scores = []
        #single_clusters = 0

        # Iterate over each row of the affinity matrix
        for i in range(n):
            #print("We are inside top 40 percent")
            # Step 1: Extract all elements in the row
            row_elements = affinity_matrix[i]
            
            # Extract all elements in the row, excluding the diagonal element
            #row_elements = np.concatenate((affinity_matrix[i][:i], affinity_matrix[i][i+1:]))

            #print("The type of row lwmwntsis", type(row_elements))            
            """ row_elemenrs is an ndarray"""
            # Step 2: Reshape row elements into a 2D array for clustering
            row_elements_reshaped = np.atleast_2d(row_elements).T

            # Step 3: Perform k-means clustering with 2 clusters
            kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(row_elements_reshaped)
            labels = kmeans.labels_

            # Step 4: Identify cluster centers
            cluster_centers = kmeans.cluster_centers_

            # Step 5: Determine which cluster has higher and lower center
            if cluster_centers[0] > cluster_centers[1]:
                C_label, I_label = 0, 1  # 0th cluster is the higher cluster
            else:
                C_label, I_label = 1, 0  # 1st cluster is the higher cluster

            # Step 6: Retrieve elements belonging to the higher cluster (C_label)
            C_elements = row_elements[labels == C_label]
            #print("The length of the ndarray of higher cluster is", len(C_elements))
            """ C_elements is an ndarray"""
            #print("The type of row elements is", type(C_elements)) 

            # Step 7: Sort the C_elements in decreasing order
            #sorted_C_elements = np.sort(C_elements)[::-1]

            #if len(C_elements) == 1:
                #single_clusters += 1

            # Step 8: Find the score corresponding to the top 40% of sorted elements
            #top_40_index = int((len(sorted_C_elements)-1) * 0.37)  # Find index of the top 40%
            #top_40_score = sorted_C_elements[top_40_index]  # Score at the 40% mark

            # Step 9: Append only the top 40% score to top_40_scores
            #top_40_scores.append(top_40_score)

            # Step 10: Retrieve elements belonging to the lower cluster (I_label)
            I_elements = row_elements[labels == I_label]

            # Step 11: Compute means and standard deviations for both clusters
            mu_C = np.mean(C_elements)
            sigma_C = np.std(C_elements)
            mu_I = np.mean(I_elements)
            sigma_I = np.std(I_elements)

            # Step 12: Calculate the optimal threshold using F-ratio normalization
            threshold = (mu_I * sigma_C + mu_C * sigma_I) / (sigma_I + sigma_C)

            # Step 13: Store the threshold for this row
            thresholds.append(threshold)
        #print("Count of the number of clusters with 1 element:", single_clusters)
        return thresholds


# The following function return the optimal threshold for pruning based on the EER and F-ratio.
    def optimal_threshold(self, matrix):
        off_diag_elements = self.lower_triangular_to_vector(matrix)

        # Reshape off_diag_elements to be 2D
        off_diag_elements_reshaped = np.array(off_diag_elements).reshape(-1, 1)

        kmeans = KMeans(n_clusters=2,random_state=0, n_init="auto").fit(off_diag_elements_reshaped) # Here we perform k-means clustering
        labels = kmeans.labels_
        #print("The labels of the points belonging to two classes are",labels, "of shape", labels.shape)

        cluster_centers = kmeans.cluster_centers_ 
        #print("The cluster are", cluster_centers.shape) 
        
        #print("The type of cluster centres is:",type(cluster_centers), "of shape", cluster_centers.shape)   
        #print(cluster_centers[0],cluster_centers[1])
        # To identify the clusters with higher centres as C and lower centres as I
        if cluster_centers[0] > cluster_centers[1]:
            C_label, I_label = 1 , 0
        else:
            C_label, I_label = 0, 1
        
        # Extract elements belonging to each cluster
        C_elements = [off_diag_elements[i] for i in range(len(off_diag_elements)) if labels[i] == C_label]
        I_elements = [off_diag_elements[i] for i in range(len(off_diag_elements)) if labels[i] == I_label]

        # Compute means and standard deviations for both clusters
        mu_C = np.mean(C_elements)
        sigma_C = np.std(C_elements)
        mu_I = np.mean(I_elements)
        sigma_I = np.std(I_elements)

        # Print the results
        #print(f"Cluster C (higher center): mu_C = {mu_C}, sigma_C = {sigma_C}")
        #print(f"Cluster I (lower center): mu_I = {mu_I}, sigma_I = {sigma_I}")

        # The optimal threshold defined using F-ratio normalization as given in equation 4 of https://publications.idiap.ch/attachments/reports/2004/rr04-46.pdf
        threshold =  (mu_I*sigma_C + mu_C*sigma_I)/(sigma_I + sigma_C)

        return threshold

    def do_spec_clust_row_wise(self, X, k_oracle, pval):
        """Function for spectral clustering.

        Arguments
        ---------
        X : array
            (n_samples, n_features).
            Embeddings extracted from the model.
        k_oracle : int
            Number of speakers (when oracle number of speakers).
        p_val : float
            p percent value to prune the affinity matrix.
        """
        
        # Similarity matrix computation
        sim_mat = self.get_sim_mat(X)
        #print("The type of Sim-mat is:", type(sim_mat))
        # row-wise thresholds computation using F-ratio

        #Upper_A = self.lower_triangular_to_vector(sim_mat)
        #threshold = self.optimal_threshold(sim_mat)
        #thresholds = self.eer_delta(sim_mat)                                # EER-Delta
        thresholds = self.optimal_threshold_top_fourty(sim_mat)            # SC-pNA
        pruned_sim_mat = self.p_pruning_score_row_wise(sim_mat, thresholds) # SC-pNA, EER-Delta
        #pruned_sim_mat = self.p_pruning_score_row_wise(sim_mat, threshold) 

        # Symmetrization
        #sym_pruned_sim_mat = self.symmetrization(pruned_sim_mat)
        sym_pruned_sim_mat = 0.5 * (pruned_sim_mat + pruned_sim_mat.T)

        # Laplacian calculation
        laplacian = self.get_laplacian(sym_pruned_sim_mat)

        # Get Spectral Embeddings
        emb, num_of_spk = self.get_spec_embs(laplacian, k_oracle)

        # Perform clustering
        #print(num_of_spk)
        # I have commented the following line to apply the GMM for clustering instead of k-means
        self.cluster_embs(emb, num_of_spk)
        print("Estimated number of speakers:",num_of_spk)
        #self.do_GMM(emb,num_of_spk)

    def symmetrization(self, affinity_matrix):
        A = affinity_matrix
        n = A.shape[0]
        
        for i in range(n):
            for j in range(i + 1, n):  # Only need to check the upper triangle
                if A[i][j] != A[j][i]:
                    max_val = 0.8  # Find the maximum value
                    A[i][j] = max_val  # Set both A[i][j] and A[j][i] to max_val
                    A[j][i] = max_val
                    
        return A


    def optimal_threshold_row_wise(self, affinity_matrix):
        """
        Calculate the optimal threshold for each row of the affinity matrix using KMeans clustering.
        
        For each row of the matrix (excluding the diagonal), we apply k-means clustering, compute
        cluster statistics, and calculate an optimal threshold based on F-ratio normalization.
        
        Parameters:
        affinity_matrix (list of list or np.array): A 2D square affinity matrix of shape N x N.
        
        Returns:
        list: A list of N optimal thresholds, one for each row of the affinity matrix.
        """
        n = len(affinity_matrix)
        thresholds = []
        centers = []

        # Iterate over each row of the affinity matrix
        for i in range(n):
            # Step 1: Extract all elements in the row except the diagonal
            row_elements = affinity_matrix[i]
            # Step 2: Reshape row elements into a 2D array for clustering
            #row_elements_reshaped = np.array(row_elements).reshape(-1, 1)
            row_elements_reshaped = np.atleast_2d(row_elements).T

            # Step 3: Perform k-means clustering with 2 clusters
            kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(row_elements_reshaped)
            labels = kmeans.labels_

            # Step 4: Identify cluster centers
            cluster_centers = kmeans.cluster_centers_

            # Step 5: Determine which cluster has higher and lower center
            if cluster_centers[0] > cluster_centers[1]:
                C_label, I_label = 1, 0
                centers.append(cluster_centers[0])
            else:
                C_label, I_label = 0, 1
                centers.append(cluster_centers[1])                

            # To retrieve the elements with C_label
            C_elements = row_elements[labels == C_label]
            
            # To retrieve the elements with I_label
            I_elements = row_elements[labels == I_label]
            # Step 6: Separate elements into two clusters based on their labels
            #C_elements = [row_elements[k] for k in range(len(row_elements)) if labels[k] == C_label]
            #I_elements = [row_elements[k] for k in range(len(row_elements)) if labels[k] == I_label]

            # Step 7: Compute means and standard deviations for both clusters
            mu_C = np.mean(C_elements)
            sigma_C = np.std(C_elements)
            mu_I = np.mean(I_elements)
            sigma_I = np.std(I_elements)

            # Step 8: Calculate the optimal threshold using F-ratio normalization
            threshold = (mu_I * sigma_C + mu_C * sigma_I) / (sigma_I + sigma_C)

            # Step 9: Store the threshold for this row
            thresholds.append(threshold)

        return centers, thresholds

    def spectral_clustering_gt(self, X):
        GS = pairwise_distances(X, X, metric='sqeuclidean')

        # Find global threshold
        prec = 0.1
        min_lim = 1
        W = np.exp(- GS / (2 * min_lim ** 2))
        if connected_components(W)[0] == 1:
            min_lim = 0.1
        max_lim = 3 * X.std()
        W = np.exp(- GS / (2 * (max_lim ** 2)))
        if connected_components(W)[0] == 1:
            max_lim = 6 * X.std()
        while 1:
            sigma = 0.5 * (max_lim + min_lim)
            W = np.exp(- GS / (2 * (sigma ** 2)))
            mid_conn = connected_components(W)[0]
            if np.abs(max_lim - sigma) <= prec:
                break
            if mid_conn == 1:
                max_lim = sigma
            elif mid_conn > 1:
                min_lim = sigma
        sigma = np.round(2 * max_lim)
        W = np.exp(- GS / (2 * (sigma ** 2)))
        print(connected_components(W)[0])
        
        # Spectral clustering with sigma threshold
        D = W.sum(axis=1)
        L = np.diag(D) - W
        eigvals, eigvecs = np.linalg.eigh(L)
        # k from eigengaps
        k = (eigvals[1:11] - eigvals[0:10]).argmax()+1
        print('k', k)
        S = eigvecs[:,0:k]
        mem = KMeans(n_clusters=k, max_iter=200, n_init=30).fit(S).labels_
        self.labels_ = mem
        #return mem, k

    def optimal_threshold_top_fourty(self, affinity_matrix):
        """
        Calculate the optimal threshold for each row of the affinity matrix using KMeans clustering and find
        the score corresponding to the top 40% of the higher cluster center.
        
        For each row, perform k-means clustering to separate the elements into two clusters, compute cluster statistics, 
        and calculate an optimal threshold. Additionally, find the similarity score corresponding to the top 40% 
        of the higher cluster center and append it to the list.
        
        Parameters:
        affinity_matrix (list of list or np.array): A 2D square affinity matrix of shape N x N.
        
        Returns:
        list: A list of N tuples, each containing the higher cluster center and the top 40% score.
        list: A list of N optimal thresholds, one for each row of the affinity matrix.
        """
        n = len(affinity_matrix)
        print("The number of embeddings are:",n)
        thresholds = []
        top_40_scores = []
        single_clusters = 0

        # Iterate over each row of the affinity matrix
        for i in range(n):
            #print("We are inside top 40 percent")
            # Step 1: Extract all elements in the row
            #row_elements = affinity_matrix[i]
            
            # Extract all elements in the row, excluding the diagonal element
            row_elements = np.concatenate((affinity_matrix[i][:i], affinity_matrix[i][i+1:]))

            #print("The type of row lwmwntsis", type(row_elements))            
            """ row_elemenrs is an ndarray"""
            # Step 2: Reshape row elements into a 2D array for clustering
            row_elements_reshaped = np.atleast_2d(row_elements).T

            # Step 3: Perform k-means clustering with 2 clusters
            kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(row_elements_reshaped)
            labels = kmeans.labels_

            # Step 4: Identify cluster centers
            cluster_centers = kmeans.cluster_centers_

            # Step 5: Determine which cluster has higher and lower center
            if cluster_centers[0] > cluster_centers[1]:
                C_label, I_label = 0, 1  # 0th cluster is the higher cluster
            else:
                C_label, I_label = 1, 0  # 1st cluster is the higher cluster

            # Step 6: Retrieve elements belonging to the higher cluster (C_label)
            C_elements = row_elements[labels == C_label]
            #print("The length of the ndarray of higher cluster is", len(C_elements))
            """ C_elements is an ndarray"""
            #print("The type of row elements is", type(C_elements)) 

            # Step 7: Sort the C_elements in decreasing order
            sorted_C_elements = np.sort(C_elements)[::-1]

            if len(C_elements) == 1:
                single_clusters += 1

            # Step 8: Find the score corresponding to the top 40% of sorted elements
            top_40_index = int((len(sorted_C_elements)-1) * 0.20)  # Find index of the top 40%
            top_40_score = sorted_C_elements[top_40_index]  # Score at the 40% mark

            # Step 9: Append only the top 40% score to top_40_scores
            top_40_scores.append(top_40_score)

            # Step 10: Retrieve elements belonging to the lower cluster (I_label)
            I_elements = row_elements[labels == I_label]

            # Step 11: Compute means and standard deviations for both clusters
            mu_C = np.mean(C_elements)
            sigma_C = np.std(C_elements)
            mu_I = np.mean(I_elements)
            sigma_I = np.std(I_elements)

            # Step 12: Calculate the optimal threshold using F-ratio normalization
            threshold = (mu_I * sigma_C + mu_C * sigma_I) / (sigma_I + sigma_C)

            # Step 13: Store the threshold for this row
            thresholds.append(threshold)
        print("Count of the number of clusters with 1 element:", single_clusters)
        return top_40_scores

    def do_spec_clust_asc(self, X, k_oracle, p_val):
        """Function for auto-tuning spectral clustering written by Md Sahidullah
        Original paper: 
        [1] Park, T.J., Han, K.J., Kumar, M. and Narayanan, S., 2019. 
        Auto-tuning spectral clustering for speaker diarization using 
        normalized maximum eigengap. IEEE Signal Processing Letters, 27, pp.381-385.

        Arguments
        ---------
        X : array
            (n_samples, n_features).
            Embeddings extracted from the model.
        k_oracle : int
            Number of speakers (when oracle number of speakers).
        p_val : float
            p percent value to prune the affinity matrix.
            Note: this is fake entry and remained unused.
        """
        """
        print("Min:", np.min(X))
        print("Max:", np.max(X))
        print("Mean:", np.mean(X))
        print("Median:", np.median(X)) 
        print("Standard deviation:", np.std(X))               
        """
        Prange = np.arange(1, X.shape[0]-1, 1)
        #Equation (4) of [1]
        A = self.get_sim_mat(X)
      
        rp = np.empty(X.shape[0]-2)
        
        for p in Prange:
            #Equation (5) of [1]
            A_p = self.binarize_matrix(A, p)

            #Equation (6) of [1]
            A_p_bar = 0.5 * (A_p + A_p.T)

            #Equation (7) of [1]
            L = self.get_laplacian(A_p_bar)
            
            #Alternative implementation of Equation (8) of [1]
            lambdas, eig_vecs = scipy.linalg.eigh(L)
            
            #Equation (9) of [1]
            lambda_p = np.sort(lambdas)          
            ep = self.getEigenGaps(lambda_p)
            
            
            epsilon = 1e-10
            
            #Equation (10) of [1]
            gp = np.max(ep) / (lambda_p[-1] + epsilon)
            
            #Equation (11) of [1]
            rp[p-1] = p / (gp + epsilon)
            

            
        # Sorting of rp in ascending order
        sorted_indices = np.argsort(rp)
        p_bar = Prange[sorted_indices]     
        
        #Do spectral clusting with p corresponding to lowest rp 
        A_p = self.binarize_matrix(A, p_bar[0])
        
        
        sym_pruned_sim_mat = 0.5 * (A_p + A_p.T)
        
        
        #The spectral clustering is not reliable if the graph is not connected.
        #In that case, keep increasing the value of p according to the list prepared in p_bar.
        if not _graph_is_connected(sym_pruned_sim_mat):
            print("Graph is not fully connected, spectral embedding may not work as expected.")
            print("Decreasing pruning rate....")
            
            for j in p_bar[1:-1]:
                A_p = self.binarize_matrix(A, j)
                sym_pruned_sim_mat = 0.5 * (A_p + A_p.T)
                if _graph_is_connected(sym_pruned_sim_mat):
                    print("Congratulations! Found good pruning rate to create fully connected graph!")
                    print(f"Pruning rate through ASC was {p_bar[0]}.")
                    print(f"Current prun rate is set to {j}.")
                    break
                
        
        
        
        
        #Computation of Laplacian for final clustering. The remaining process is very similar to the earlier
        #standard spectral clustering algorithm.
        laplacian = self.get_laplacian(sym_pruned_sim_mat)  
        
        
        emb, num_of_spk = self.get_spec_embs(laplacian, k_oracle)        


        self.cluster_embs(emb, num_of_spk)

    def get_sim_mat(self, X):
        """Returns the similarity matrix based on cosine similarities.

        Arguments
        ---------
        X : array
            (n_samples, n_features).
            Embeddings extracted from the model.

        Returns
        -------
        M : array
            (n_samples, n_samples).
            Similarity matrix with cosine similarities between each pair of embedding.
        """

        # Cosine similarities
        M = sklearn.metrics.pairwise.cosine_similarity(X, X)
        return M

    def p_pruning(self, A, pval):
        """Refine the affinity matrix by zeroing less similar values.

        Arguments
        ---------
        A : array
            (n_samples, n_samples).
            Affinity matrix.
        pval : float
            p-value to be retained in each row of the affinity matrix.

        Returns
        -------
        A : array
            (n_samples, n_samples).
            pruned affinity matrix based on p_val.
        """

        n_elems = int((1 - pval) * A.shape[0])

        # For each row in a affinity matrix
        for i in range(A.shape[0]):
            low_indexes = np.argsort(A[i, :])
            low_indexes = low_indexes[0:n_elems]

            # Replace smaller similarity values by 0s
            A[i, low_indexes] = 0

        return A

# The following function is written by Sahid da, for pruning the elements in the affinity matrix on the basis of a threshold value.
    def p_pruning_score(self, A, pval):
            """Refine the affinity matrix by zeroing less similar values.

            Arguments
            ---------
            A : array
                (n_samples, n_samples).
                Affinity matrix.
            pval : float
                Threshold value. Retain elements greater than this number and set all other elements to zero.

            Returns
            -------
            A : array
                (n_samples, n_samples).
                pruned affinity matrix based on p_val.
            """

            A = np.where(A > pval, A, 0)

            return A

    def p_pruning_score_row_wise(self, A, thresholds):
        """
        Refine the affinity matrix by zeroing less similar values based on row-wise thresholds.

        Arguments
        ---------
        A : array (n_samples, n_samples)
            Affinity matrix.
        thresholds : list of floats
            List of threshold values for each row. Retain elements greater than the row's threshold
            and set all other elements to zero.

        Returns
        -------
        A : array (n_samples, n_samples)
            Pruned affinity matrix where values are zeroed based on row-wise thresholds.
        """
        # Iterate over each row and apply the corresponding threshold
        #print("The max threshold is:", np.max(thresholds))
        for i in range(len(A)):
            #A[i] = np.where(A[i] > np.mean(thresholds), A[i], 0)
            A[i] = np.where(A[i] > thresholds[i], A[i], 0)
            #A[i] = np.where(A[i] > threshold, A[i], 0) # Upper traingular 
        
        return A

    def get_laplacian(self, M):
        """Returns the un-normalized laplacian for the given affinity matrix.

        Arguments
        ---------
        M : array
            (n_samples, n_samples)
            Affinity matrix.

        Returns
        -------
        L : array
            (n_samples, n_samples)
            Laplacian matrix.
        """

        M[np.diag_indices(M.shape[0])] = 0
        D = np.sum(np.abs(M), axis=1)
        D = np.diag(D)
        L = D - M
        return L

    def binarize_matrix(self, X, k):
    
        binary_matrix = np.zeros_like(X, dtype=int)
        
        n = X.shape[0]

        for i in range(n):

            largest_k_indices = np.argsort(X[i])[-k:]        

            binary_matrix[i, largest_k_indices] = 1      



        return binary_matrix 
        
    def get_spec_embs(self, L, k_oracle=4):
        """Returns spectral embeddings and estimates the number of speakers
        using maximum Eigen gap.

        Arguments
        ---------
        L : array (n_samples, n_samples)
            Laplacian matrix.
        k_oracle : int
            Number of speakers when the condition is oracle number of speakers,
            else None.

        Returns
        -------
        emb : array (n_samples, n_components)
            Spectral embedding for each sample with n Eigen components.
        num_of_spk : int
            Estimated number of speakers. If the condition is set to the oracle
            number of speakers then returns k_oracle.
        """
 
        lambdas, eig_vecs = scipy.linalg.eigh(L)

        # if params["oracle_n_spkrs"] is True:
        if k_oracle is not None:
            num_of_spk = k_oracle
        else:
            lambda_gap_list = self.getEigenGaps(lambdas[1 : self.max_num_spkrs])

            num_of_spk = (
                np.argmax(
                    lambda_gap_list[
                        : min(self.max_num_spkrs, len(lambda_gap_list))
                    ]
                )
                if lambda_gap_list
                else 0
            ) + 2

            if num_of_spk < self.min_num_spkrs:
                num_of_spk = self.min_num_spkrs
        #print("The number of estimated speakers:", num_of_spk)
        emb = eig_vecs[:, 0:num_of_spk]

        return emb, num_of_spk

    def cluster_embs(self, emb, k):
        """Clusters the embeddings using kmeans.

        Arguments
        ---------
        emb : array (n_samples, n_components)
            Spectral embedding for each sample with n Eigen components.
        k : int
            Number of clusters to kmeans.

        Returns
        -------
        self.labels_ : self
            Labels for each sample embedding.
        """
        _, self.labels_, _ = k_means(emb, k)

    def getEigenGaps(self, eig_vals):
        """Returns the difference (gaps) between the Eigen values.

        Arguments
        ---------
        eig_vals : list
            List of eigen values

        Returns
        -------
        eig_vals_gap_list : list
            List of differences (gaps) between adjacent Eigen values.
        """

        eig_vals_gap_list = []
        for i in range(len(eig_vals) - 1):
            gap = float(eig_vals[i + 1]) - float(eig_vals[i])
            # eig_vals_gap_list.append(float(eig_vals[i + 1]) - float(eig_vals[i]))
            eig_vals_gap_list.append(gap)

        return eig_vals_gap_list


#####################


def do_spec_clustering(
    diary_obj, out_rttm_file, rec_id, k, pval, affinity_type, n_neighbors
):
    """Performs spectral clustering on embeddings. This function calls specific
    clustering algorithms as per affinity.

    Arguments
    ---------
    diary_obj : StatObject_SB type
        Contains embeddings in diary_obj.stat1 and segment IDs in diary_obj.segset.
    out_rttm_file : str
        Path of the output RTTM file.
    rec_id : str
        Recording ID for the recording under processing.
    k : int
        Number of speaker (None, if it has to be estimated).
    pval : float
        `pval` for prunning affinity matrix.
    affinity_type : str
        Type of similarity to be used to get affinity matrix (cos or nn).
    """

    if affinity_type == "cos":
        clust_obj = Spec_Clust_unorm(min_num_spkrs=2, max_num_spkrs=10)
        k_oracle = k  # use it only when oracle num of speakers
        #clust_obj.spectral_clustering_gt(diary_obj.stat1)
        #clust_obj.do_spec_clust(diary_obj.stat1, k_oracle, pval) # pval based
        clust_obj.do_spec_clust_row_wise(diary_obj.stat1, k_oracle, pval) # SC-pNA and EER-Delta based
        #labels = clust_obj.labels_
        #clust_obj.do_GMM(diary_obj.stat1)
        labels = clust_obj.labels_ 
    else:
        clust_obj = Spec_Cluster(
            n_clusters=k,
            assign_labels="kmeans",
            random_state=1234,
            affinity="nearest_neighbors",
        )
        clust_obj.perform_sc(diary_obj.stat1, n_neighbors)
        labels = clust_obj.labels_

    # Convert labels to speaker boundaries
    subseg_ids = diary_obj.segset
    lol = []

    for i in range(labels.shape[0]):
        spkr_id = rec_id + "_" + str(labels[i])

        sub_seg = subseg_ids[i]

        splitted = sub_seg.rsplit("_", 2)
        rec_id = str(splitted[0])
        sseg_start = float(splitted[1])
        sseg_end = float(splitted[2])
        

        a = [rec_id, sseg_start, sseg_end, spkr_id]
        lol.append(a)

    # Sorting based on start time of sub-segment
    lol.sort(key=lambda x: float(x[1]))

    # Merge and split in 2 simple steps: (i) Merge sseg of same speakers then (ii) split different speakers
    # Step 1: Merge adjacent sub-segments that belong to same speaker (or cluster)
    lol = merge_ssegs_same_speaker(lol)

    # Step 2: Distribute duration of adjacent overlapping sub-segments belonging to different speakers (or cluster)
    # Taking mid-point as the splitting time location.
    lol = distribute_overlap(lol)

    # logger.info("Completed diarizing " + rec_id)
    write_rttm(lol, out_rttm_file)

def do_GMM(
    diary_obj, out_rttm_file, rec_id, k, affinity_type, n_neighbors
):
    """Performs spectral clustering on embeddings. This function calls specific
    clustering algorithms as per affinity.

    Arguments
    ---------
    diary_obj : StatObject_SB type
        Contains embeddings in diary_obj.stat1 and segment IDs in diary_obj.segset.
    out_rttm_file : str
        Path of the output RTTM file.
    rec_id : str
        Recording ID for the recording under processing.
    k : int
        Number of speaker (None, if it has to be estimated).
    pval : float
        `pval` for prunning affinity matrix.
    affinity_type : str
        Type of similarity to be used to get affinity matrix (cos or nn).
    """

    if affinity_type == "cos":
        clust_obj = Spec_Clust_unorm(min_num_spkrs=2, max_num_spkrs=10)
        k_oracle = k  # use it only when oracle num of speakers
        clust_obj.do_GMM(diary_obj.stat1)
        labels = clust_obj.labels_ 
    else:
        clust_obj = Spec_Cluster(
            n_clusters=k,
            assign_labels="kmeans",
            random_state=1234,
            affinity="nearest_neighbors",
        )
        clust_obj.perform_sc(diary_obj.stat1, n_neighbors)
        labels = clust_obj.labels_

    # Convert labels to speaker boundaries
    subseg_ids = diary_obj.segset
    lol = []

    for i in range(labels.shape[0]):
        spkr_id = rec_id + "_" + str(labels[i])

        sub_seg = subseg_ids[i]

        splitted = sub_seg.rsplit("_", 2)
        rec_id = str(splitted[0])
        sseg_start = float(splitted[1])
        sseg_end = float(splitted[2])
        

        a = [rec_id, sseg_start, sseg_end, spkr_id]
        lol.append(a)

    # Sorting based on start time of sub-segment
    lol.sort(key=lambda x: float(x[1]))

    # Merge and split in 2 simple steps: (i) Merge sseg of same speakers then (ii) split different speakers
    # Step 1: Merge adjacent sub-segments that belong to same speaker (or cluster)
    lol = merge_ssegs_same_speaker(lol)

    # Step 2: Distribute duration of adjacent overlapping sub-segments belonging to different speakers (or cluster)
    # Taking mid-point as the splitting time location.
    lol = distribute_overlap(lol)

    # logger.info("Completed diarizing " + rec_id)
    write_rttm(lol, out_rttm_file)

def do_mean_shift_clustering(
    diary_obj, out_rttm_file, rec_id, k, affinity_type, n_neighbors
):
    """Performs spectral clustering on embeddings. This function calls specific
    clustering algorithms as per affinity.

    Arguments
    ---------
    diary_obj : StatObject_SB type
        Contains embeddings in diary_obj.stat1 and segment IDs in diary_obj.segset.
    out_rttm_file : str
        Path of the output RTTM file.
    rec_id : str
        Recording ID for the recording under processing.
    k : int
        Number of speaker (None, if it has to be estimated).
    pval : float
        `pval` for prunning affinity matrix.
    affinity_type : str
        Type of similarity to be used to get affinity matrix (cos or nn).
    """

    if affinity_type == "cos":
        clust_obj = Spec_Clust_unorm(min_num_spkrs=2, max_num_spkrs=10)
        k_oracle = k  # use it only when oracle num of speakers
        clust_obj.do_Mean_Shift(diary_obj.stat1)
        labels = clust_obj.labels_ # Correct this
    else:
        clust_obj = Spec_Cluster(
            n_clusters=k,
            assign_labels="kmeans",
            random_state=1234,
            affinity="nearest_neighbors",
        )
        clust_obj.perform_sc(diary_obj.stat1, n_neighbors)
        labels = clust_obj.labels_

    # Convert labels to speaker boundaries
    subseg_ids = diary_obj.segset
    lol = []

    for i in range(labels.shape[0]):
        spkr_id = rec_id + "_" + str(labels[i])

        sub_seg = subseg_ids[i]

        splitted = sub_seg.rsplit("_", 2)
        rec_id = str(splitted[0])
        sseg_start = float(splitted[1])
        sseg_end = float(splitted[2])
        

        a = [rec_id, sseg_start, sseg_end, spkr_id]
        lol.append(a)

    # Sorting based on start time of sub-segment
    lol.sort(key=lambda x: float(x[1]))

    # Merge and split in 2 simple steps: (i) Merge sseg of same speakers then (ii) split different speakers
    # Step 1: Merge adjacent sub-segments that belong to same speaker (or cluster)
    lol = merge_ssegs_same_speaker(lol)

    # Step 2: Distribute duration of adjacent overlapping sub-segments belonging to different speakers (or cluster)
    # Taking mid-point as the splitting time location.
    lol = distribute_overlap(lol)

    # logger.info("Completed diarizing " + rec_id)
    write_rttm(lol, out_rttm_file)

def do_spec_clustering_asc(
    diary_obj, out_rttm_file, rec_id, k, pval, affinity_type, n_neighbors
):
    """Performs spectral clustering on embeddings. This function calls specific
    clustering algorithms as per affinity.

    Arguments
    ---------
    diary_obj : StatObject_SB type
        Contains embeddings in diary_obj.stat1 and segment IDs in diary_obj.segset.
    out_rttm_file : str
        Path of the output RTTM file.
    rec_id : str
        Recording ID for the recording under processing.
    k : int
        Number of speaker (None, if it has to be estimated).
    pval : float
        `pval` for prunning affinity matrix.
    affinity_type : str
        Type of similarity to be used to get affinity matrix (cos or nn).
    """

    if affinity_type == "cos":
        clust_obj = Spec_Clust_unorm(min_num_spkrs=2, max_num_spkrs=10)
        k_oracle = k  # use it only when oracle num of speakers
        clust_obj.do_spec_clust_asc(diary_obj.stat1, k_oracle, pval)
        labels = clust_obj.labels_
    else:
        clust_obj = Spec_Cluster(
            n_clusters=k,
            assign_labels="kmeans",
            random_state=1234,
            affinity="nearest_neighbors",
        )
        clust_obj.perform_sc(diary_obj.stat1, n_neighbors)
        labels = clust_obj.labels_

    # Convert labels to speaker boundaries
    subseg_ids = diary_obj.segset
    lol = []

    for i in range(labels.shape[0]):
        spkr_id = rec_id + "_" + str(labels[i])

        sub_seg = subseg_ids[i]

        splitted = sub_seg.rsplit("_", 2)
        rec_id = str(splitted[0])
        sseg_start = float(splitted[1])
        sseg_end = float(splitted[2])
        

        a = [rec_id, sseg_start, sseg_end, spkr_id]
        lol.append(a)

    # Sorting based on start time of sub-segment
    lol.sort(key=lambda x: float(x[1]))

    # Merge and split in 2 simple steps: (i) Merge sseg of same speakers then (ii) split different speakers
    # Step 1: Merge adjacent sub-segments that belong to same speaker (or cluster)
    lol = merge_ssegs_same_speaker(lol)

    # Step 2: Distribute duration of adjacent overlapping sub-segments belonging to different speakers (or cluster)
    # Taking mid-point as the splitting time location.
    lol = distribute_overlap(lol)

    # logger.info("Completed diarizing " + rec_id)
    write_rttm(lol, out_rttm_file)




def do_kmeans_clustering(
    diary_obj, out_rttm_file, rec_id, k_oracle=4, p_val=0.3
):
    """Performs kmeans clustering on embeddings.

    Arguments
    ---------
    diary_obj : StatObject_SB type
        Contains embeddings in diary_obj.stat1 and segment IDs in diary_obj.segset.
    out_rttm_file : str
        Path of the output RTTM file.
    rec_id : str
        Recording ID for the recording under processing.
    k : int
        Number of speaker (None, if it has to be estimated).
    pval : float
        `pval` for prunning affinity matrix. Used only when number of speakers
        are unknown. Note that this is just for experiment. Prefer Spectral clustering
        for better clustering results.
    """

    if k_oracle is not None:
        num_of_spk = k_oracle
    else:
        # Estimate num of using max eigen gap with `cos` affinity matrix.
        # This is just for experimentation.
        # Not doing full spectral clustering. Just re-using the code till
        # estimating num of speakers.
        clust_obj = Spec_Clust_unorm(min_num_spkrs=2, max_num_spkrs=10)

        # clust_obj.do_spec_clust(diary_obj.stat1, k_oracle, pval)
        # labels = clust_obj.labels_

        # Get sim matrix
        sim_mat = clust_obj.get_sim_mat(diary_obj.stat1)
        pruned_sim_mat = clust_obj.p_pruning(sim_mat, p_val)

        # Symmetrization
        sym_pruned_sim_mat = 0.5 * (pruned_sim_mat + pruned_sim_mat.T)

        # Laplacian calculation
        laplacian = clust_obj.get_laplacian(sym_pruned_sim_mat)

        # Get Spectral Embeddings
        _, num_of_spk = clust_obj.get_spec_embs(laplacian, k_oracle)

    # Perform kmeans directly on deep embeddings
    _, labels, _ = k_means(diary_obj.stat1, num_of_spk)

    # Convert labels to speaker boundaries
    subseg_ids = diary_obj.segset
    lol = []

    for i in range(labels.shape[0]):
        spkr_id = rec_id + "_" + str(labels[i])

        sub_seg = subseg_ids[i]

        splitted = sub_seg.rsplit("_", 2)
        rec_id = str(splitted[0])
        sseg_start = float(splitted[1])
        sseg_end = float(splitted[2])

        a = [rec_id, sseg_start, sseg_end, spkr_id]
        lol.append(a)

    # Sorting based on start time of sub-segment
    lol.sort(key=lambda x: float(x[1]))

    # Merge and split in 2 simple steps: (i) Merge sseg of same speakers then (ii) split different speakers
    # Step 1: Merge adjacent sub-segments that belong to same speaker (or cluster)
    lol = merge_ssegs_same_speaker(lol)

    # Step 2: Distribute duration of adjacent overlapping sub-segments belonging to different speakers (or cluster)
    # Taking mid-point as the splitting time location.
    lol = distribute_overlap(lol)

    # logger.info("Completed diarizing " + rec_id)
    write_rttm(lol, out_rttm_file)


def do_AHC(diary_obj, out_rttm_file, rec_id, k_oracle=4, p_val=0.3):
    """Performs Agglomerative Hierarchical Clustering on embeddings.

    Arguments
    ---------
    diary_obj : StatObject_SB type
        Contains embeddings in diary_obj.stat1 and segment IDs in diary_obj.segset.
    out_rttm_file : str
        Path of the output RTTM file.
    rec_id : str
        Recording ID for the recording under processing.
    k : int
        Number of speaker (None, if it has to be estimated).
    pval : float
        `pval` for prunning affinity matrix. Used only when number of speakers
        are unknown. Note that this is just for experiment. Prefer Spectral clustering
        for better clustering results.
    """

    from sklearn.cluster import AgglomerativeClustering

    # p_val is the threshold_val (for AHC)
    # Normalizing embeddings.
    diary_obj.norm_stat1()

    # processing
    if k_oracle is not None:
        num_of_spk = k_oracle

        clustering = AgglomerativeClustering(
            n_clusters=num_of_spk, affinity="cosine", linkage="ward",
        ).fit(diary_obj.stat1)
        labels = clustering.labels_

    else:
        # Estimate num of using max eigen gap with `cos` affinity matrix.
        # This is just for experimentation.
        clustering = AgglomerativeClustering(
            n_clusters=None,
            # Here I am replacing affinity with metric
            affinity="cosine", # Original Configuration
            #metric="cosine",
            # Here I am replacing the linkage "ward" with linkage='average' as it does not support the euclidea distance
            linkage="ward", # Original Configuration
            #linkage="average",
            distance_threshold=p_val,
        ).fit(diary_obj.stat1)
        labels = clustering.labels_

    # Convert labels to speaker boundaries
    subseg_ids = diary_obj.segset
    lol = []

    for i in range(labels.shape[0]):
        spkr_id = rec_id + "_" + str(labels[i])

        sub_seg = subseg_ids[i]

        splitted = sub_seg.rsplit("_", 2)
        rec_id = str(splitted[0])
        sseg_start = float(splitted[1])
        sseg_end = float(splitted[2])

        a = [rec_id, sseg_start, sseg_end, spkr_id]
        lol.append(a)

    # Sorting based on start time of sub-segment
    lol.sort(key=lambda x: float(x[1]))

    # Merge and split in 2 simple steps: (i) Merge sseg of same speakers then (ii) split different speakers
    # Step 1: Merge adjacent sub-segments that belong to same speaker (or cluster)
    lol = merge_ssegs_same_speaker(lol)

    # Step 2: Distribute duration of adjacent overlapping sub-segments belonging to different speakers (or cluster)
    # Taking mid-point as the splitting time location.
    lol = distribute_overlap(lol)

    # logger.info("Completed diarizing " + rec_id)
    write_rttm(lol, out_rttm_file)
    
'''    
# Begin experiment!
if __name__ == "__main__":  # noqa: C901
    stat_file = 'test.pkl'
    with open(stat_file, "rb") as in_file:
        stat_obj = pickle.load(in_file)
    
    
    clust_obj = Spec_Clust_unorm(min_num_spkrs=2, max_num_spkrs=10)
    k_oracle = None  # use it only when oracle num of speakers
    pval = 0.01
    clust_obj.do_spec_clust(stat_obj.stat1, k_oracle, pval)
    labels = clust_obj.labels_
    print(labels[0:-1])
'''    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


