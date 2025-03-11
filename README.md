# Self-Tuning Spectral Clustering for Speaker Diarization 
### About
This repository contains the implementation for the paper: 
- "Raghav, Nikhil and Gupta, Avisek and Sahidullah, Md and Das, Swagatam, "Self-Tuning Spectral Clustering for Speaker Diarization", to appear in Proc. of ICASSP 2025.
The details of the technique can be found [here]([https://arxiv.org/pdf/2410.00023?](https://ieeexplore.ieee.org/abstract/document/10890194) "paper link")

## Dependencies
Our implementation is based on a modified version of the AMI recipe provided in the SpeechBrain toolkit.
- Follow the installation guidelines for the SpeechBrain toolkit provided [here](https://github.com/speechbrain/speechbrain "SpeechBrain toolkit link") 
## Method
The following three files were modified from the exisitng AMI recipe, and were adapted for the experiments on the DIHARD-III dataset. It contains, the scripts for the proposed SC-pNA technique:
- experiment.py located at /speechbrain/recipes/AMI/Diarization/experiment.py
- ecapa_tdnn.yaml located at /speechbrain/recipes/AMI/Diarization/ecapa_tdnn.yaml
- diarization.py located at /speechbrain/speechbrain/processing/diarization.py

## Citation
If you find our approach useful in your research, please consider citing:

```
@article{raghav2024self,
  title={Self-Tuning Spectral Clustering for Speaker Diarization},
  author={Raghav, Nikhil and Gupta, Avisek and Sahidullah, Md and Das, Swagatam},
  journal={arXiv preprint arXiv:2410.00023},
  year={2024}
}
```
## License
This project is licensed under the MIT License. The full terms of the MIT License can be found in the LICENSE.md file at the root of this project.

