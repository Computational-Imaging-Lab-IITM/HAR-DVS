# HAR-DVS
This project is a part of our work, _"Dynamic Vision Sensors for Human Activity Recognition" - Stefanie Anna Baby, Bimal Vinod, Chaitanya Chinni, Kaushik Mitra_,
accepted at _the 4th IAPR Asian Conference on Pattern Recognition (ACPR) 2017_.

The links to our work can be found at [IEEE][ieee_link], [arXiv][arxiv_link] and [webpage][lab_page].

## IITM DVS128 Gesture Dataset
The **IITM DVS128 Gesture Dataset** contains 10 hand gestures by 12 subjects with 10 sets each totalling to a 1200 hand gestures.
These gestures are captured using a [DVS128 camera][dvs128_link].

The `aedat` dataset and the corresponding converted `avi` dataset can be downloaded from [IITM_DVS_10][IITM_DVS_10_link].

## Folder structure
    .
    ├── lib
    │   ├── dense_trajectory_release_v1.2
    │   │   ├── ...
    │   │   └── ...
    │   ├── bov_encode.m
    │   ├── generate_codebook.m
    │   ├── generate_motion_maps.m
    │   ├── groupfile_indices.m
    │   ├── groupfile_indices_mm.m
    │   ├── hidden_indices.m
    │   ├── normalize_image.m
    │   ├── parsave.m
    │   ├── svm_loo.m
    │   └── run_dense.sh
    ├── README.md
    ├── extract_features.m
    ├── run_dvs.m
    └── startup.m

The `lib` folder contains a bunch of utility snippets used by the main code files.

The `lib/dense_trajectory_release_v1.2` code is taken from http://lear.inrialpes.fr/people/wang/dense_trajectories which was from the work by _"Action Recognition by Dense Trajectories" by Wang et al._.
More details can be found in the `README` file associated with it.

## Adding the datasets
The `data` folder should contain all the data in the root directory with the following structure. The extracted DVS data should be put in `data/<dataset_name>/original_data` with individual folders for each class.

The code expects `.avi` files and **not** `aedat`. All the features (motion-maps and dense-trajectories) are extracted by `extract_features.m` into the `data/<dataset_name>/features_extracted` folder.

The `run_dvs.m` program automatically segregates the `test` and `train` data into `data/<dataset_name>/encoded_data` for **K-fold cross-validation**.

The same structure is followed for any other dataset. The `<dataset_name>` can be `IITM_DVS_10` or `UCF11_DVS` etc.

    .
    ├── data
    │   ├── IITM_DVS_10
    │   │   ├── original_data
    │   │   │   ├── comeHere
    │   │   │   │   └── *.avi
    │   │   │   ├── left_swipe
    │   │   │   │   └── *.avi
    │   │   │   └── ...
    │   │   ├── features_extracted
    │   │   │   └── ...
    │   │   └── encoded_data
    │   │   │   ├── test
    │   │   │   └── train
    │   ├── UCF11_DVS
    │   │   ├── ...
    │   │   └── ...
    │   └── ...
    └── ...

## Citation
If you find our work helpful in your publications or projects, please consider citing our paper.

S. A. Baby and B. Vinod and C. Chinni and K. Mitra, "Dynamic Vision Sensors for Human Activity Recognition,"
2017 4th IAPR Asian Conference on Pattern Recognition (ACPR), Nanjing, China, 2017.

    @inproceedings{sababy:hardvs:2017,
        author={S. A. {Baby} and B. {Vinod} and C. {Chinni} and K. {Mitra}},
        booktitle={2017 4th IAPR Asian Conference on Pattern Recognition (ACPR)},
        title={Dynamic Vision Sensors for Human Activity Recognition},
        year={2017},
        pages={316-321},
        doi={10.1109/ACPR.2017.136},
        ISSN={2327-0985},
        month={Nov}
    }

## Credits
Thanks to **Heng Wang** for the dense-trajectories code - http://lear.inrialpes.fr/people/wang/dense_trajectories

[ieee_link]: https://ieeexplore.ieee.org/document/8575843
[arxiv_link]: https://arxiv.org/abs/1803.04667
[lab_page]: http://www.ee.iitm.ac.in/comp_photolab/project-dynamic-vision-sensors-human-activity-recognition.html
[dvs128_link]: https://inivation.com/support/hardware/dvs128/
[IITM_DVS_10_link]: https://www.dropbox.com/sh/ppjgszi51no884f/AACb3xMxmftHD3P3XqpahIJOa?dl=0
