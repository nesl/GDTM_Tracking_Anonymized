

# GDTM-Tracking

**GDTM** is a new multi-hour dataset collected with a network of multimodal sensors for the indoor geospatial tracking problem. It features time-synchronized steoreo-vision camera, LiDAR camera, mmWave radar, and microphone arrays, as well as ground truth data containing the position and orientations of the sensing target (remote controlled cars on a indoor race track) and the sensor nodes. For details of the dataset please refer to [GitHub](https://anonymous.4open.science/r/GDTM_Anonymized-4469/README.md) and PDF (still under review).

This repository contains our baseline applications described in PDF (still under review) built to use GTDM data. It features two architectures (early fusion and late fusion and two choices of sensor sets (camera only and all-modalities) to track the locations of a target RC car.

**Note** for dataset documentation and pre-processing, please refer to [GitHub](https://anonymous.4open.science/r/GDTM_Anonymized-4469/README.md).



## Installation Instuctions

### Environment
The code is tested with:
Ubuntu 20.04
Anaconda 22.9.0 (for virtual python environment)
NVIDIA-driver 525.105.17
The code should be compatible with most Anaconda, NVIDIA-driver, and Ubuntu versions available around 2023/06.

### Code Repository Structure
We only release the early fusion, all modalities version of the model. Further variants will be released upon acceptance. Details are described in **Baseline 1** section of PDF (Still under review).



As step one, please clone the desired branch using terminal. It is not possible to clone the anonymous repo, and these instructions will be updated before the camera-ready. 
```
cd ~/Desktop
git clone https://anonymous.4open.science/r/GDTM_Anonymized-4469.git
```
or
```
cd ~/Desktop
git clone --branch <branchname> https://anonymous.4open.science/r/GDTM_Anonymized-4469.git
```

### Install Dependencies
First, place the repository folder on Desktop and rename it "mmtracking".
```
mv <path-to-cloned-repository> ~/Desktop/mmtracking
```
Create a new conda environment using
```
cd ~/Desktop/mmtracking
conda create -n iobt python=3.9
conda activate iobt
```
Install a few torch and mmcv using pip:
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
```
Install other dependencies:
```
pip  install  -r  requirements/build.txt
```
Install a few local packages (the terminal should still be in ~/Desktop/mmtracking):
```
pip install -e src/cad
pip install -e src/mmdetection
pip install -e src/TrackEval
pip install -e src/mmclassification
pip install -e src/resource_constrained_tracking
pip install -e src/yolov7
pip install  -v  -e  .
```

## Data Preparation
#### Sample dataset
Please visit the [data repository](https://drive.google.com/drive/folders/1ccdCBq1Xh9tW6CHoGbzizIo79T18nMfo?usp=drive_link) for sample data to test this repository. Due to constraints of uploading data to an anonymous google drive, we have only provided two instances of the data, good lighting (view 3) and poor lighting (view 6) under single-view and all modality conditions, for only the test data. 

#### Full dataset
We are going to release the full dataset on a later date. Check for updates at [GitHub](https://anonymous.4open.science/r/GDTM_Anonymized-4469/README.md).

#### Unzip the data

Please unzip the data, rename it to **"mcp-sample-dataset/"** and put it on the Desktop.
The final data structure should be like following:
```
└── Desktop/mcp-sample-dataset/
	├── test/
	│   ├── node1/       
	│   │   ├── mmwave.hdf5
	│   │   ├── realsense.hdf5
	│   │   ├── respeaker.hdf5
	│   │   └── zed.hdf5
	│   ├── node2/
	│   │   └── same as node 1
	│   ├── node3/
	│   │   └── same as node 1  
	│   └── mocap.hdf5
	├── train/
	│   └── same as test
	└── val/
	    └── same as test
```
Note that you only need test/ if you are running test from checkpoints only.
#### Specify Filepath
Open mmtracking/configs/\_base\_/datasets/one_car_early_fusion.py
In Line 75, Line 114, and Line 153, change the data_root to absolute path:
e.g. ~/Desktop/... -> /home/USER_NAME/Desktop/...

## Code Usage
### Make Inference Using A Checkpoint (Testing)
Please download the pretrained checkpoints [here](https://drive.google.com/drive/folders/17RXrXc4Qd0yKhGVMyyZxVxF9NrwolRgn?usp=sharing).

Note that for single-view case (Baseline 1 in the paper), please make sure to use the checkpoints corresponding to the code and data of your choice. 

For example, if we use view 3 data (single view, good lighting condition) and master branch code (single view, early fusion, all modalities), we should download "dataset_singleview3.zip". 

After downloading the checkpoint, please rename it to **logs/** and put it under "mmtracking" folder using this hierachy. 

```
└── Desktop/mmtracking/
    └── logs/
        └── early_fusion_zed_mmwave_audio/
            ├── val
            ├── epoch_xx.pth
            └── latest.pth (to be created)
```
where the "latest.pth" above is created by (in a terminal in early\_fusion\_zed\_mmwave\_audio\/):
```
ln -s epoch_40.pth latest.pth
```

Then, you could run the evaluations by running (still in terminal under ~/Desktop/mmtracking, make sure you have used "conda activate iobt")
```
bash ./tools/test_from_config_nll_local.sh ./configs/mocap/early_fusion_zed_mmwave_audio.py 1
```
---
**Warning**: This script will cache the dataset in system memory (/dev/shm)
If the dataset loading operation was not successful, or you have changed the dataset in "~/Desktop/mcp-sample-dataset", please make sure to run this line **before** the "test_from_config_nll_local.sh" above:
```
rm -r /dev/shm/cache_*
```
---


The visualization results will apprear in 
```
mmtracking/logs/early_fusion_early_fusion_zed_mmwave_audio/test_nll/latest_vid.mp4
```
and numerical results appears at the last two lines of 
```
mmtracking/logs/early_fusion_early_fusion_zed_mmwave_audio/test_nll/mean.txt
```

If you would like to train a model from scratch instead , please refer to the “training” and “scaling” sections down below.

### Training
Set up the data as instructed by previous sections, and run
```
bash ./tools/train_from_config_local.sh ./configs/mocap/early_fusion_zed_mmwave_audio.py 1
```
where the last digit indicate the number of GPU you have for training.

### Scaling
After training, some additional data is required to perform a post-hoc model recalibration as described in the paper to better capture model prediction uncertainties. More specifically, We apply an affine transformation Σ′ = aΣ + bI to the output covariance matrix Σ with parameters a and b that minimize the calibration data’s NLL. 

Instructions for scaling:
```
bash ./tools/val_from_config_local.sh ./configs/mocap/early_fusion_zed_mmwave_audio.py 1
```
The last digit must be "1". Scaling with multiple GPU will cause an error.


## Troubleshooting

Here we list a few files to change in case some error happens during your configurations.
#### Data not found error
This is where the filepath are stored
mmtracking/configs/\_base\_/datasets/one_car_early_fusion.py

Don't forget to do "rm -r /dev/shm/cache_*" after you fix this error. Otherwise a "List out of range" error will pop up.

#### GPU OOM Error, Number of Epoches, Inteval of checkpoints
mmtracking/configs/mocap/early_fusion_zed_mmwave_audio.py
Reduce "samples_per_gpu" in Line 127 helps with OOM error.
Line 169-187 changes the training configurations.

This configuration also defines (1) the valid modalities (2) backbone, adapter, and output head architecture hyperparameters

#### Something wrong with dataset caching
mmtracking/mmtrack/datasets/mocap/cacher.py

#### Something wrong with model training/inferences
mmtracking/mmtrack/models/mocap/early_fusion.py
Function forward_train() for training
Fuction forward_track() for testing

#### Something wrong with final visualzations
mmtracking/mmtrack/datasets/mocap/hdf5_dataset.py
in function write_videos()

#### Backbone definitions
mmtracking/mmtrack/models/backbones/tv_r50.py

## Citation and Acknowledgements

@misc{mmtrack2020,
    title={{MMTracking: OpenMMLab} video perception toolbox and benchmark},
    author={MMTracking Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmtracking}},
    year={2020}
}
```

