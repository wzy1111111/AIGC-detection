# Detecting fake images

**** <br>Semantic Discrepancy-aware Detector for Image Forgery Identification<br>

## Contents

- [Setup](#setup)
- [Data](#data)
- [Evaluation](#evaluation)
- [Training](#training)


## Setup 

1. Clone this repository 

2. Install the necessary libraries
```bash
pip install torch torchvision
```
3. 
## Data

- Of the 19 models studied overall (Table 1/2 in the main paper), 11 are taken from a [previous work](https://arxiv.org/abs/1912.11035). Download the test set, i.e., real/fake images for those 11 models given by the authors from [here](https://drive.google.com/file/d/1z_fD3UKgWQyOTZIBbYSaQ-hz4AzUrLC1/view) (dataset size ~19GB).
- Download the file and unzip it in `datasets/test`. You could also use the bash scripts provided by the authors, as described [here](https://github.com/PeterWang512/CNNDetection#download-the-dataset) in their code repository.
- This should create a directory structure as follows:
```

datasets
└── test					
      ├── progan	
      │── cyclegan   	
      │── biggan
      │      .
      │      .
	  
```
- Each directory (e.g., progan) will contain real/fake images under `0_real` and `1_fake` folders respectively.
- Dataset for the diffusion models (e.g., LDM/Glide) can be found [here](https://drive.google.com/file/d/1FXlGIRh_Ud3cScMgSVDbEWmPDmjcrm1t/view?usp=drive_link).
- Download and unzip the file into `./diffusion_datasets` directory.


## Evaluation
 
- You can evaluate the model on all the dataset at once by running:
```bash
python test.py   
```


Note that if no arguments are provided for `real_path` and `fake_path`, the script will perform the evaluation on all the domains specified in `dataset_paths.py`.

- The results will be stored in `results/<folder_name>` in two files: `ap.txt` stores the Average Prevision for each of the test domains, and `acc.txt` stores the accuracy (with 0.5 as the threshold) for the same domains.

## Training

- Our main model is trained on the same dataset used by the authors of [this work](https://arxiv.org/abs/1912.11035). Download the official training dataset provided [here](https://drive.google.com/file/d/1iVNBV0glknyTYGA9bCxT_d0CVTOgGcKh/view) (dataset size ~ 72GB). 

- Download and unzip the dataset in `datasets/train` directory. The overall structure should look like the following:
```
datasets
└── train			
      └── progan			
           ├── airplane
           │── bird
           │── boat
           │      .
           │      .
```
- A total of 20 different object categories, with each folder containing the corresponding real and fake images in `0_real` and `1_fake` folders.
- The model can then be trained with the following command:
```bash
python train.py  --blur_prob 0.1 --blur_sig 0.0,3.0 --jpg_prob 0.1 --jpg_method cv2,pil --jpg_qual 30,100 --detect_method LOOD --dataroot /workspace/datasets/progan
```
- **Important**: do not forget to use the `--fix_backbone` argument during training, which makes sure that the only the linear layer's parameters will be trained.

## Acknowledgement

We would like to thank [Sheng-Yu Wang](https://github.com/PeterWang512) and [Utkarsh Ojha*](https://utkarshojha.github.io/) for releasing the real/fake images from different generative models. Our training pipeline is also inspired by his [open-source code](https://github.com/WisconsinAIVision/UniversalFakeDetect).
## Citation


