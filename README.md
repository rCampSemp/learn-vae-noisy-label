# A Probabilistic Deep Learning Approach to Reconstructing Ground Truths in Medical Image Segmentation #

## Introduction ## 
This repo holds code for my MSc thesis. I provide a joint CNN-VAE model which simultaneously learns a ground truth from noisy labels only and confusion matrices to model and remove annotator noise from an aggregation of all input noisy labels. The VAE part of our model allows us to generate different confusion matrices indefinitely. For a binary segmentation task

Extension of work by [Mou-Cheng Xu](https://moucheng2017.github.io/) from [Other repo](https://github.com/moucheng2017/Learn_Noisy_Labels_Medical_Images).

### data ### 

We use MNIST and LIDC-IDRI datasets generated from the scripts found in /preprocessing. 

Morphological operations using OpenCV are used to generate synthetic annotator types in from the MNIST dataset.

![our-mnist](figures/our-mnist.png)

LIDC-IDRI dataset is prepared using Prepare_LIDC.py. This requires the user to download the data from the official LIDC-IDRI website. 

[LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI "Main webpage of LIDC-IDRI dataset")

Below is an example of the LIDC data.

![LIDC](figures/lidc-inputs.png)

Use the correct file path to your LIDC download then run Prepare_LIDC.py. 

### Requirements ###
Packages can be found in exported environment.yml file.

## Model Structure ##
![our-model](figures/ourmodel(1).png)

## How to run ## 
Use Run.py to run both probabilistuc unet and our model

## Repo structure ##
```bash
├── adamW.py  # optimizer
├── environment.yml   # conda environment file containing package requirements and dependencies
├── figures 
│   ├── full-lidc-ims.png
│   ├── lidc-inputs.png
│   ├── our-mnist.png
│   ├── ourmodel(1).png
│   └── recon_cm.png
├── hyperparam-optim  # hyperparameter search
│   └── hyper-optim-stochastic.ipynb
├── image-analysis.ipynb  # LIDC image analysis
├── LICENSE.md
├── LIDC_examples
│   ├── meta
│   │   └── metadata.csv    # metadata of lidc data
│   ├── test
│   │   ├── masks   # binary masks with noisy annotation labels
│   │   └── scans   # cropped normalised CT scans
│   ├── train
│   │   ├── masks
│   │   └── scans
│   └── validate
│       ├── masks
│       └── scans
├── MNIST_examples
│   ├── test
│   │   ├── Gaussian    # input labels
│   │   ├── GT          # Noisy labels (good-segmentation)
│   │   ├── Over        # Noisy labels (over-segmented)
│   │   ├── Under       # Noisy labels (under-segmented)
│   │   └── Wrong       # Noisy labels (wrong-segmentation)
│   ├── train
│   │   ├── All # all label types (manually created)
│   │   ├── Gaussian
│   │   ├── GT
│   │   ├── Over
│   │   ├── Under
│   │   └── Wrong
│   └── validate
│       ├── Gaussian
│       ├── GT
│       ├── Over
│       ├── Under
│       └── Wrong
├── preprocessing   # # used for generating our datasets
│   ├── Prepare_LIDC.py
│   └── Prepare_MNIST.py
├── punet_Model.py  # contains probabilistic unet model
├── README.md
├── Run.py  # main hub to run all models 
├── Stochastic_CM.py    # our proposed model
├── Stochastic_LIDC.py  # example script for training LIDC
├── Stochastic_Loss.py  # proposed loss function 
├── Stochastic_MNIST.py # example script for training MNIST
├── Train_punet.py      # script for probabilstic unet
├── Train_VAE.py        # script for our proposed model
└── Utilis.py           # contains main functions for training

```
## hyperparameter search
We also provide a hyperprameter search scheme using optuna nad mlflow for tracking in hyperparam-optim folder.

## Example qualitative results

LIDC
![lidcresults](figures/full-lidc-ims.png)

MNIST
![mnistresults](figures/recon_cm.png)
