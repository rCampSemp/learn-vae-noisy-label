# A Probabilistic Deep Learning Approach to Reconstructing Ground Truths in Medical Image Segmentation #

├── adamW.py
├── hyperparam-optim
│   ├── hyper-optim-deterministic.ipynb
│   └── hyper-optim-stochastic.ipynb
├── image-analysis.ipynb
├── Legacy-code
│   ├── conda_env.yml
│   ├── data_simulation
│   │   ├── artificial_wrong_mask.py
│   │   ├── multiclass_data_simulator.m
│   │   ├── over_segmemtation.m
│   │   └── under_segmentation.m
│   ├── Deterministic_CM.py
│   ├── Deterministic_LIDC.py
│   ├── Deterministic_Loss.py
│   ├── Deterministic_MNIST.py
│   ├── figures
│   │   ├── Brats_1.jpg
│   │   ├── Brats_2.jpg
│   │   ├── brats-compare.jpg
│   │   ├── consensus_1.jpg
│   │   ├── consensus_2.jpg
│   │   ├── deterministicLIDC-hyperoptim.png
│   │   ├── figure_thumbnail.001.png
│   │   ├── horizontal8.png
│   │   ├── horizontal.png
│   │   ├── humanerror_2.png
│   │   ├── lidc-annot-masks.png
│   │   ├── LIDC-compare.jpg
│   │   ├── lidc-graph.png
│   │   ├── LIDC.jpg
│   │   ├── lidc-scan.png
│   │   ├── mask_threshold.png
│   │   ├── mnist_ex.png
│   │   ├── MNIST.jpg
│   │   ├── Morph.png
│   │   ├── MS.jpg
│   │   ├── Multi-class.png
│   │   ├── NIPS.png
│   │   ├── over-sample.png
│   │   ├── samplesMNISTex.png
│   │   ├── under-sample.png
│   │   └── wrong-sample.png
│   ├── legacy_utils.py
│   ├── preprocessing
│   │   ├── Prepare_BRATS_noisy_label.py
│   │   └── Prepare_BRATS.py
│   ├── Segmentation.py
│   ├── Train_GCM.py
│   ├── Train_LIDC.py
│   ├── Train_ours.py
│   └── Train_unet.py
├── LIDC_examples
│   ├── meta
│   │   └── metadata.csv
│   ├── test
│   │   ├── masks
│   │   └── scans
│   ├── train
│   │   ├── masks
│   │   └── scans
│   └── validate
│       ├── masks
│       └── scans
├── MNIST_examples
│   ├── test
│   │   ├── Gaussian
│   │   ├── GT
│   │   ├── Over
│   │   ├── Under
│   │   └── Wrong
│   ├── train
│   │   ├── All
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
├── Models.py
├── preprocessing
│   ├── Prepare_LIDC.py
│   └── Prepare_MNIST.py
├── README.md
├── Run.py
├── Stochastic_CM.py
├── Stochastic_LIDC.py
├── Stochastic_Loss.py
├── Stochastic_MNIST.py
├── Train_punet.py
├── Train_VAE.py
└── Utilis.py
