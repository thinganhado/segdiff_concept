# SegDiff Concept Workflow (ASVspoof2019)

This repository contains tools and training scripts for running SpecSegDiff-style experiments on the ASVspoof2019 dataset. The general workflow is as follows:

## Installation
### Conda environment
To create the environment use the conda environment command
```
conda env create -f environment.yml
```

## 1. Generate Input Masks

To begin, generate the required mask data for training using the ASVspoof2019 dataset.

```bash
python tools/masks/make_masks_asv2019.py

The output should have the following format
```
voc.v4/
    Test/
        img/
            XX.tif
        mask/
            XX.png
    Training/
        img/
            XX.tif
        mask/
            XX.png
```

## Train and Evaluate
Execute the following commands (multi gpu is supported for training, set the gpus with CUDA_VISIBLE_DEVICES and -n for the actual number)

Training options:
```
# Training
--batch-size    Batch size
--lr            Learning rate

# Architecture
--rrdb_blocks       Number of rrdb blocks
--dropout           Dropout
--diffusion_steps   number of steps for the diffusion model

```

### MonuSeg
Training script example:
```
python image_train_diff_medical.py --data_dir /home/opc/SegDiff/data/voc.v4 --rrdb_blocks 12 --batch_size 2 --lr 0.0001 --diffusion_steps 100

```

Evaluation script example:
```
python image_sample_diff_medical.py --model_path path-for-model-weights

```

## 3. Crop and Concept Tools

The following directory contains scripts related to crop generation and concept detection: tools/crops/
