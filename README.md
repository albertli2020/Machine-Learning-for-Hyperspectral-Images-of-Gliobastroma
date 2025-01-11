# Hyperspectral Imaging (HSI) Tumor Classifier

## Progressive Neural Network - HSI-2DCNN-8LoP.py

### Overview

This program implements an approach for tumor classification using HSI data centered around 2D CNNs. Similar to Progressive GANs, it starts with a simple convolutional neural network and gradually adds complexity by introducing new layers during training, with each deeper network inheriting parameters from simpler versions. Every new layer added inherits parameters from simpler networks to initialize deeper ones. The network processes HSI patches of size 87x87 pixels with 275 spectral bands, and can grow up to 8 layers deep. It also includes an evaluation  pipeline with metrics like accuracy, sensitivity, and specificity.

### Training Process

1. Starts from scratch with a single-layer network
2. Progressively adds layers while inheriting previous parameters
3. Uses different learning rates for different network depths, decreasing as more layers are added
4. Saves the model based on the validation accuracy

### Input Data Structure

Input data is organized organized by patient (P1-P13)
Patches are preprocessed from cubes, each cube being split into 99 patches
Patches are labeled as tumor (T) or non-tumor (NT) based on the label of the ROI
Each HSI patch has 275 spectral bands, down from 826, averaged from the 2 neighboring bands

### Requirements

* PyTorch
* NumPy
* scikit-learn
* matplotlib
* CUDA or MPS compatible system (compatible with both CUDA (NVIDIA) and MPS (Apple Metal) backends)

## Partition by ROI instead of by patient - Partition_Per_ROI.py

Shuffles the ROIs in the dataset, skipping Patient 6, and then iterates through them seperating into two sets, a training set and a validation and testing set. The split is about 70% (about 29,000 patches) training and 30% (about 12,000 patches) validation and testing. 
