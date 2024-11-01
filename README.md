# Overview of LLOKI

<img width="649" alt="Screenshot 2024-11-01 at 12 34 29 AM" src="https://github.com/user-attachments/assets/5ab3fa0f-91c7-428e-9085-25df9e1cf321">

LLOKI is a novel framework for scalable spatial transcriptomics (ST) integration across diverse technologies without requiring shared gene panels. The framework comprises two key components: 
LLOKI-FP, which leverages optimal transport and feature propagation to perform a spatially informed transformation of ST gene expression profiles, aligning their sparsity with that of scRNA-seq to optimize the utility of scGPT embeddings; and LLOKI-CAE, a conditional autoencoder that integrates embeddings across ST technologies using a novel loss function that balances batch integration with the preservation of robust biological information from the LLOKI-FP embeddings. This unique combination ensures alignment of both features and batches, enabling robust ST data integration while preserving biological specificity and local spatial interactions.

# Running LLOKI
# Input Data Format

LLOKI requires spatial transcriptomics data to be provided as AnnData objects, structured as follows:

Spatial Coordinates:
The spatial coordinates of each cell should be included in the .obsm attribute of the AnnData object.
The coordinates must be stored under .obsm['spatial'] and formatted as an array with dimensions [number of cells, 2] (representing x and y coordinates for each cell)

Gene Expression Data:
Gene expression data should be stored in .X as a sparse matrix (recommended for large datasets) or a dense matrix, with dimensions [number of cells, number of genes]

While not required, any additional metadata (e.g., cell types, batch labels) can be stored in .obs

# Installation

## Step 1: Create a Conda Environment

Before installing any Python packages, we strongly recommend using Anaconda (please refer to the Anaconda webpage for conda installation instructions) to create a python 3.8 environment using the following command:

conda install --name lloki python=3.8

After creating the environment, activate it using:

conda activate lloki

## Step 2: Install Dependencies

Install PyTorch with CUDA (optional)

If you have an NVIDIA GPU and want to use CUDA for acceleration, install PyTorch with the desired CUDA version. For example, to install PyTorch 2.1.0 with CUDA 11.8, run:

conda install pytorch==2.1.0 cudatoolkit=11.8 -c pytorch

Note: For a CPU-only installation, you can omit the cudatoolkit argument.

## Install Remaining Dependencies

You can install all other necessary packages using the requirements.txt file included in the project:

pip install -r requirements.txt

# Running Code

To run our method, you need to run the pipeline in two parts: first LLOKI-FP and then LLOKI-CAE. 

To run our code for LLOKI-FP, you first want to run LLOKI_scgpt.py, which takes as input your ST anndata object. 

Next, LLOKI-CAE uses the output cell embeddings from LLOKI-FP as cell features in the conditional autoencoder. We have a tutorial notebook CondAutoencoder.ipynb you can follow to integrate the output LLOKI-FP embeddings for different ST slices in order to perform slice integration. 


