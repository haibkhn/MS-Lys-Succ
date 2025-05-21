# Lysine Succinylation Prediction

This repository contains the code implementation of Master Thesis of Nguyen Hoang Hai at University of Freiburg under the supervision of Dr. Anup Kumar.

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/haibkhn/MS-Lys-Succ)

## Project Overview

This project aims to predict lysine succinylation using a multi-modal approach combining:
- Sequence-based methods
- Protein Language Model (PLM) based approaches
- Graph Neural Network (GNN) based methods

## Setup Instructions

### Option 1: Using conda environment file
```bash
# Clone the repository
git clone https://github.com/haibkhn/MS-Lys-Succ.git

# Create and activate environment from yml file. Update the prefix to the location of your conda env.
conda env create -f lys-succ-final-environment.yml
conda activate lys-succ
```

### Option 2: Manual setup
```bash
# Create and activate environment
conda create -n lys-succ python=3.8
conda activate lys-succ 

# Install basic dependencies
conda install numpy pandas matplotlib scikit-learn jupyter ipykernel

# Install PyTorch
conda install pytorch torchvision torchaudio -c pytorch

# Install PyTorch Geometric
conda install pyg -c pyg
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)")+cpu.html

# Install additional dependencies
pip install tensorflow
pip install tqdm h5py tensorboard
pip install biopython

# If there is any missing dependencies, install using 
pip install [package-name]
```

## Project Structure

```
data/
├── full_sequence/          # Original FASTA files
├── info/                   # Positive and negative lysine positions
├── structure/             # AlphaFold predicted structures (PDB files)
├── test/                  # Test dataset
│   ├── fasta/            # Extracted sequences (length 33, lysine in middle)
│   ├── PLM/              # Protein Language Model features
│   └── structure/        # Structure features (CSV files)
└── train/                # Training dataset
    ├── fasta/           # Extracted sequences
    ├── PLM/             # Protein Language Model features
    └── structure/       # Structure features

training/
├── 1_standalone/        # Single-track models
│   ├── PLM/            # PLM-only models
│   ├── sequence/       # Sequence-based models
│   └── structure/      # Structure-based models
├── 2_two-component/    # Combined models
│   └── 0_seq_GCN/      # Best performing two-component model
├── 3_three-component/  # Three-component model (sequence + ProtT5 + GCN)
└── fallback/           # Models without structure features for samples without AlphaFold predictions

utils/
├── visualization/      # Visualization tools and scripts
├── feature_important/  # Feature importance analysis
├── prott5utils/       # PLM embedding extraction utilities
├── alphafold/         # AlphaFold PDB file download and processing
├── reports/           # Analysis reports and results
└── structure_extraction/ # Structure feature extraction using PyRosetta
```

### Data Format Details

- **FASTA files**: Extracted sequences with length 33, with lysine (K) in the middle position (17th position)
- **Edge cases**: Sequences at the edges are padded with "-"
- **Structure features**: CSV files containing extracted sequence, position, and structure features
  - Each column represents a feature
  - Each feature contains 33 values corresponding to the 33 amino acids
  - Lysine (K) is at the 17th position

### Model Performance

The best performing model of the two-component model is the combination of sequence and GCN features (`2_two-component/0_seq_GCN`). The three-component model combines sequence, ProtT5 embeddings, and GCN features for enhanced prediction accuracy.