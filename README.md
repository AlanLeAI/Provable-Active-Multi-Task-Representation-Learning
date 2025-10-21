# Provable Active Multi Task Representation Learning - IEEE TSP 2026

This repository contains the implementation of the Active Low-Rank Representation Learning algorithm for multi-task learning with adaptive sampling.

## Overview

The algorithm learns low-dimensional linear representations from multiple source tasks through an adaptive sampling approach. By utilizing more samples from source tasks that are more relevant to the target task, the algorithm accelerates the learning process compared to uniform sampling approaches.

## Key Features

- **Adaptive Sampling**: Samples are allocated proportionally to task relevance (∝ ν̂(m)²)
- **Low-Rank Representation**: Learns compact representations using alternating gradient descent
- **Spectral Initialization**: Robust initialization using truncated spectral method
- **Multi-Task Learning**: Leverages knowledge from multiple source tasks to improve target task performance

## Repository Structure

```
.
├── data/
│   ├── mnist_c/                    # MNIST-C corruption datasets
│   ├── mnist_c_full_processed/     # Processed MNIST-C data
│   ├── ml-100k/                    # MovieLens-100K dataset
│   └── estimated_matrix.npy        # Completed MovieLens rating matrix
├── utils.py                        # Core algorithm implementations
├── train_all.py                    # MNIST-C experiments
├── train_movielens_2.py           # MovieLens experiments
├── preprocessing.py                # Data preprocessing utilities
├── plot_methods_comparison.ipynb   # Visualization notebooks
└── README.md
```

## Experiments

### MovieLens-100K Experiment

#### Dataset Description
- **Size**: 100,000 ratings from 943 users on 1,682 movies
- **Task**: Collaborative filtering and rating prediction
- **Preprocessing**: 
  - Sparse matrix completion using matrix factorization (50 iterations, lr=0.01, reg=0.1)
  - Normalization to [0,1] range
  - NMF decomposition to extract latent features

#### Task Generation
1. **NMF Decomposition**: Rating matrix R ∈ ℝ^(943×1682) → W ∈ ℝ^(943×d), H ∈ ℝ^(1682×d)
2. **K-means Clustering**: Apply clustering on item features H to create M-1 source tasks
3. **Target Task**: Linear combination of two linearly independent source tasks
   - θ_target = 0.9 × θ_0 + 0.1 × θ_1

#### Data Generation
- For each task i: y = x^T θ_i + ε, where ε ~ N(0, 0.01²)
- User features from W serve as input features
- Labels generated using dot product with task parameters

#### Parameters
- **d**: 10 (latent feature dimension)
- **M**: 30 (29 source tasks + 1 target task)
- **k**: 2 (rank parameter, determined by matrix rank)
- **samples_per_task**: 50 samples per source task
- **num_target_sample**: 20 target task samples
- **epochs**: 4
- **learning_rate**: 1e-4
- **gd_iterations**: 500

#### Running MovieLens Experiment
```bash
python train_movielens.py
```

### MNIST-C Experiment

#### Dataset Description
- **Source**: MNIST-C corruption dataset
- **Corruptions**: Multiple corruption types (brightness, blur, etc.)
- **Tasks**: Binary classification for each digit (0-9)

#### Task Generation
1. Load corrupted MNIST images for each corruption type
2. Create 10 binary classification tasks (one per digit)
3. Generate target task by corrupting clean MNIST images

#### Parameters
- **d**: Varies by experiment
- **k**: Varies by experiment (typically 2-30)
- **M**: Number of source tasks (varies by corruption)
- **samples_per_epoch**: 50
- **epochs**: 4

#### Preprocessing
```bash
python preprocessing.py
```

This script:
- Loads MNIST-C corruption datasets
- Splits data into 10 subsets (one per digit)
- Creates binary labels for each subset
- Saves processed data to `data/mnist_c_full_processed/`

#### Running MNIST-C Experiment
```bash
python train_all.py
```

## Data Setup

### MovieLens-100K
1. Download MovieLens-100K dataset from [GroupLens](https://grouplens.org/datasets/movielens/100k/)
2. Extract to `data/ml-100k/`
3. Run collaborative filtering to generate `data/estimated_matrix.npy`

### MNIST-C
1. Download MNIST-C dataset from [repository](https://github.com/google-research/mnist-c)
2. Extract to `data/mnist_c/`
3. Run preprocessing: `python preprocessing.py`

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your-paper,
  title={Provable Active Multi Task Representation Learning},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2025}
}
```

## License

[Your chosen license]

## Contact

[Your contact information]

## Acknowledgments

- MovieLens-100K dataset: Harper and Konstan (2015)
- MNIST-C dataset: Mu and Gilmer (2019)