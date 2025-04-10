# Machine Learning Optimization Project

This project implements various machine learning models and optimization algorithms for hyperparameter tuning.

## Project Structure

```
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── optimizers/        # Optimization algorithms
│   ├── utils/             # Utility functions
│   └── data/              # Data processing
├── notebooks/             # Jupyter notebooks
├── data/                  # Dataset files
└── requirements.txt       # Project dependencies
```

## Models
- MLP (Multi-Layer Perceptron)
- CNN (Convolutional Neural Network)
- VGG16

## Optimization Algorithms
- Particle Swarm Optimization (PSO)
- Tree-structured Parzen Estimators (TPE)
- Gaussian Process Regression (GP)
- Gradient-based PSO (GRPSO)

## Requirements
See requirements.txt for all dependencies.

## Usage
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the main script:
```bash
python src/main.py
```

## Notebooks
- `notebooks/test.ipynb`: General testing notebook
- `notebooks/psotest.ipynb`: PSO testing notebook
- `notebooks/TPE.ipynb`: TPE optimization notebook
- `notebooks/lossResults.ipynb`: Loss visualization notebook 