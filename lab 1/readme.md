# ELEC 425 Lab 1: k-NN and Perceptron

A MATLAB implementation of two fundamental machine learning algorithms: k-Nearest Neighbors (k-NN) and Perceptron, designed for educational purposes.

## Overview

This project implements and demonstrates two classic machine learning algorithms:
- **k-Nearest Neighbors (k-NN)**: A non-parametric classification algorithm
- **Perceptron**: A linear binary classifier with geometric interpretation

## Project Structure

```
├── knn.m                           # Core k-NN implementation
├── knn_demo.m                      # Complete k-NN demonstration
├── knn_tvt_incomplete.m            # k-NN train/validation/test (incomplete)
├── perceptron_incomplete.m         # Perceptron implementation (incomplete)
├── perceptron_training_demo.m      # Perceptron visualization demo
├── a1digits.mat                    # Handwritten digits dataset
├── a1distribs.mat                  # Distribution data
├── 2d_points.mat                   # 2D points for perceptron classification
└── README.md                       # This file
```

## Getting Started

### Prerequisites
- MATLAB R2018b or later
- Statistics and Machine Learning Toolbox (for some distance metrics)

### Data Files
The project includes the following data files:
- `a1digits.mat` - Handwritten digits dataset (8×8 pixel images)
- `2d_points.mat` - 2D points for perceptron classification
- `a1distribs.mat` - Additional distribution data

## Datasets

### Digits Dataset
- **Format**: 8×8 pixel grayscale images (64 features)
- **Classes**: Digits 1-10 
- **Training**: 700 samples per digit (7,000 total)
- **Testing**: 400 samples per digit (4,000 total)

### 2D Points Dataset
- **Format**: 2D coordinate points
- **Classes**: Binary classification (+1, -1)
- **Size**: 60 points (30 per class)

## Usage

### k-Nearest Neighbors

#### Complete Demo
```matlab
% Run the complete k-NN demonstration
knn_demo
```

#### Train/Validation/Test Split
```matlab
% Run k-NN with hyperparameter tuning
knn_tvt_incomplete
```

**Key Features:**
- Hyperparameter tuning using validation set
- Multiple k values tested: [2, 3, 4, 5, 10, 20, 30, 50, 100]
- Various distance metrics supported
- Performance visualization

#### Supported Distance Metrics
- `'euclidean'` - L2 distance (default)
- `'cityblock'` - Manhattan distance (L1)
- `'cosine'` - Cosine similarity
- `'correlation'` - Correlation distance
- `'chebychev'` - Chebyshev distance
- `'mahalanobis'` - Mahalanobis distance
- `'user_define'` - Custom distance function

### Perceptron

#### Training with Visualization
```matlab
% Run perceptron with real-time decision boundary visualization
perceptron_training_demo
```

**Features:**
- Real-time visualization of decision boundary evolution
- Highlights currently processed data point
- Convergence to linearly separable solution
- Adjustable learning rate

## Core Functions

### `knn(querys, trains, labels, k, distance, dist_func)`
**Parameters:**
- `querys`: Query points to classify (M×d matrix)
- `trains`: Training data (N×d matrix)  
- `labels`: Training labels (N×1 vector)
- `k`: Number of nearest neighbors
- `distance`: Distance metric string
- `dist_func`: Custom distance function handle (optional)

**Returns:**
- `pred`: Predicted labels
- `knn_label`: Labels of k nearest neighbors
- `knn_ind`: Indices of k nearest neighbors

### `perceptron(X, t, eta)`
**Parameters:**
- `X`: Feature matrix (N×d)
- `t`: True labels (N×1, values: +1 or -1)
- `eta`: Learning rate

**Returns:**
- `w`: Learned weights
- `b`: Learned bias
- `h_w`: Weight history (for visualization)
- `h_b`: Bias history (for visualization)
- `upd_ind`: Update indices (for visualization)

## Expected Results

### k-NN Performance
- **Typical accuracy**: 85-95% on digits dataset
- **Best k**: Usually between 3-10
- **Training time**: ~1-5 seconds depending on k

### Perceptron Convergence
- **Convergence**: Guaranteed for linearly separable data
- **Iterations**: Typically 50-200 epochs
- **Visualization**: Real-time decision boundary updates

## Learning Objectives

This project helps understand:

1. **k-NN Algorithm**:
   - Non-parametric classification
   - Impact of k value on bias-variance tradeoff
   - Distance metric selection
   - Validation set usage for hyperparameter tuning

2. **Perceptron Algorithm**:
   - Linear classification
   - Gradient-based learning
   - Geometric interpretation of decision boundaries
   - Convergence properties

3. **Machine Learning Workflow**:
   - Train/validation/test splits
   - Hyperparameter tuning
   - Performance evaluation
   - Model selection

## Notes

- The files marked "incomplete" are designed as exercises
- Perceptron only works with linearly separable data
- k-NN can be computationally expensive for large datasets
- Ensure proper data normalization for distance-based methods

## Troubleshooting

**Common Issues:**
- Missing data files: Ensure `a1digits.mat` and `2d_points.mat` are in the working directory
- Memory issues: Reduce dataset size or use smaller k values
- Convergence issues: Check if data is linearly separable for perceptron

For additional help, check MATLAB documentation or course materials.