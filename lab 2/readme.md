# ELEC 425 Lab 2: K-means Clustering Algorithm

A MATLAB implementation of the K-means clustering algorithm with interactive visualization, designed for educational purposes.

## Overview

This project implements the K-means clustering algorithm with real-time visualization of the Expectation-Maximization (EM) process. The implementation shows step-by-step how data points are assigned to clusters (E-step) and how cluster centers are updated (M-step).

## Project Structure

```
├── k_means.m           # Core K-means implementation with visualization
├── k_means_main.m      # Main script to run clustering
├── data.mat           # Sample 2D dataset
└── README.md          # This file
```

## Getting Started

### Prerequisites
- MATLAB R2016b or later
- Statistics and Machine Learning Toolbox (for pdist2 function)

### Data Files
The project includes:
- `data.mat` - Sample 2D dataset for clustering demonstration

## Usage

### Basic Usage
```matlab
% Run the complete K-means demonstration
k_means_main
```

### Custom Usage
```matlab
% Load your own data
load('data.mat');

% Run K-means with different number of clusters (1-4)
[membership, centres] = k_means(data, 3);
```

## Algorithm Features

### Core Implementation
- **Initialization**: Random cluster center initialization within data bounds
- **Distance Metric**: Squared Euclidean distance
- **Assignment**: Each point assigned to nearest cluster center
- **Update**: Cluster centers updated as mean of assigned points
- **Convergence**: Stops when no points change cluster membership

### Visualization Features
- **Real-time Display**: Shows algorithm progression step-by-step
- **Color Coding**: Different colors/symbols for each cluster (up to 4)
- **Center Marking**: Cluster centers clearly marked with filled symbols
- **Step Annotations**: Text annotations for each algorithm phase
- **Pause Control**: Adjustable pause time between steps (default: 4 seconds)

## Function Reference

### `k_means(X, n_cluster)`

**Parameters:**
- `X`: Data matrix (N×d) where rows are data points and columns are features
- `n_cluster`: Number of clusters (maximum 4 for visualization)

**Returns:**
- `membership`: N×1 vector indicating cluster assignment for each data point
- `centres`: n_cluster×d matrix of final cluster centers

**Visualization Symbols:**
- Cluster 1: Red circles (ro)
- Cluster 2: Green pentagrams (gp) 
- Cluster 3: Blue diamonds (bd)
- Cluster 4: Black triangles (k^)

## Algorithm Steps

The implementation follows the standard K-means algorithm:

1. **Initialization**: 
   - Randomly initialize cluster centers within data bounds
   - Display initial setup

2. **E-Step (Expectation)**:
   - Calculate squared Euclidean distances between all points and centers
   - Assign each point to the closest cluster center
   - Visualize new assignments

3. **M-Step (Maximization)**:
   - Update cluster centers as the mean of assigned points
   - Handle empty clusters by keeping previous center
   - Visualize updated centers

4. **Convergence Check**:
   - Compare current and previous assignments
   - Stop if no points changed clusters
   - Otherwise, repeat from step 2

## Key Implementation Details

### Distance Calculation
```matlab
distance = pdist2(X, centres, 'squaredeuclidean');
```

### Point Assignment
```matlab
[~, membership] = min(distance, [], 2);
```

### Center Update
```matlab
for i = 1:n_cluster
    if any(membership == i)
        centres(i,:) = mean(X(membership == i, :));
    end
end
```

## Expected Results

### Typical Behavior
- **Convergence**: Usually 3-10 iterations for well-separated clusters
- **Visualization**: Clear progression from initial random assignment to final clustering
- **Performance**: Real-time visualization with 4-second pauses between steps

### Limitations
- Maximum 4 clusters for visualization
- May converge to local minima depending on initialization
- Assumes spherical clusters with similar sizes
- Sensitive to outliers

## Learning Objectives

This implementation helps understand:

1. **K-means Algorithm**:
   - Iterative optimization process
   - E-step vs M-step distinction
   - Convergence criteria
   - Local minima issues

2. **Clustering Concepts**:
   - Centroid-based clustering
   - Distance-based similarity
   - Cluster assignment strategies
   - Algorithm visualization

3. **Implementation Skills**:
   - Matrix operations in MATLAB
   - Vectorized distance calculations
   - Real-time plotting and visualization
   - Algorithm flow control

## Customization

### Adjusting Pause Time
Modify the pause duration in the `show` function:
```matlab
pause(2); % Change from 4 to 2 seconds
```

### Adding More Clusters
To support more than 4 clusters, extend the symbol array:
```matlab
symbol = ['ro'; 'gp'; 'bd'; 'k^'; 'r*'; 'co'; 'mp'];
```

### Different Initialization
Replace random initialization with specific values:
```matlab
centres = [5.5, 4; 4.5, 3.2; 6.5, 3.5]; % Fixed centers
```

## Notes

- The visualization is optimized for 2D data
- For higher dimensions, only the algorithm will work (no visualization)
- Empty clusters are handled by maintaining previous center positions
- The algorithm guarantees convergence for finite datasets

## Troubleshooting

**Common Issues:**
- "Too many clusters" error: Reduce n_cluster to 4 or less
- No convergence: Check for identical data points or increase iteration limit
- Poor clustering: Try different random seeds or initialization methods
- Visualization issues: Ensure data is 2D for proper plotting

For additional help, check MATLAB documentation for `pdist2`, `min`, and `mean` functions.