# ELEC 425 Lab 3: Neural Network with Backpropagation

A MATLAB implementation of a feedforward neural network with manual backpropagation, designed for educational purposes to demonstrate the fundamentals of deep learning.

## Overview

This project implements a 3-layer feedforward neural network from scratch, featuring:
- Manual implementation of forward and backward propagation
- Sigmoid activation functions with gradient computation
- Adam optimizer for weight updates
- Real-time training visualization
- Synthetic 2D regression problem

## Project Structure

```
├── feedforward_net_sigmoid.m                              # Main training script
├── compute_gradient_for_weights_and_one_layer_below.m     # Gradient computation
├── weighted_sum.m                                         # Matrix multiplication
├── activation.m                                           # Sigmoid activation
├── activation_gradient.m                                  # Sigmoid derivative
└── README.md                                              # This file
```

## Getting Started

### Prerequisites
- MATLAB R2016b or later
- No additional toolboxes required

### Running the Network
```matlab
% Run the complete neural network training
feedforward_net_sigmoid
```

## Network Architecture

### Model Structure
```
Input Layer (2D)    →    Hidden Layer (256)    →    Hidden Layer (256)    →    Output Layer (1D)
     [x₁, x₂]       →         ReLU              →         ReLU              →      Linear
```

### Problem Setup
- **Input**: 2D points [x₁, x₂] where x₁, x₂ ∈ [0, 1]
- **Target Function**: y = cos(3πx₁) + sin(πx₂) + 2
- **Dataset**: 1000 synthetic data points
- **Task**: Nonlinear regression

### Training Configuration
- **Batch Size**: 50 samples
- **Epochs**: 550
- **Optimizer**: Adam SGD
- **Learning Rate**: 0.01
- **Loss Function**: Mean Squared Error (MSE)

## Core Components

### Forward Propagation
```matlab
layer1_alpha = weighted_sum(X, W1);           % Linear transformation
layer1_h = activation(layer1_alpha);          % Sigmoid activation

layer2_alpha = weighted_sum(layer1_h, W2);    % Linear transformation  
layer2_h = activation(layer2_alpha);          % Sigmoid activation

output_layer_alpha = weighted_sum(layer2_h, W3);  % Final linear layer
output_layer = output_layer_alpha;            % No activation (regression)
```

### Backward Propagation
The backpropagation follows the chain rule through each layer:

1. **Output Gradient**: `∂L/∂output = 2(output - target)/batch_size`
2. **Weight Gradients**: `∂L/∂W = input^T × received_gradient`
3. **Input Gradients**: `∂L/∂input = received_gradient × W^T`
4. **Activation Gradients**: `∂L/∂α = ∂L/∂h × σ'(h)`

### Activation Functions

#### Sigmoid Activation
```matlab
y = 1./(1+exp(-alpha));
```

#### Sigmoid Derivative
```matlab
gradient = y.*(1-y);
```

## Function Reference

### `feedforward_net_sigmoid.m`
Main training script that orchestrates the entire learning process.

### `compute_gradient_for_weights_and_one_layer_below(RG, W, X)`
**Parameters:**
- `RG`: Received gradient [batch_size, out_dim]
- `W`: Layer weights [in_dim, out_dim]  
- `X`: Layer input [batch_size, in_dim]

**Returns:**
- `W_grad`: Gradient for weights [in_dim, out_dim]
- `h_grad`: Gradient for layer input [batch_size, in_dim]

### `weighted_sum(X, W)`
Computes linear transformation: `alpha = X * W`

### `activation(alpha)`
Applies sigmoid activation: `y = 1/(1+exp(-alpha))`

### `activation_gradient(y)`
Computes sigmoid derivative: `gradient = y*(1-y)`

## Training Process

### Data Preparation
1. Generate 1000 2D training points
2. Compute target values using the nonlinear function
3. Shuffle data for each epoch
4. Process in batches of 50 samples

### Training Loop
1. **Forward Pass**: Compute predictions through all layers
2. **Loss Calculation**: MSE between predictions and targets
3. **Backward Pass**: Compute gradients using chain rule
4. **Weight Update**: Apply Adam optimizer
5. **Visualization**: Plot current predictions vs targets

### Optimization
Uses Adam optimizer with parameters:
- β₁ = 0.9 (momentum)
- β₂ = 0.999 (RMSprop)
- ε = 1e-8 (numerical stability)
- Learning rate = 0.01

## Expected Results

### Training Behavior
- **Initial Loss**: ~2-4 (random initialization)
- **Final Loss**: <0.1 (after 550 epochs)
- **Convergence**: Smooth decrease over epochs
- **Visualization**: Real-time plot showing fit improvement

### Learning Progression
1. **Early epochs**: Random predictions, high loss
2. **Middle epochs**: Gradual shape learning, decreasing loss
3. **Late epochs**: Fine-tuning, low loss convergence

## Learning Objectives

This implementation teaches:

1. **Neural Network Fundamentals**:
   - Forward propagation mechanics
   - Backpropagation algorithm
   - Gradient computation
   - Chain rule application

2. **Implementation Details**:
   - Matrix operations for batch processing
   - Activation function design
   - Weight initialization strategies
   - Training loop structure

3. **Optimization Concepts**:
   - Gradient-based learning
   - Batch processing
   - Adam optimizer mechanics
   - Loss function design

4. **Practical Skills**:
   - Debugging neural networks
   - Training visualization
   - Hyperparameter effects
   - Convergence analysis

## Key Implementation Details

### Weight Initialization
```matlab
W1 = 0.01*randn(2, 256);    % Small random weights
W2 = 0.01*randn(256, 256);
W3 = 0.01*randn(256, 1);
```

### Gradient Computation
Manual implementation of backpropagation using matrix operations for efficiency.

### Batch Processing
All operations vectorized to handle 50 samples simultaneously.

### Real-time Visualization
Plots updated each epoch showing:
- True target function (circles)
- Current predictions (dots)
- Current epoch and loss

## Customization

### Change Network Architecture
```matlab
W1 = 0.01*randn(2, 128);    % Smaller hidden layer
W2 = 0.01*randn(128, 64);   % Different second layer size
W3 = 0.01*randn(64, 1);
```

### Modify Target Function
```matlab
y_all = sin(2*pi*X_all(:,1)) .* cos(pi*X_all(:,2));  % New function
```

### Adjust Training Parameters
```matlab
batch_size = 32;   % Different batch size
learning_rate = 0.001;  % Lower learning rate
```

## Notes

- This is a pedagogical implementation prioritizing clarity over efficiency
- All gradients computed manually to demonstrate backpropagation
- Adam optimizer included as "black box" for focus on core concepts
- Visualization helps understand training dynamics
- Network can be extended to classification with output activation

## Troubleshooting

**Common Issues:**
- Exploding gradients: Reduce learning rate or use gradient clipping
- Vanishing gradients: Check weight initialization or use ReLU activations
- Poor convergence: Increase network capacity or training epochs
- Visualization lag: Reduce plot update frequency

**Debug Tips:**
- Check gradient magnitudes at each layer
- Monitor weight changes during training
- Verify activation function implementations
- Test with simpler target functions first

For additional help, refer to the comments in the code and standard neural network literature.