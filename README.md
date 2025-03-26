# T3KAN: Train-at-Test-Time with Kolmogorov-Arnold Networks

This project explores the combination of Train-at-Test-Time (TTT) approaches with Kolmogorov-Arnold Networks (KAN), investigating whether these innovative architectures can work effectively together for sequence modeling tasks.

## Project Overview

T3KAN is an experimental research project that aims to combine two cutting-edge neural network architectures:

1. **Train-at-Test-Time (TTT)** - A novel approach to sequence modeling that achieves linear complexity while maintaining expressiveness by making the hidden state a machine learning model that updates through self-supervised learning even during inference.

2. **Kolmogorov-Arnold Networks (KAN)** - An alternative to traditional MLPs that replaces fixed activation functions with learnable activation functions parametrized as splines, potentially offering better accuracy and interpretability.

The project investigates whether combining these approaches can yield an architecture that maintains the linear complexity advantages of TTT while benefiting from the representational power of KAN.

## Core Components

### 1. Dynamic Sequence Generation System
* **Path:** `t3kan/data/dataset.py` + `config/dataset/generate.yaml`
* Creates synthetic numerical sequences with controllable properties:
  * Multiple concurrent patterns (`components_number`)
  * Controlled complexity via exponential weighting (`seq_components_weights`)
  * Padding/truncation logic for variable lengths (`pad_id`)
  * Enhanced unpredictability through pattern removal (`enhance: true`)

Example sequences might look like `16,83,20,87,24,91,28,...` (combining two sequences: 16 with step=4 and 83 with step=4).

### 2. Train-at-Test-Time (TTT) Engine
* **Path:** `t3kan/models/ttt.py`
* Implements the core TTT approach where the hidden state is a machine learning model itself
* Features:
  * Per-head adaptive learning rates
  * Gradient composition with decay factors
  * Mini-batch optimized parameter updates

### 3. KAN Implementation
* Features:
  * Learnable grid positions for basis functions
  * B-spline basis with configurable degrees
  * Coefficient optimization during inference

### 4. Model Variants
The project includes three model variants:
* **T3Lin** - Implementation of TTT with linear hidden states (baseline)
* **T3MLP** - Implementation of TTT with MLP hidden states (baseline)
* **T3KAN** - Experimental implementation combining TTT with KAN architecture

## Research Background

This project builds upon two key research papers:

1. [**"Learning to (Learn at Test Time): RNNs with Expressive Hidden States"** by Yu Sun et al.](https://arxiv.org/abs/2407.04620)
   - Introduces the Test-Time Training (TTT) approach
   - Demonstrates linear complexity sequence modeling while maintaining expressiveness
   - Shows promising results in long-context tasks

2. [**"KAN: Kolmogorov-Arnold Networks"** by Ziming Liu et al.](https://arxiv.org/abs/2404.19756)
   - Proposes KANs as alternatives to MLPs
   - Replaces fixed activation functions with learnable edge functions
   - Demonstrates improved accuracy and interpretability

## Current Status

The project is in the experimental phase. Key observations:
- The implementation of T3Lin and T3MLP follows the original TTT paper's approach
- T3KAN is a novel combination designed to explore the synergy between TTT and KAN
- Current performance metrics show that T3KAN doesn't yet outperform the baseline models (T3Lin, T3MLP)
- Further investigation is needed to understand the optimization challenges in the combined architecture

## Installation

### Requirements
- Python 3.12+
- Poetry for dependency management

### Setup
1. Clone the repository
2. Install dependencies with Poetry:
```bash
poetry install
```

## Usage

### Dataset Generation
Generate synthetic sequence datasets:
```bash
poetry run generate-data
```

Configuration options in `t3kan/config/dataset/default.yaml`:
- `samples`: Number of sequences to generate
- `vocab_size`: Size of the vocabulary
- `seq_step`: Maximum step size in arithmetic sequences
- `allow_descending`: Allow descending patterns
- `components_number`: Maximum number of joined sequences
- `enhance`: Whether to enhance unpredictability

### Training
Train a model on generated sequences:
```bash
poetry run train
```

Configuration options in `t3kan/config/model/train.yaml`:
- Model selection (T3Lin, T3MLP, T3KAN)
- Training parameters (batch size, learning rate, etc.)
- Early stopping criteria

## Technical Challenges

The project faced several interesting challenges:
1. Implementing the basis function that works across all dimensions
2. Manually calculating gradients for the inner function in TTT since autograd cannot be used
3. Understanding the interaction between TTT's inner-loop optimization and KAN's spline-based representation

## Future Directions

- Investigate optimization techniques to improve T3KAN performance
- Explore different spline parameterizations for the KAN component
- Benchmark with more complex sequence prediction tasks
- Explore memory optimization techniques for better scalability

## Contributing

This is a personal research project, but suggestions and discussions are welcome. Feel free to open issues or submit pull requests.
