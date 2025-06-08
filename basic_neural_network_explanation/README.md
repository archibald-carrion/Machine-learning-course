# Neural Network Analysis: 4-3-2 Feedforward Network

## Overview

This project contains a complete analysis of a feedforward neural network with a 4-3-2 architecture (4 input neurons, 3 hidden neurons, 2 output neurons). The network uses sigmoid activation functions and is fully connected between layers.

## Network Architecture

```mermaid
graph LR
    %% Input Layer
    I1[Input 1<br/>x₁ = 2] --> H1[Hidden 1]
    I2[Input 2<br/>x₂ = 3] --> H1
    I3[Input 3<br/>x₃ = -1] --> H1
    I4[Input 4<br/>x₄ = 2] --> H1
    
    I1 --> H2[Hidden 2]
    I2 --> H2
    I3 --> H2
    I4 --> H2
    
    I1 --> H3[Hidden 3]
    I2 --> H3
    I3 --> H3
    I4 --> H3
    
    %% Hidden to Output
    H1 --> O1[Output 1<br/>0.952559]
    H2 --> O1
    H3 --> O1
    
    H1 --> O2[Output 2<br/>0.999955]
    H2 --> O2
    H3 --> O2
    
    %% Styling
    classDef inputNode fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef hiddenNode fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef outputNode fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    
    class I1,I2,I3,I4 inputNode
    class H1,H2,H3 hiddenNode
    class O1,O2 outputNode
```

## Network Specifications

- **Architecture**: 4-3-2 (Input-Hidden-Output)
- **Activation Function**: Sigmoid σ(x) = 1/(1 + e^(-x))
- **Total Parameters**: 20 (18 weights + 2 biases)
- **Network Type**: Feedforward, fully connected

## Input Data

The network processes the following input vector:
```
x = [2, 3, -1, 2]
```

## Results Summary

| Layer | Neuron | Pre-activation | Post-activation |
|-------|--------|----------------|-----------------|
| Hidden | 1 | 15.000 | 1.000000 |
| Hidden | 2 | 15.000 | 1.000000 |
| Hidden | 3 | 8.000 | 0.999665 |
| Output | 1 | 2.999665 | 0.952559 |
| Output | 2 | 9.998656 | 0.999955 |

**Final Network Output**: [0.952559, 0.999955]

## Implementation

The forward propagation calculations were performed using a Python script:

```bash
python network.py
```

This script implements the complete forward pass through the network, including:
- Matrix multiplication for weight applications
- Bias addition
- Sigmoid activation function application
- Step-by-step calculation verification

## Weight Matrices

**Input to Hidden Layer (W¹)**:
```
[[2, 3, 0, 1],
 [4, 3, 2, 0],
 [1, 1, 1, 2]]
```

**Hidden to Output Layer (W²)**:
```
[[1, 7, 1],
 [7, 1, 4]]
```

**Bias Vector**: [-6, -2]

## Files

- `network.py` - Python implementation of the neural network forward pass
- `report_neural_network.pdf` - Detailed mathematical analysis and derivations

### Files Structure

```
basic_neural_network_explanation/
├── README.md
├── network.py
├── neural_network_log.txt
└── report_neural_network.pdf
```

## Detailed Analysis

For comprehensive mathematical derivations, step-by-step calculations, and theoretical background, please refer to the complete report in `report_neural_network.pdf`.