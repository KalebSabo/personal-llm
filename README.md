# Personal Language Model

A character-level transformer implementation for educational exploration and experimentation with large language models. This project demonstrates the fundamental architecture and training process of modern language models using PyTorch.

## Overview

This repository contains a simplified implementation of a GPT-style transformer model that generates text character by character. The model is trained on the Tiny Shakespeare dataset and demonstrates core concepts of neural language modeling.

## Features

- **Character-level tokenization** for fine-grained text generation
- **Multi-head self-attention** mechanism with causal masking
- **Transformer architecture** with position embeddings and layer normalization
- **Configurable hyperparameters** for model architecture and training
- **Real-time training progress** monitoring with validation loss tracking

## Architecture

- **Context Length**: 32 characters
- **Embedding Dimension**: 64
- **Transformer Layers**: 4 blocks
- **Attention Heads**: 4 per layer
- **Total Parameters**: ~85K (lightweight for educational purposes)

## Getting Started

### Prerequisites

```bash
pip install torch requests
```

### Usage

Run the training and generation pipeline:

```bash
python llm.py
```

The script will:
1. Download the Tiny Shakespeare dataset
2. Train the model for 5,000 iterations
3. Generate 500 characters of new text

## Model Details

The implementation follows the transformer architecture with:
- Token and positional embeddings
- Multi-head self-attention with causal masking
- Feed-forward networks with ReLU activation
- Layer normalization and dropout for regularization
- Cross-entropy loss for next-character prediction

## Future Enhancements

- [ ] **Domain-Specific Training**: Heat exchanger modeling and engineering data integration
- [ ] **Multimodal Capabilities**: Image-to-text generation and visual understanding
- [ ] **Cloud Integration**: AWS deployment for scalable inference and training
- [ ] **Model Persistence**: Save/load functionality for reusable trained models
- [ ] **Advanced Architectures**: Exploration of newer transformer variants

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the issues page for open tasks or suggest new features.
