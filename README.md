[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/D2himself/mlp-from-scratch/blob/main/notebook/mlp_from_scratch.ipynb)


# 🧠 Deep Learning From Scratch: Understanding Neural Networks at the Foundation Level

> *"When you build something from scratch, you don't just use the tool — you internalize it."*

A comprehensive implementation of neural network fundamentals built from the ground up using only basic PyTorch tensor operations. This project demonstrates how modern deep learning frameworks work under the hood by recreating core components manually.

## 🎯 Project Overview

This repository contains a step-by-step implementation of a complete neural network training pipeline, built from first principles to understand what happens behind PyTorch's abstractions. Instead of using high-level APIs, every component is implemented manually and then validated against PyTorch's built-in functions.

### What Makes This Different

- **No shortcuts**: Every layer, loss function, and optimization step implemented manually
- **Validation**: Custom implementations tested against PyTorch's native functions
- **Progressive complexity**: Building from simple linear layers to complete training pipelines

## 📚 What I've Learnt

- **Neural Network Fundamentals**: Linear layers, activations, and forward propagation
- **Backpropagation**: Manual gradient computation using the chain rule
- **Loss Functions**: From MSE to CrossEntropy, understanding why each matters
- **Training Dynamics**: Optimizers, learning rates, and training loops
- **Data Pipeline**: Custom Dataset, DataLoader, and Sampler implementations
- **PyTorch Internals**: How `nn.Module`, autograd, and other components actually work

## 🛠️ Implementation Roadmap

### Phase 1: Basic Neural Network
- [x] Linear layer implementation (`y = xW + b`)
- [x] ReLU activation function
- [x] Forward pass for 2-layer MLP
- [x] MSE loss computation

### Phase 2: Backpropagation
- [x] Manual gradient computation for linear layers
- [x] Gradient flow through ReLU activation
- [x] Chain rule implementation for multi-layer networks
- [x] Validation against PyTorch's autograd

### Phase 3: Object-Oriented Architecture
- [x] Custom `Module` base class
- [x] `Linear` and `ReLU` layer classes
- [x] Model composition and parameter management
- [x] Integration with PyTorch's `nn.Module`

### Phase 4: Advanced Loss Functions
- [x] Log-softmax implementation
- [x] Negative log-likelihood loss
- [x] CrossEntropy loss from components
- [x] Numerical stability considerations

### Phase 5: Training Infrastructure
- [x] Custom optimizer implementation
- [x] Training loop with validation
- [x] Accuracy metrics and monitoring
- [x] Comparison with `torch.optim.SGD`

### Phase 6: Data Pipeline
- [x] Custom `Dataset` class
- [x] `DataLoader` with batching support
- [x] `Sampler` and `BatchSampler` implementations
- [x] Integration with PyTorch's data utilities


## 📖 Code Examples

### Manual Linear Layer
```python
def lin(x, w, b):
    """Basic linear transformation: y = xW + b"""
    return x @ w + b

# With gradient computation
def lin_grad(inp, out, w, b):
    """Backward pass for linear layer"""
    inp.g = out.g @ w.t()     # Gradient w.r.t input
    w.g = inp.t() @ out.g     # Gradient w.r.t weights
    b.g = out.g.sum(0)        # Gradient w.r.t bias
```

### Custom Module System
```python
class Module():
    def __call__(self, *args):
        self.args = args
        self.out = self.forward(*args)
        return self.out

    def forward(self):
        raise Exception('not implemented')
    
    def backward(self):
        self.bwd(self.out, *self.args)

class Linear(Module):
    def __init__(self, n_in, n_out):
        self.w = torch.randn(n_in, n_out) * math.sqrt(2/n_in)
        self.b = torch.zeros(n_out)
    
    def forward(self, inp):
        return inp @ self.w + self.b
```

## 📊 Results & Validation

All custom implementations are validated against PyTorch's built-in functions:

- ✅ Forward pass outputs match `nn.Linear`


## 📈 Next Steps

- [ ] Implement attention mechanisms from scratch


Please feel free to open issues or submit pull requests.


## 🙏 Acknowledgments

- **FastAI** for excellent educational resources
- **PyTorch team** for building such an elegant framework
- **ChatGPT** for patient tutoring through the mathematical concepts

## 📞 Connect

If this project helped you understand deep learning better, I'd love to hear about it!


- 📝 [Medium Profile]

---

*"The best way to understand something is to build it yourself."*

⭐ **Found this helpful?** Give it a star and share it with others who might benefit from understanding neural networks at this fundamental level!

