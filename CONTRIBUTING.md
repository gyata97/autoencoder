# Contributing to Autoencoder Project

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- PyTorch
- Git

### Setting Up Your Development Environment

1. **Fork the repository** on GitHub

2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/autoencoder.git
   cd autoencoder
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify the setup** by running a test forward pass:
   ```bash
   python main.py --model ae --test-forward
   ```

## Development Workflow

### 1. Create a Branch

Always create a new branch for your changes:
```bash
git checkout -b feature/your-feature-name
```
or
```bash
git checkout -b fix/your-bug-fix
```

Use descriptive branch names that indicate what you're working on.

### 2. Make Your Changes

- Write clear, readable code
- Follow the existing code style in the project
- Add comments for complex logic
- Keep functions focused and modular

### 3. Test Your Changes

Before submitting, ensure your changes work correctly:

- Run the model with `--test-forward` flag to verify the architecture:
  ```bash
  python main.py --model <model-name> --test-forward
  ```

- Test with different parameters to ensure flexibility:
  ```bash
  python main.py --model ae --batch-size 32 --channels 1 --test-forward
  ```

### 4. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Add feature: description of what you added"
```

Good commit messages:
- Start with a verb (Add, Fix, Update, Remove, etc.)
- Be specific about what changed
- Keep the first line under 72 characters if possible

### 5. Push and Create a Pull Request

1. **Push your branch** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request** on GitHub:
   - Provide a clear title and description
   - Explain what changes you made and why
   - Reference any related issues
   - Include screenshots or examples if applicable

## Code Style Guidelines

### Python Style

- Follow PEP 8 style guide
- Use meaningful variable and function names
- Keep functions focused on a single responsibility
- Add docstrings for classes and functions

### Model Architecture

- Keep model definitions in separate files (e.g., `autoencoder.py`, `vautoencoder.py`, `alexnet.py`)
- Use consistent naming conventions for layers
- Document any architectural decisions or design choices

### Configuration

- Use `settings.py` for shared configuration values
- Avoid hardcoding values that might need to change

## What to Contribute

We welcome contributions in the following areas:

- **New model architectures** (following the existing pattern)
- **Bug fixes** and improvements
- **Documentation** improvements
- **Performance optimizations**
- **Code refactoring** for better maintainability
- **Additional features** that align with the project's goals

## Pull Request Process

1. **Ensure your code works** - Test thoroughly before submitting
2. **Update documentation** - If you add features, update README.md or relevant docs
3. **Keep PRs focused** - One feature or fix per pull request
4. **Respond to feedback** - Be open to suggestions and make requested changes
5. **Keep your branch updated** - Rebase or merge from main if needed

## Questions or Issues?

- **Found a bug?** Open an issue describing the problem
- **Have a question?** Open an issue with the "question" label
- **Want to discuss a feature?** Open an issue to start a discussion

## Code of Conduct

- Be respectful and considerate
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Respect different viewpoints and experiences

Thank you for contributing! 🎉

