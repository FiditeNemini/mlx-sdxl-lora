# Contributing to MLX-SDXL-LoRA Caption Tool

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/mlx-sdxl-lora.git
   cd mlx-sdxl-lora
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Testing Your Changes

Before submitting a pull request:

1. Test the UI launches correctly:
   ```bash
   python caption_tool.py
   ```

2. Test basic functionality:
   ```bash
   python example_usage.py
   ```

3. Verify syntax:
   ```bash
   python -m py_compile caption_tool.py
   ```

## Code Style

- Follow PEP 8 guidelines
- Use descriptive variable and function names
- Add docstrings to classes and functions
- Keep functions focused and modular

## Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a Pull Request on GitHub

## Feature Requests and Bug Reports

Please open an issue on GitHub with:
- Clear description of the feature/bug
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)

## Areas for Contribution

- Additional VLM model support
- UI/UX improvements
- Performance optimizations
- Documentation improvements
- Bug fixes

## Questions?

Feel free to open an issue for any questions or clarifications.

Thank you for contributing! 🎉
