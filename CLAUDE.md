# Claude's Code Guide

This document contains important information about the CSM (Conversational Speech Model) codebase, including common commands, code style guidelines, and project structure.

⚠️ **IMPORTANT: Always run tests after making changes** ⚠️  
Run `python -m pytest` to validate your changes before committing them.  
This ensures code stability and prevents regressions.

## Project Structure

```
csm/
├── docs/                     # Documentation files
├── src/                      # Source code
│   └── csm/                  # Main package
│       ├── __init__.py       # Package initialization
│       ├── watermarking/     # Audio watermarking functionality
│       │   ├── __init__.py
│       │   ├── utils.py
│       │   └── silentcipher/ # Vendored silentcipher code
│       ├── models/           # Model definitions
│       ├── training/         # Training code
│       ├── inference/        # Inference utilities
│       ├── data/             # Data loading and processing
│       └── utils/            # Utility functions
├── tests/                    # Test suite
├── .gitignore                # Git ignore file
├── pyproject.toml            # Project configuration
├── LICENSE                   # License file
└── README.md                 # Project documentation
```

## Development Setup

```bash
# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install the package in development mode
pip install -e ".[dev]"
```

## Development Commands

```bash
# Run tests with coverage (ALWAYS do this before committing changes)
python -m pytest

# Run specific tests
python -m pytest tests/unit/test_model_args.py

# View detailed coverage report
open htmlcov/index.html

# Format code with black and isort
black src/ tests/
isort src/ tests/

# Lint code with ruff
ruff src/ tests/

# Type check with mypy
mypy src/ tests/
```

## Code Style Guidelines

1. Use Python 3.11+ features
2. Follow PEP 8 guidelines
3. Maximum line length is 100 characters
4. Use type annotations for all function definitions
5. Document all public functions and classes with docstrings
6. Use black for code formatting
7. Organize imports with isort
8. Use ruff for linting
9. Maintain test coverage for new functionality

## Notes

1. The `silentcipher` dependency has been vendored into `src/csm/watermarking/silentcipher/` to avoid external Git dependencies
2. The project uses PyTorch for all model implementations, with MLX acceleration support for Apple Silicon
3. The codebase follows a modular design to facilitate future expansion

## MLX Acceleration Architecture

The MLX acceleration for Apple Silicon is implemented as a modular system:

### Core Components in `src/csm/cli/mlx_components/`

1. `src/csm/cli/mlx_components/utils.py`: Utility functions for MLX acceleration
   - Compatibility checking and error handling
   - Performance measurement functions
   - Debug helpers and type formatting

2. `src/csm/cli/mlx_components/config.py`: Configuration management
   - Voice preset definitions and handling
   - Default parameter values
   - Model configuration

3. `src/csm/cli/mlx_components/transformer.py`: Transformer implementation
   - MLX-optimized transformer blocks
   - Attention mechanisms
   - Position embeddings and mask handling

4. `src/csm/cli/mlx_components/sampling.py`: Token sampling operations
   - Top-k sampling implementation
   - Temperature-based sampling
   - Categorical sampling utilities

5. `src/csm/cli/mlx_components/model_wrapper.py`: Model conversion
   - PyTorch to MLX model conversion
   - Parameter handling and transfer
   - Forward pass implementation

6. `src/csm/cli/mlx_components/generator.py`: Speech generation
   - Text to audio token generation
   - Audio token decoding
   - Watermarking integration
   - Multiple fallback paths for robustness

### Supporting Files

1. `src/csm/cli/mlx_layers.py`: Core MLX layer implementations
   - Transformer layers and components
   - RoPE implementation and attention mechanisms

2. `src/csm/cli/mlx_embedding.py`: Embedding operations
   - Text and audio embedding functions
   - Shape-safe tensor operations

3. `src/csm/cli/mlx_kvcache.py`: Key-value cache implementation
   - Optimized cache for transformer inference
   - Position-based indexing

4. `src/csm/cli/mlx_ops.py`: Low-level MLX operations
   - Tensor manipulation utilities
   - Math operations compatible with MLX constraints
   - Conversion between PyTorch and MLX tensors

5. `src/csm/cli/mlx_generation.py`: Generation pipeline
   - Frame generation logic
   - Error handling and fallbacks

6. `src/csm/cli/mlx_wrapper.py`: PyTorch-MLX bridge
   - Model parameter conversion
   - Support for both direct Model and Generator classes

7. `src/csm/cli/generate_mlx.py`: Command-line interface
   - Main entry point for MLX acceleration
   - Multi-stage fallback system for robustness
   - Integration with watermarking and audio processing
   - Performance tracking and reporting

When running on Apple Silicon, the system first attempts pure MLX execution for maximum performance. If any issues are encountered, it automatically falls back to hybrid mode and ultimately to PyTorch if needed. The architecture includes special handling for MLX's tensor operations, particularly around reshape operations which differ from PyTorch's implementation.