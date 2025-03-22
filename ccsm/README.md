# CCSM: C++ Inference Engine for CSM

CCSM is a high-performance C++ inference engine for the Conversational Speech Model (CSM), providing efficient text-to-speech capabilities. The project aims to create portable, efficient binaries that can run CSM inference on various platforms, with a focus on performance and low resource consumption.

## Features

- CPU-based inference using GGML from llama.cpp
- MLX acceleration for Apple Silicon (in progress)
- Cross-platform compatibility
- Low memory footprint
- Efficient C++ implementation

## Architecture

CCSM follows a modular design:

1. **Core Components**: Common abstractions and utilities
2. **Backend-Specific Implementations**:
   - CPU backend using GGML
   - MLX backend for Apple Silicon
3. **Model Loading**: Efficient weight loading and management
4. **Tokenization**: Text and audio tokenizers
5. **Generation Pipeline**: Text to speech conversion logic

## Building on macOS

### Prerequisites

- CMake
- C++17 compiler
- MLX (for Apple Silicon acceleration)

### Setup on macOS

1. Run the setup script to install dependencies:

```bash
./macos_setup.sh
```

2. Build the project:

```bash
./build.sh
```

### Building Manually

```bash
mkdir -p build
cd build
cmake ..
make
```

### Build Options

The following options can be passed to the build script:

```bash
./build.sh --no-mlx      # Disable MLX support
./build.sh --coverage    # Build with code coverage instrumentation
./build.sh --help        # Show help message
```

## Testing

Run the tests using CTest:

```bash
cd build
ctest
```

### Code Coverage

First, install the required dependencies:

```bash
# On macOS
brew install lcov

# On Ubuntu/Debian
sudo apt-get install lcov
```

Then, to generate a code coverage report, build with coverage instrumentation and run:

```bash
./build.sh --coverage
cd build
ctest                # Run tests to collect coverage data
make coverage        # Generate coverage report
```

Then open the coverage report in your browser:

```bash
open build/coverage_unit/index.html
```

## Usage

### CPU Version

```bash
./build/ccsm-generate --text "Hello, world!" --output output.wav
```

### MLX Version (Apple Silicon)

```bash
./build/ccsm-generate-mlx --text "Hello, world!" --output output.wav
```

## Options

- `--text`, `-t`: Text to generate speech for
- `--speaker`, `-s`: Speaker ID (0-9)
- `--output`, `-o`: Output WAV file path
- `--temperature`: Sampling temperature (default: 0.9)
- `--top-k`: Top-k sampling parameter (default: 50)
- `--max-audio-length-ms`: Maximum audio length in milliseconds (default: 10000)
- `--seed`: Random seed for reproducible generation (default: random)
- `--model`: Path to custom model file (optional)
- `--debug`: Enable debug mode for detailed logs
- `--version`: Show version information
- `--help`: Show usage information

## Implementation Status

The project is currently in active development. For detailed status information, implementation plans, and testing strategy, see [DEVELOPMENT.md](DEVELOPMENT.md).

Key components implemented:
- ✅ Core Tensor System (basic operations, broadcasting)
- ✅ CPU Backend with GGML Integration
- ✅ SIMD Optimizations (matrix multiplication, activation functions)
- ✅ Transformer Architecture Implementation
- ✅ Tokenization Module with SentencePiece
- ✅ CLI Arguments Parser
- ✅ Basic Audio I/O

Work in progress:
- 🔄 Advanced Tensor Operations
- 🔄 GGML Quantization Support
- 🔄 Model Generation Pipeline
- 🔄 MLX Acceleration for Apple Silicon
- 🔄 Thread Pool Improvements

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the same license as the CSM model from Sesame.