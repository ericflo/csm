# CSM Documentation

This directory contains comprehensive documentation for the Conversational Speech Model (CSM) project - a state-of-the-art text-to-speech model designed for high-quality, context-aware conversational applications.

## Overview

The Conversational Speech Model (CSM) is a neural text-to-speech system that generates remarkably natural and expressive speech with appropriate prosody, emotion, and voice characteristics based on conversational context. Unlike traditional TTS systems, CSM understands the broader context of a conversation, allowing it to generate speech that responds appropriately to the emotional tone, topic, and flow of the dialogue.

### Key Features

- **Context-aware speech generation** - Produces speech with natural prosody based on conversational history
- **High-quality audio synthesis** - Generates realistic, expressive speech with appropriate emotions and intonation
- **Multi-speaker handling** - Maintains consistent voice characteristics across turns and speakers
- **MLX acceleration** - Optimized for Apple Silicon with significant performance improvements
- **Watermarking technology** - Includes audio watermarking capabilities for content verification
- **Open-source implementation** - Provides a complete, accessible codebase for research and applications

## Architecture

CSM utilizes a dual-transformer architecture based on a LLaMA-style backbone that processes both text and audio tokens to generate expressive speech. The model consists of:

1. A large **Backbone Transformer** (available in 1B, 3B, and 8B parameter variants)
2. A smaller **Audio Decoder Transformer** (approximately 300M parameters)
3. The **Mimi neural audio codec** for audio tokenization and synthesis

This architecture enables CSM to model long-range dependencies in conversations (up to 2048 tokens or approximately 2 minutes of dialogue) and generate appropriate speech based on the full conversational context.

## Documentation Structure

- [Command-Line Interface](cli.md) - Detailed guide for using CSM command-line tools
- [API Reference](api_reference.md) - Complete API documentation for the CSM package
- [Architecture](architecture.md) - Detailed overview of the CSM model architecture
- [Training](training.md) - Comprehensive information about training CSM models
- [Inference](inference.md) - Guide for running inference with CSM models
- [Contributing](contributing.md) - Guidelines for contributing to the project
- [Reference Documentation](reference/README.md) - Technical reference documentation on CSM and related technologies

## Quick Start

To generate speech using CSM, install the package and use one of the command-line tools:

```bash
# Install the package (with Apple Silicon acceleration)
pip install -e ".[apple]"

# Generate speech with MLX acceleration (Mac with Apple Silicon)
csm-generate-mlx --text "Hello, this is a test." --voice warm

# Generate speech on other platforms
csm-generate --text "Hello, this is a test." --device cuda
```

For detailed usage information, see the [Command-Line Interface](cli.md) documentation.

## Platform Support

CSM provides optimized implementations for different platforms:

- **Apple Silicon** (recommended for Mac users): MLX-accelerated implementation with up to 2-3x faster performance
- **NVIDIA GPUs**: CUDA-optimized implementation for high-performance on Linux/Windows systems
- **CPU**: Fallback implementation for systems without GPU acceleration

## License and Citation

CSM is released under the Apache 2.0 license. If you use CSM in your research or applications, please cite the project appropriately.

## Related Resources

- [GitHub Repository](https://github.com/SesameAILabs/csm)
- [Hugging Face Model](https://huggingface.co/sesame/csm_1b)
- [Project Website](https://sesame.ai/csm)