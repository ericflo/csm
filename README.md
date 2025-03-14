# CSM - Text-to-Speech Made Easy

CSM (Conversational Speech Model) is a powerful text-to-speech system that generates natural-sounding voices from text. This fork adds enhanced user experience, improved performance, and Apple Silicon acceleration.

## 🚀 Getting Started in 30 Seconds

### Install CSM

```bash
# Clone the repository
git clone https://github.com/ericflo/csm.git
cd csm

# Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install the package
pip install -e .

# For Apple Silicon users (recommended for Mac)
pip install -e ".[apple]"
```

### Generate Your First Audio

```bash
# Generate with a warm voice
csm-generate --text "Hello, this is a test of the CSM speech model." --voice warm

# Using Apple Silicon acceleration
csm-generate-mlx --text "Hello, this is a test of the CSM speech model." --voice warm
```

Your generated audio is saved as `audio.wav` in the current directory.

## 🎭 Voice Presets

CSM includes a variety of built-in voice presets:

- `neutral` - Balanced, default voice
- `warm` - Warmer, friendlier tone
- `deep` - Deeper voice
- `bright` - Brighter, higher pitch
- `soft` - Softer, more gentle voice
- `energetic` - More energetic/animated
- `calm` - Calmer, measured tone
- `clear` - Clearer articulation
- `resonant` - More resonant voice
- `authoritative` - More authoritative tone

Use any preset by specifying `--voice [preset_name]` when generating audio.

## 🖥️ Command-Line Interface

CSM provides two commands for generating speech:

- `csm-generate`: Standard version (works on all platforms)
- `csm-generate-mlx`: MLX-accelerated version for Apple Silicon Macs

```bash
# Basic usage
csm-generate --text "Hello, world!" --voice calm

# With longer duration (in milliseconds)
csm-generate --text "This is a longer example" --max-audio-length-ms 20000

# With different temperature (controls variability)
csm-generate --text "Creative variations" --temperature 1.2

# Save to a specific file
csm-generate --text "Save to a custom file" --output my-audio.wav

# Show detailed performance metrics
csm-generate-mlx --text "Benchmarking" --debug
```

## 📚 Python API

For integration into your Python applications:

```python
from csm.generator import load_csm_1b, Segment
import torchaudio

# Load the model (downloads automatically if needed)
generator = load_csm_1b(model_path=None, device="cuda")  # or "cpu" or "mps"

# Generate speech
audio = generator.generate(
    text="Hello, I'm the CSM model!",
    speaker=1,  # 0-9, corresponds to voice presets
    context=[],
    max_audio_length_ms=10_000,
    temperature=0.9,
)

# Save the audio
torchaudio.save("output.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

### Contextual Generation

CSM produces more natural speech when provided with context:

```python
# Create context segments
segments = [
    Segment(
        text="Hi there, how are you doing today?",
        speaker=0,  # First speaker
        audio=previous_audio_1  # Optional reference audio
    ),
    Segment(
        text="I'm doing great, thanks for asking!",
        speaker=1,  # Second speaker
        audio=previous_audio_2
    )
]

# Generate a response continuing the conversation
response_audio = generator.generate(
    text="That's wonderful to hear. What have you been up to?",
    speaker=0,  # First speaker again
    context=segments,  # Provide conversation context
    max_audio_length_ms=15_000,
)
```

## 🍎 Apple Silicon Acceleration

CSM features state-of-the-art MLX-based acceleration optimized for Apple Silicon:

### 🔄 Modular Architecture

The MLX acceleration features a robust, modular architecture that delivers excellent performance and reliability:

- **Pure MLX Mode**: Full MLX transformer utilizing the Apple Neural Engine and GPU for maximum speed
- **Hybrid Mode**: MLX for embeddings and sampling with PyTorch transformer core (automatic fallback)
- **PyTorch Mode**: Full compatibility mode for all platforms (final fallback)

### 🚀 Key Features

- **Adaptive Generation**: System automatically selects the optimal execution path based on hardware
- **Robust Error Handling**: Multi-level fallback system ensures audio generation even with compatibility issues
- **Audio Token Optimization**: Advanced reshaping and permutation for efficient token generation
- **Watermark Integration**: Seamless integration with the watermarking system
- **Voice Presets**: Pre-configured voice settings for different speech characteristics

### 💻 Using MLX Acceleration

On a Mac with Apple Silicon:

```bash
# Install with Apple optimizations
pip install -e ".[apple]"

# Use the MLX-accelerated generator (basic)
csm-generate-mlx --text "Accelerated with Apple Silicon" --voice warm

# With performance debugging
csm-generate-mlx --text "Show me the performance metrics" --debug

# Try different voice presets
csm-generate-mlx --text "This is the energetic voice preset" --voice energetic
csm-generate-mlx --text "And this is the calm voice preset" --voice calm
```

### ⚡ Performance

The MLX acceleration can provide significant performance improvements:
- Up to 2-4x faster generation on M1/M2/M3 chips vs CPU-only execution
- Reduced memory footprint compared to CUDA/MPS implementations
- Optimized tensor operations specifically for Apple's hardware architecture

## 🔧 Technical Details

<details>
<summary>Click to expand technical information</summary>

### Model Architecture

CSM consists of two main components:

1. **Backbone**: A 1B parameter Llama 3.2 transformer for processing text and encoding context
2. **Audio Decoder**: A 100M parameter Llama 3.2 decoder for generating audio tokens

The model generates audio by:

1. Encoding text and optional audio context
2. Generating RVQ (Residual Vector Quantization) tokens using the dual transformer architecture
3. Decoding tokens to waveform using the Mimi codec

### Performance

- **Sample Rate**: 24kHz high-quality audio
- **Generation Speed**: ~2-3 frames per second on Apple Silicon (~0.3-0.4s per frame)
- **Watermarking**: All generated audio includes an inaudible watermark

### MLX Acceleration Architecture

The MLX acceleration is implemented through a sophisticated modular architecture:

#### Core Components (`src/csm/cli/mlx_components/`)

- **Generator**: Text-to-speech pipeline with multi-stage fallback
- **Model Wrapper**: PyTorch to MLX model conversion and parameter mapping
- **Transformer**: MLX-optimized transformer implementation
- **Sampling**: Token sampling operations with temperature control
- **Config**: Voice preset management and model configuration
- **Utils**: Compatibility checking and performance measurement

#### Supporting Modules

- **MLX Wrapper**: Direct bridge between PyTorch and MLX representations
- **MLX KV-Cache**: Optimized key-value cache for transformer inference
- **MLX Layers**: Core transformer layer implementations
- **MLX Embedding**: Specialized embedding and tensor operations
- **MLX Ops**: Low-level operations optimized for MLX constraints

All components are designed with robust error handling and graceful fallbacks, ensuring that the system can adapt to hardware variations, compatibility issues, and model inconsistencies while still producing high-quality audio output.

</details>

## ❓ FAQ

**Does this model come with pre-trained voices?**

CSM includes 10 voice presets through the `--voice` parameter. These are base voices without specific personality or character traits, but they provide a good starting point.

**Can I fine-tune it on my own voice?**

Fine-tuning capability is planned for future updates. Currently, CSM works best with the included voice presets.

**Does it work on Windows/Linux?**

Yes! CSM works on all platforms through the standard `csm-generate` command. The `csm-generate-mlx` command is Mac-specific for Apple Silicon acceleration.

**How much GPU memory does it need?**

The 1B parameter model works well on consumer GPUs with 8GB+ VRAM. For CPU-only operation, 16GB of system RAM is recommended.

**Does it support other languages?**

CSM is primarily trained on English, but has some limited capacity for other languages. Your mileage may vary with non-English text.

## ⚠️ Responsible Use Guidelines

This project provides high-quality speech generation for creative and educational purposes. Please use responsibly:

- **Do not** use for impersonation without explicit consent
- **Do not** create misleading or deceptive content
- **Do not** use for any illegal or harmful activities

CSM includes watermarking to help identify AI-generated audio.

## 📄 License

This project is based on the CSM model from Sesame. See LICENSE for details.

## 🙏 Acknowledgements

Special thanks to the original CSM authors: Johan Schalkwyk, Ankit Kumar, Dan Lyth, Sefik Emre Eskimez, Zack Hodari, Cinjon Resnick, Ramon Sanabria, Raven Jiang, and the Sesame team for releasing this incredible model.

---

Made with ❤️ by the CSM community
