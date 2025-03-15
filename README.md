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

# For fine-tuning capabilities
pip install -e ".[finetune,lora]"
```

### Generate Your First Audio

```bash
# Generate speech
csm-generate --text "Hello, this is a test of the CSM speech model."

# Using Apple Silicon acceleration
csm-generate-mlx --text "Hello, this is a test of the CSM speech model."
```

Your generated audio is saved as `audio.wav` in the current directory.

## 🖥️ Command-Line Interface

CSM provides several commands for speech generation and model fine-tuning:

### Speech Generation
- `csm-generate`: Standard version (works on all platforms)
- `csm-generate-mlx`: MLX-accelerated version for Apple Silicon Macs

### Model Training
- `csm-train`: Standard training (works on all platforms)
- `csm-train-mlx`: MLX-accelerated training for Apple Silicon Macs

### Fine-tuning with LoRA
- `csm-finetune-lora`: Fine-tune with LoRA on Apple Silicon
- `csm-finetune-lora-multi`: Multi-speaker fine-tuning with LoRA

### Benchmarking and Optimization
- `csm-benchmark-lora`: Benchmark LoRA configurations
- `csm-benchmark-mlx`: Benchmark MLX acceleration

```bash
# Basic usage
csm-generate --text "Hello, world!"

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

### Fine-tuning with LoRA

CSM supports parameter-efficient fine-tuning using LoRA on Apple Silicon:

#### Command-line Fine-tuning with Hugging Face Datasets

The simplest way to fine-tune on voice datasets from Hugging Face:

```bash
# Download from Hugging Face, fine-tune, and generate samples
python examples/huggingface_lora_finetune.py \
  --model-path path/to/model.safetensors \
  --output-dir ./hf_finetuned_model \
  --dataset mozilla-foundation/common_voice_16_0 \
  --language en \
  --num-samples 100 \
  --lora-r 8 \
  --batch-size 2 \
  --epochs 5
```

#### Verify Hugging Face Workflow

To verify the Hugging Face workflow works correctly:

```bash
# Test the workflow with a specific model
python examples/test_lora_finetune.py \
  --model-path path/to/model.safetensors \
  --num-samples 20 \
  --output-dir ./test_output

# Comprehensive testing of all LoRA functionality
python -m csm.training.test_lora_comprehensive
```

#### Python API for Custom Fine-tuning

```python
from csm.training.lora_trainer import CSMLoRATrainer

# Initialize LoRA trainer
trainer = CSMLoRATrainer(
    model_path="path/to/model.safetensors",
    output_dir="./fine_tuned_model",
    lora_r=8,               # LoRA rank
    lora_alpha=16.0,        # LoRA scaling factor
    target_modules=["q_proj", "v_proj"]  # Which modules to fine-tune
)

# Prepare optimizer
trainer.prepare_optimizer()

# Train the model
trainer.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=2,
    epochs=5
)

# Save the fine-tuned model (LoRA weights only)
trainer.save_model("fine_tuned_model.safetensors", save_mode="lora")

# Generate speech with the fine-tuned model
audio = trainer.generate_sample(
    text="This is speech from the fine-tuned model.",
    speaker_id=0
)
```

For more details on fine-tuning, see the [examples directory](./examples/) and [LoRA documentation](./docs/reference/sesame_csm/lora_finetuning.md).

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

### 🔄 Pure MLX Implementation

The MLX acceleration features an optimized pure-MLX implementation that delivers exceptional performance and reliability:

- **High-Performance Transformer**: Complete MLX transformer pipeline utilizing the Apple Neural Engine and GPU for maximum performance
- **PyTorch-Matching Sampling**: Precisely engineered token sampling that matches PyTorch's quality with MLX's speed
- **Memory-Optimized Operations**: Carefully designed tensor operations that minimize memory usage while maintaining accuracy
- **Automatic Fallbacks**: Intelligent fallback system that ensures reliability while prioritizing performance

### 🚀 Key Features

- **Optimized Token Generation**: Advanced token sampling that achieves >95% distribution similarity to PyTorch while running entirely on MLX
- **Vectorized Operations**: Carefully tuned matrix operations that leverage Apple Silicon's parallel processing capabilities
- **Numeric Stability**: Meticulous implementation of sampling algorithms with proper temperature scaling and top-k filtering
- **Intelligent Caching**: Strategic memory management and key-value caching to reduce redundant computations
- **Parameter Optimization**: Carefully tuned temperature and sampling parameters for optimal audio quality with MLX

### 💻 Using MLX Acceleration

On a Mac with Apple Silicon:

```bash
# Install with Apple optimizations
pip install -e ".[apple]"

# Basic usage with MLX acceleration
csm-generate-mlx --text "Accelerated with Apple Silicon"

# Enable performance optimization with environmental variables
MLX_AUTOTUNE=1 MLX_NUM_THREADS=6 csm-generate-mlx --text "Fully optimized for Apple Silicon"

# With performance debugging to see metrics
csm-generate-mlx --text "Show me the performance metrics" --debug

# Try different parameter combinations
csm-generate-mlx --text "High temperature and top-k" --temperature 1.3 --topk 80
csm-generate-mlx --text "Low temperature and top-k" --temperature 0.8 --topk 20
```

### ⚡ Performance

The optimized MLX implementation delivers impressive performance gains:
- Up to 2-4x faster generation on M1/M2/M3 chips vs CPU-only execution
- Reduced memory footprint compared to CUDA/MPS implementations
- Token generation optimized to achieve >95% distribution similarity to PyTorch
- Specialized tensor operations that leverage Apple Silicon's unique architecture
- Environmental variable tuning (MLX_AUTOTUNE, MLX_NUM_THREADS) for maximum performance

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

The MLX acceleration is implemented through a sophisticated architecture optimized for Apple Silicon:

#### Core Components (`src/csm/cli/mlx_components/`)

- **Generator**: Pure MLX text-to-speech pipeline with optimized token generation
- **Transformer**: MLX-optimized transformer with custom attention mechanisms
- **Sampling**: High-fidelity token sampling that precisely matches PyTorch distributions
- **Model Wrapper**: Efficient PyTorch to MLX model conversion with BFloat16 support
- **Config**: Voice preset management optimized for MLX performance characteristics
- **Utils**: Performance profiling and compatibility verification

#### Advanced Implementation

- **Gumbel-Max Sampling**: Precise implementation of the Gumbel-max trick for categorical sampling
- **Optimized KV-Cache**: Specialized key-value cache designed for MLX's memory model
- **Core Tensor Operations**: Carefully crafted low-level operations that work around MLX constraints
- **Token Distribution Analysis**: Comprehensive testing to ensure sampling matches PyTorch quality
- **Memory Optimization**: Intelligent array reuse and caching to minimize memory allocations

Each component was designed with both performance and accuracy in mind, enabling the system to achieve PyTorch-level audio quality while leveraging the full computational power of Apple Silicon's Neural Engine and GPU architectures.

</details>

## ❓ FAQ

**Does this model come with pre-trained voices?**

CSM includes support for 10 different speaker IDs (0-9) through the `--speaker` parameter. These are base voices without specific personality or character traits.

**Can I fine-tune it on my own voice?**

Fine-tuning capability is planned for future updates. Currently, CSM works best with the included speaker IDs.

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
