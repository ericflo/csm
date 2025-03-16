# LoRA Fine-Tuning with CSM

This document covers the Low-Rank Adaptation (LoRA) fine-tuning capabilities in the CSM framework, including integration with Hugging Face datasets and testing infrastructure for validation.

## Overview

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that significantly reduces the number of trainable parameters while maintaining performance. CSM implements LoRA for speech model fine-tuning with both local data and Hugging Face datasets.

Key features:
- Parameter-efficient fine-tuning with customizable rank and target modules
- MLX acceleration optimized for Apple Silicon
- Multiple save modes (lora, full, and both)
- Integrated support for Hugging Face datasets
- Robust testing infrastructure 
- Multi-level audio generation fallbacks

## Basic Usage

Fine-tuning a CSM model with LoRA from the command line:

```bash
python -m csm.cli.finetune_lora \
  --model-path /path/to/model.safetensors \
  --output-dir ./fine_tuned_model \
  --audio-dir /path/to/audio \
  --transcript-dir /path/to/transcripts \
  --lora-r 8 \
  --lora-alpha 16.0 \
  --batch-size 2 \
  --epochs 5
```

## Hugging Face Integration

CSM provides seamless integration with Hugging Face datasets:

```bash
python examples/huggingface_lora_finetune.py \
  --model-path /path/to/model.safetensors \
  --output-dir ./hf_finetuned_model \
  --dataset mozilla-foundation/common_voice_16_0 \
  --language en \
  --num-samples 100 \
  --lora-r 8 \
  --batch-size 2 \
  --epochs 5
```

For detailed usage, see `examples/huggingface_lora_finetuning.md`.

## Recommended Datasets

These Hugging Face datasets work well for fine-tuning CSM models:

- `mozilla-foundation/common_voice_16_0`: Multi-language crowd-sourced voice dataset
- `openslr/librispeech_asr`: English audiobook recordings
- `facebook/voxpopuli`: Multi-language European Parliament recordings
- `jonatasgrosman/ljspeech_format`: Converted audiobooks in LJSpeech format

## Implementation Details

### LoRA Architecture

The CSM implementation of LoRA follows the original paper with these key components:

1. **LoRALinear**: Base adaptation unit for linear layers that manages low-rank decomposition (matrices A and B) with proper scaling by alpha/r.

2. **LoRATransformerLayer**: Wrapper for transformer layers that selectively applies LoRA to specific projection matrices (query, key, value, etc.).

3. **apply_lora_to_model**: High-level function to apply LoRA to a CSM model, adding helper methods for parameter management and weight merging.

Default configuration:
- Target modules: query and value projection matrices (`q_proj` and `v_proj`)
- Rank (r): 8
- Alpha scaling factor: 16.0
- No dropout (0.0)

### Saving Options

The trainer supports three save modes:

1. **lora**: Save only LoRA parameters (smallest files, requires original model at inference time)
2. **full**: Save the full model with merged weights (largest files, standalone use)
3. **both**: Save both LoRA parameters and merged model (most flexible, largest storage)

### Inference with LoRA Models

**IMPORTANT**: When generating audio with a LoRA-fine-tuned model, you have two options:

1. **Direct use (recommended)**: The latest CSM version directly supports LoRA models for inference without any extra steps:

```python
# Just use the LoRA model directly - it will work without modifications
from csm.mlx.mlx_wrapper import generate_audio
audio = generate_audio(lora_model, "Your text here")

# Or with the MLXGenerator
from csm.mlx.components.generator import MLXGenerator
generator = MLXGenerator(lora_model)
audio = generator.generate("Your text here")
```

2. **Explicit merging**: For performance reasons or to create a standalone model, you can merge LoRA weights with the base model:

```python
# Merge LoRA weights explicitly and use the merged model
merged_model = lora_model.merge_lora_weights()
audio = generate_audio(merged_model, "Your text here")

# Or use the merge_lora parameter to do it automatically 
audio = generate_audio(lora_model, "Your text here", merge_lora=True)
```

Our system now supports both approaches, so you can choose the one that best fits your workflow. Explicit merging can be faster for repeated inference, while direct use is more convenient.

## Testing Infrastructure

CSM includes a comprehensive test script for validating all aspects of LoRA fine-tuning, including the Hugging Face integration:

```bash
# Run all tests
python -m csm.training.test_lora_comprehensive

# Run only Hugging Face integration tests
python -m csm.training.test_lora_comprehensive --test huggingface

# Run real dataset tests (requires network access)
python -m csm.training.test_lora_comprehensive --test huggingface-real
```

The test script validates:

1. Basic LoRA initialization with different configurations
2. CLI fine-tuning functionality
3. Hugging Face integration with local and remote datasets
4. Audio generation fallback mechanisms
5. Different save modes (lora, full, both)
6. Performance benchmarks for different LoRA configurations

### Validating the Hugging Face Workflow

To specifically verify the primary use case of downloading voice samples from Hugging Face, training on that data, and running inference, use:

```bash
python -m csm.training.test_lora_comprehensive --test huggingface-real
```

This test:
1. Downloads a small subset of speech samples from Hugging Face datasets
2. Processes the data for fine-tuning
3. Performs minimal LoRA fine-tuning on the data
4. Generates a sample audio with the fine-tuned model
5. Validates that all steps completed successfully

The test tries multiple datasets to ensure robustness against dataset API changes or network issues.

## Performance Optimization

For best performance on Apple Silicon:

- **M1**: Use rank 4-8, batch size 1-2
- **M2**: Use rank 8-16, batch size 2-4
- **M3**: Use rank 16-32, batch size 4-8

Generally:
- Smaller ranks (4-8) are more efficient for small datasets
- Target modules `q_proj` and `v_proj` offer the best efficiency/performance balance
- Using more target modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`) increases expressivity at the cost of more parameters
- For a rough performance estimate, run the benchmark test:

```bash
python -m csm.training.test_lora_comprehensive --test benchmark
```

## Troubleshooting

Common issues and solutions:

1. **Out of memory errors**: Reduce batch size or LoRA rank
2. **Slow training**: Try reducing the number of target modules or use a smaller rank
3. **Poor adaptation quality**: Try increasing rank, targeting more modules, or increasing epochs
4. **Downloading failures**: Check network connection, try a different dataset or use local data
5. **Audio generation issues**:
   - Our system now directly supports LoRA models for inference, so you can use them without merging
   - For best performance, especially with multiple generations, you can merge LoRA weights with the base model
   - If you hear test tones or silence, check the error logs or error.txt file for detailed information
   - For older CSM versions (before the direct LoRA support), you still need to merge weights explicitly

For detailed validation, run the comprehensive test with specific components:

```bash
# Test audio generation fallbacks
python -m csm.training.test_lora_comprehensive --test fallbacks

# Test save modes
python -m csm.training.test_lora_comprehensive --test save
```