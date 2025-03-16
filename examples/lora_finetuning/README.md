# CSM LoRA Fine-tuning Examples

This directory contains example scripts for fine-tuning CSM (Conversational Speech Model) models using LoRA (Low-Rank Adaptation) with MLX on Apple Silicon.

## Overview

LoRA is a parameter-efficient fine-tuning technique that significantly reduces memory requirements and training time by only updating a small number of adapter parameters while keeping the base model frozen. These examples demonstrate different LoRA fine-tuning scenarios for CSM text-to-speech models.

## Prerequisites

- Apple Silicon Mac (M1/M2/M3)
- Python 3.11+
- MLX installed (`pip install mlx`)
- CSM package installed (`pip install -e .` from repository root)
- A pre-trained CSM model (safetensors or PyTorch format)

## Example Scripts

### 1. Basic LoRA Fine-tuning

The `basic_lora_finetune.py` script demonstrates the simplest way to fine-tune a CSM model using LoRA for voice adaptation.

**Usage:**

```bash
python basic_lora_finetune.py \
  --model-path /path/to/model.safetensors \
  --audio-dir /path/to/audio \
  --transcript-dir /path/to/transcripts \
  --output-dir ./basic_finetuned \
  --speaker-id 0 \
  --epochs 5 \
  --batch-size 2 \
  --learning-rate 1e-4 \
  --generate-sample
```

This uses standard LoRA settings (r=8, alpha=16) targeting the query and value projection matrices in the transformer layers, which is a good default configuration for most voice adaptation tasks.

### 2. Style Transfer Fine-tuning

The `style_transfer_finetune.py` script demonstrates how to use LoRA for style transfer, such as adapting a model to generate emotional, whispered, or accented speech.

**Usage:**

```bash
python style_transfer_finetune.py \
  --model-path /path/to/model.safetensors \
  --audio-dir /path/to/style/audio \
  --transcript-dir /path/to/style/transcripts \
  --output-dir ./style_finetuned \
  --style-name "cheerful" \
  --lora-r 32 \
  --lora-alpha 64 \
  --target-modules q_proj v_proj o_proj \
  --epochs 10 \
  --learning-rate 5e-5 \
  --semantic-weight 120.0 \
  --generate-samples \
  --sample-prompts "This is a happy message!" "How wonderful to see you today!"
```

Style transfer uses more aggressive LoRA settings with higher rank and alpha values to better capture style characteristics. It also targets more modules (output projection included) to enhance expressivity.

### 3. Low-Resource Fine-tuning

The `low_resource_finetune.py` script demonstrates how to fine-tune a CSM model with minimal data (5-10 minutes of audio) while preventing overfitting.

**Usage:**

```bash
python low_resource_finetune.py \
  --model-path /path/to/model.safetensors \
  --audio-dir /path/to/limited/audio \
  --transcript-dir /path/to/limited/transcripts \
  --output-dir ./low_resource_finetuned \
  --augmentation \
  --augmentation-factor 3 \
  --lora-r 4 \
  --lora-alpha 8 \
  --target-modules q_proj v_proj \
  --epochs 15 \
  --batch-size 1 \
  --learning-rate 5e-5 \
  --weight-decay 0.05 \
  --generate-samples
```

Low-resource adaptation uses a smaller LoRA rank, higher regularization via weight decay and dropout, and optional data augmentation to maximize the effectiveness of limited training data.

### 4. Multi-Speaker Fine-tuning

The `multi_speaker_finetune.py` script demonstrates how to fine-tune multiple speakers in a single training run, sharing common components for efficiency while maintaining speaker-specific adaptation layers.

**Usage:**

```bash
python multi_speaker_finetune.py \
  --model-path /path/to/model.safetensors \
  --speakers-config speakers_config_example.json \
  --output-dir ./multi_speaker_finetuned \
  --share-backbone \
  --lora-r 8 \
  --lora-alpha 16.0 \
  --target-modules q_proj v_proj \
  --epochs 10 \
  --batch-size 2 \
  --learning-rate 5e-5 \
  --generate-samples \
  --merge-models \
  --shared-weight 0.5
```

Multi-speaker fine-tuning requires a configuration file (`speakers_config_example.json`) that defines the speakers, their IDs, and paths to their audio and transcript files:

```json
{
    "speakers": [
        {
            "id": 0,
            "name": "Speaker1",
            "audio_dir": "/path/to/speaker1/audio",
            "transcript_dir": "/path/to/speaker1/transcripts"
        },
        {
            "id": 1,
            "name": "Speaker2",
            "audio_dir": "/path/to/speaker2/audio",
            "transcript_dir": "/path/to/speaker2/transcripts"
        }
    ]
}
```

This approach enables efficient multi-speaker training by sharing the backbone transformer but keeping speaker-specific layers, significantly reducing training time and memory requirements compared to training each speaker individually.

## Tips for Effective Fine-tuning

### General Tips

1. **Audio Quality Matters**: Use high-quality audio without background noise for best results
2. **Transcript Accuracy**: Ensure transcripts match the audio precisely
3. **Start Small**: Begin with lower LoRA rank (4-8) and increase if needed
4. **Monitor Validation Loss**: Early stopping based on validation loss helps prevent overfitting
5. **Save Checkpoints**: Save frequent checkpoints to find the best model version

### ⚠️ Important: Generating Audio with LoRA Models

When generating audio with a LoRA-fine-tuned model, you have two options:

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

### Hyperparameters to Adjust

* **LoRA rank (r)**: Controls capacity of the adapter (4-32)
  * Smaller values (4-8): Good for voice adaptation with limited data
  * Larger values (16-32): Better for style transfer or with lots of data
  
* **LoRA alpha**: Scaling factor, typically 2x the rank value
  * Higher values increase the impact of LoRA updates

* **Target modules**: Which attention components to adapt
  * Minimal (q_proj, v_proj): Most efficient, good for voice adaptation
  * Extended (q_proj, k_proj, v_proj, o_proj): Better for style transfer
  * Full (all attention + MLP components): Most expressive but needs more data

* **Learning rate**: Typical values between 1e-5 and 1e-4
  * Start with 5e-5 and adjust based on training dynamics

## Adding Your Own Examples

To create your own fine-tuning example:

1. Create audio files (WAV format)
2. Create matching transcript files (.txt with same base filename)
3. Organize them in directories (audio_dir and transcript_dir)
4. Run the appropriate fine-tuning script with your settings

## Multi-speaker Support

For multi-speaker models, use the `--speaker-id` parameter to specify which speaker ID to use during training and generation. Different speaker IDs can be fine-tuned separately with different LoRA weights.

## Additional Resources

For more information about LoRA fine-tuning with CSM models, check the detailed documentation:

- [LoRA Fine-tuning Documentation](../../docs/reference/sesame_csm/lora_finetuning.md)
- [CSM Project Documentation](../../docs/README.md)
- [Original LoRA Paper](https://arxiv.org/abs/2106.09685)