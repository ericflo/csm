# CSM Reference Documentation

This directory contains authoritative technical reference documentation for the Conversational Speech Model (CSM) and related speech AI technologies. The documentation provides in-depth technical details about model architectures, training methodologies, inference strategies, and implementation considerations, serving as a comprehensive resource for researchers and developers.

## Documentation Structure

The reference documentation is organized into separate directories for each major technology, with a focus on comprehensive technical explanations:

### Moshi Documentation

The [moshi/](./moshi/) directory contains extensive documentation about Moshi, a full-duplex speech-text model architecture developed by Kyutai Labs that enables real-time, concurrent spoken conversation.

- [moshi/index.md](./moshi/index.md): Overview and table of contents for Moshi documentation
- [moshi/architecture.md](./moshi/architecture.md): Core architecture including Helium (7B LLM backbone), Mimi (streaming neural audio codec), and multi-stream transformer design
- [moshi/training.md](./moshi/training.md): Detailed training procedures covering the 7M hour dataset, multi-phase training approach, and optimization techniques
- [moshi/inference.md](./moshi/inference.md): Full-duplex streaming mechanisms achieving ~200ms latency, real-time dialogue behavior, and deployment considerations
- [moshi/model_architecture.md](./moshi/model_architecture.md): In-depth breakdown of model topology, transformer specifications, and input/output representations
- [moshi/implementation.md](./moshi/implementation.md): Technical implementation details including code structure, validation strategies, and optimization techniques
- [moshi/ethics.md](./moshi/ethics.md): Comprehensive ethical considerations around voice synthesis, bias mitigation, and privacy safeguards

### Sesame CSM Documentation

The [sesame_csm/](./sesame_csm/) directory contains comprehensive technical documentation about Sesame AI Lab's Conversational Speech Model, which represents the foundation for this project.

- [sesame_csm/index.md](./sesame_csm/index.md): Overview and table of contents for CSM documentation
- [sesame_csm/introduction.md](./sesame_csm/introduction.md): Introduction to Sesame AI Lab's "voice presence" concept and CSM's end-to-end multimodal approach
- [sesame_csm/architecture.md](./sesame_csm/architecture.md): Technical breakdown of the dual-transformer framework with backbone and audio decoder components
- [sesame_csm/components.md](./sesame_csm/components.md): Detailed examination of the LLaMA-based 8B backbone, contextual understanding capabilities, Mimi RVQ tokenizer, and emotion modeling
- [sesame_csm/training.md](./sesame_csm/training.md): Comprehensive overview of the 1M hour training dataset, hyperparameters, compute amortization techniques, and fine-tuning methods
- [sesame_csm/inference.md](./sesame_csm/inference.md): Analysis of low-latency inference strategies achieving ~380ms response time, token efficiency, and multi-speaker scenario handling
- [sesame_csm/implementation.md](./sesame_csm/implementation.md): Available resources including GitHub repository, pretrained 1B model, and Hugging Face integration
- [sesame_csm/comparison.md](./sesame_csm/comparison.md): Technical comparison between CSM and Moshi architectures, highlighting differences in full-duplex capabilities, language modeling integration, and efficiency
- [sesame_csm/considerations.md](./sesame_csm/considerations.md): Step-by-step development plan for recreating CSM, compute requirements, and potential implementation challenges

## Architectural Comparison

### Moshi Architecture

Moshi employs a **full-duplex, three-stream architecture** that enables concurrent listening and speaking:

- **Helium Temporal Transformer** (7B): Models time steps at 12.5 Hz across multiple streams
- **Depth Transformer**: Generates multiple audio codec tokens per time step
- **Mimi Audio Codec**: Tokenizes speech into discrete tokens at 12.5 Hz (1.1 kbps)
- **Features**: Processes two audio streams simultaneously (user input and AI output) with minimal latency (~200ms)
- **Streaming capability**: Can listen and speak concurrently with natural backchanneling and interruption handling

### CSM Architecture

CSM uses a **dual-transformer, turn-based architecture** focused on high-quality speech generation:

- **Backbone Transformer** (1B/3B/8B): Processes text and audio context to generate semantic audio tokens
- **Audio Decoder** (~300M): Generates acoustic details based on semantic tokens
- **Mimi Audio Codec**: Same tokenization approach as Moshi (shared technology)
- **Features**: Maintains 2048 tokens of conversational context (approximately 2 minutes of dialogue)
- **Response latency**: Achieves ~380ms end-to-end latency for high-quality speech synthesis

## Model Capabilities and Applications

The documented speech models enable advanced applications in conversational AI:

- **Moshi**: Full-duplex, real-time spoken dialogue with concurrent speech input and output, enabling natural conversation flow with overlapping speech, backchanneling, and dynamic interaction
- **CSM**: High-quality, context-aware speech generation with natural prosody and emotion, optimized for realistic voice assistants that understand conversation history and respond appropriately

## Technical Specifications

| Feature | Moshi | CSM |
|---------|-------|-----|
| Model Architecture | Global-local dual transformer | Context-decoder dual transformer |
| Parameter Count | 7B (temporal) + smaller depth transformer | 1B/3B/8B (backbone) + 300M (decoder) |
| Audio Representation | Mimi codec (12.5 Hz, 8 codebooks) | Mimi codec (12.5 Hz, 8 codebooks) |
| Context Length | Variable (streaming) | 2048 tokens (~2 minutes) |
| End-to-End Latency | ~200ms | ~380ms |
| Speech Recognition | Integrated | External |
| Language Generation | Integrated | External |
| Compute Requirements | NVIDIA L4 GPU / Apple M3 | RTX 4090 or equivalent |

## Implementation Considerations

Both models represent significant advances in speech AI, but have different implementation requirements:

- **Moshi**: More complex integration due to full-duplex capabilities, but potentially more natural for continuous dialogue
- **CSM**: Easier to integrate with existing LLMs and ASR systems, better suited for voice assistants and turn-based interfaces

## Important Notice on Contributions

**Contributions to this reference directory are not welcomed.**

The `reference/` directory is exclusively for documentation provided by third-party providers and original authors. This documentation serves as authoritative reference material and should remain unchanged to preserve its accuracy and authority.

If you wish to create new documentation:

1. Place it outside the `reference/` directory in the main `docs/` folder
2. Create appropriate subdirectories as needed for organization
3. Only add content to the `reference/` directory if explicitly requested by project maintainers

This policy ensures that reference documentation remains clearly distinguished from community contributions and maintains its integrity as authoritative source material.