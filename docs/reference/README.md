# CSM Reference Documentation

This directory contains technical reference documentation for the Conversational Speech Model (CSM) and related technologies.

## Documentation Structure

The documentation is organized into separate directories for each major technology:

### Moshi Documentation

The [moshi/](./moshi/) directory contains comprehensive documentation about Moshi, a full-duplex speech-text model architecture developed by Kyutai Labs.

- [moshi/index.md](./moshi/index.md): Overview and table of contents for all Moshi documentation
- [moshi/architecture.md](./moshi/architecture.md): Core components and architectural design of the Moshi model
- [moshi/training.md](./moshi/training.md): Training procedures, datasets, and optimization techniques
- [moshi/inference.md](./moshi/inference.md): Real-time inference, streaming, and deployment strategies
- [moshi/model_architecture.md](./moshi/model_architecture.md): Detailed technical breakdown of model topology and components
- [moshi/implementation.md](./moshi/implementation.md): Code structure, validation, and engineering challenges
- [moshi/ethics.md](./moshi/ethics.md): Ethical considerations, safety measures, and responsible usage guidelines

### Sesame CSM Documentation

The [sesame_csm/](./sesame_csm/) directory contains detailed documentation about Sesame AI Lab's Conversational Speech Model.

- [sesame_csm/index.md](./sesame_csm/index.md): Overview and table of contents for all CSM documentation
- [sesame_csm/introduction.md](./sesame_csm/introduction.md): Introduction to Sesame AI Lab and the CSM project goals
- [sesame_csm/architecture.md](./sesame_csm/architecture.md): Dual-transformer architecture and design principles
- [sesame_csm/components.md](./sesame_csm/components.md): Technical components including backbone transformer and Mimi codec
- [sesame_csm/training.md](./sesame_csm/training.md): Training methodology, data processing, and compute amortization techniques
- [sesame_csm/inference.md](./sesame_csm/inference.md): Low-latency inference strategies and deployment optimization
- [sesame_csm/implementation.md](./sesame_csm/implementation.md): Open-source code resources and implementation guides
- [sesame_csm/comparison.md](./sesame_csm/comparison.md): Detailed comparison between CSM and Moshi architectures
- [sesame_csm/considerations.md](./sesame_csm/considerations.md): Practical considerations for implementing CSM from scratch

## Documentation Format

All documentation is written in Markdown for easy viewing on GitHub and other Markdown renderers. Each file includes:

- A clear title and introduction
- Structured sections with appropriate heading levels
- Navigation links to related documentation
- References to source materials where applicable

## Model Capabilities

The documented systems enable:

- **Moshi**: Full-duplex, real-time spoken dialogue with concurrent speech input and output
- **CSM**: High-quality, context-aware speech generation with natural prosody and emotion

## Important Notice on Contributions

**Contributions to this directory are not welcomed.**

The `reference/` directory is exclusively for documentation provided by third-party providers and original authors. This documentation serves as reference material and should remain unchanged to preserve its accuracy and authority.

If you wish to create new documentation:

1. Place it outside the `reference/` directory in the main `docs/` folder
2. Create appropriate subdirectories as needed for organization
3. Only add content to the `reference/` directory if explicitly requested by project maintainers

This policy ensures that reference documentation remains clearly distinguished from community contributions and maintains its integrity as authoritative source material.