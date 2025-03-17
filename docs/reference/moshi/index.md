# Moshi Documentation

## Overview

Moshi is a revolutionary full-duplex speech-text model architecture developed by Kyutai Labs that enables real-time, conversational speech interactions with unprecedented naturalness. As the first full-duplex large language model for speech, Moshi can simultaneously listen and speak, creating a truly interactive conversation experience that closely mimics human-to-human dialogue.

Unlike traditional voice assistants that rely on separate modules for speech recognition (ASR), language processing (NLP), and speech synthesis (TTS), Moshi treats conversation as a single end-to-end process. It directly **generates speech from speech**, using text only as an internal intermediate representation, preserving non-verbal cues like emotion and intonation while enabling natural overlapping speech without rigid turn-taking.

## Key Capabilities

- **Full-duplex conversation**: Simultaneously processes incoming speech and generates outgoing speech with ~200ms latency
- **Overlapping dialogue**: Handles interruptions and backchanneling (e.g., "mm-hmm") while the user is speaking
- **Preserves non-verbal cues**: Maintains emotion, intonation, and conversational dynamics
- **End-to-end architecture**: Single unified model rather than a pipeline of separate components
- **Low latency**: Achieves ~200ms end-to-end latency on an NVIDIA L4 GPU or Apple M3 chip
- **Cross-modal understanding**: Processes audio content directly without requiring explicit transcription

## Technical Specifications

- **Model architecture**: Dual-transformer hierarchy (Temporal + Depth transformers)
- **Backbone model**: 7-billion parameter "Helium" Temporal Transformer (LLaMA-like architecture)
- **Audio representation**: Mimi neural audio codec with semantic and acoustic tokens at 12.5 Hz
- **Training data**: 7 million hours of audio, synthetic conversations, and Fisher telephone corpus
- **Context window**: 4,096 tokens
- **Bitrate**: Compresses audio to ~1.1 kbps representation
- **Sampling rate**: 24 kHz output audio

## Documentation Structure

This documentation has been split into multiple sections for better readability and organization. Each page focuses on a specific aspect of the Moshi model, providing detailed technical insights suitable for ML engineers looking to understand or recreate the system.

### Table of Contents

1. **[Architecture and Components](architecture.md)**
   - [Helium: Text Language Model Backbone (7B LLM)](architecture.md#helium-text-language-model-backbone-7b-llm) - The foundational language model that drives understanding and generation
   - [Mimi: Streaming Neural Audio Codec](architecture.md#mimi-streaming-neural-audio-codec) - The audio compression system that converts between waveforms and discrete tokens
   - [Multi-Stream Modeling: Temporal & Depth Transformers](architecture.md#multi-stream-modeling-temporal--depth-transformers-for-dual-audio-streams) - The hierarchical transformer architecture that processes parallel audio streams

2. **[Training Procedures](training.md)**
   - [Datasets and Preprocessing](training.md#datasets-and-preprocessing) - The massive 7M hour dataset and four-phase training approach
   - [Loss Functions and Optimization](training.md#loss-functions-and-optimization) - Technical details on loss weighting, acoustic token delay, and optimization strategies

3. **[Inference and Processing](inference.md)**
   - [Full-Duplex Streaming Mechanism](inference.md#full-duplex-streaming-mechanism) - How Moshi processes continuous audio frames in real-time
   - [Real-Time Dialogue Behavior](inference.md#real-time-dialogue-behavior) - Implementation of backchanneling, interruption handling, and dynamic interaction
   - [Deployment, Scalability, and Hardware](inference.md#deployment-scalability-and-hardware) - Hardware requirements and optimization techniques

4. **[Model Architecture Details](model_architecture.md)**
   - [Topology and Component Interaction](model_architecture.md#topology-and-component-interaction) - Comprehensive breakdown of model structure and data flow
   - [Helium and Depth Transformer Details](model_architecture.md#helium-and-depth-transformer-details) - Specific architecture parameters for both transformer components
   - [Mimi Audio Codec Architecture](model_architecture.md#mimi-audio-codec-architecture) - Technical design of the neural audio codec
   - [Input/Output Representations and Tokenization](model_architecture.md#inputoutput-representations-and-tokenization) - How text and audio are represented and processed

5. **[Implementation](implementation.md)**
   - [Code Structure and Frameworks](implementation.md#code-structure-and-frameworks) - Suggested code organization and PyTorch implementation
   - [Validation and Testing](implementation.md#validation-and-testing) - Evaluation metrics and testing strategies
   - [Dependencies and Engineering Challenges](implementation.md#dependencies-and-engineering-challenges) - Technical hurdles and solutions for implementation
   - [Performance and Debugging Tools](implementation.md#performance-and-debugging-tools) - Optimization techniques and troubleshooting approaches

6. **[Ethical Considerations](ethics.md)**
   - [Safety Measures](ethics.md#safety-measures) - Controls for toxic content, bias mitigation, and voice usage
   - [Responsible AI Practices](ethics.md#responsible-ai-practices) - Guidelines for ethical deployment
   - [Limitations and Mitigations](ethics.md#limitations-and-mitigations) - Known issues and approaches to address them

## Comparison with CSM

While both Moshi and CSM (Conversational Speech Model) represent significant advances in speech AI, they have different design philosophies:

- **Moshi** is a true **speech-dialogue foundation model** that decides both _what_ to say and _how_ to say it, with integrated speech recognition capabilities
- **CSM** is an advanced **text-to-speech model** that focuses on generating high-quality, contextually appropriate speech, but requires external text input

For a detailed comparison, see the [Comparison with Moshi](../sesame_csm/comparison.md) page in the CSM documentation.

## Resources

- [Kyutai Labs Repository](https://github.com/kyutai-labs/moshi)
- [Mimi Codec on Hugging Face](https://huggingface.co/kyutai/mimi)
- [Helium Model](https://huggingface.co/kyutai/helium-7b)