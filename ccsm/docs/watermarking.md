# Audio Watermarking in CCSM

This document provides an overview of the audio watermarking functionality in CCSM, including usage examples and configuration options.

## Overview

CCSM includes robust audio watermarking functionality that allows embedding and detecting hidden information in audio signals. The implemented watermarking algorithm is designed to be:

- **Inaudible**: The watermark is embedded in a way that doesn't noticeably affect audio quality
- **Robust**: The watermark can survive common audio processing operations like volume changes, filtering, and mild compression
- **Verifiable**: Watermarks can be detected and verified with configurable confidence thresholds
- **Payload-capable**: Custom information can be embedded as part of the watermark

The primary implementation is based on the SilentCipher technique, which uses frequency-domain modifications and spread spectrum encoding to create robust watermarks.

## Basic Usage

### Creating a Watermarker

```cpp
#include <ccsm/watermarking.h>

// Create a watermarker with a secret key
auto watermarker = ccsm::Watermarker::create("my_secret_key");

// Or use the specific SilentCipher implementation
auto silent_cipher = ccsm::SilentCipherWatermarker::create("my_secret_key");
```

### Embedding a Watermark

```cpp
// Apply a watermark to audio data
std::vector<float> watermarked = watermarker->apply_watermark(audio_samples);

// Or embed with a specific payload and sample rate
std::vector<float> watermarked = silent_cipher->embed(
    audio_samples,     // Audio data
    24000.0f,          // Sample rate in Hz
    "custom_payload"   // Payload to embed
);
```

### Detecting a Watermark

```cpp
// Basic detection
bool has_watermark = watermarker->detect_watermark(audio_samples);

// Advanced detection with confidence and payload extraction
ccsm::WatermarkResult result = silent_cipher->detect(audio_samples, 24000.0f);
if (result.detected) {
    std::cout << "Watermark detected with confidence: " << result.confidence << std::endl;
    std::cout << "Extracted payload: " << result.payload << std::endl;
}
```

### Verifying a Watermark with a Specific Key

```cpp
// Verify with a specific key
bool is_valid = silent_cipher->verify_watermark(audio_samples, "expected_key");
```

## Configuration Options

The SilentCipher watermarker provides several configuration options:

### Watermark Strength

```cpp
// Set watermark strength (0.0 to 1.0)
silent_cipher->set_strength(0.2f); // Default is 0.1
```

Higher strength values make the watermark more detectable but may affect audio quality.

### Frame and Hop Size

```cpp
// Set analysis frame size (must be a power of 2)
silent_cipher->set_frame_size(2048); // Default is 1024

// Set hop size between consecutive frames
silent_cipher->set_hop_size(512);    // Default is 256
```

Larger frame sizes provide more spectral resolution but less temporal resolution. The hop size controls the overlap between consecutive frames.

## Implementation Details

The watermarking implementation uses the following techniques:

1. **Short-Time Fourier Transform (STFT)**: Converts audio from time domain to frequency domain
2. **Spread Spectrum Encoding**: Distributes watermark information across multiple frequency bins
3. **Error Correction Coding**: Uses redundancy to improve detection reliability
4. **Perceptual Masking**: Focuses on frequency bands where modifications are less audible
5. **Key-based Encryption**: Uses the provided key to generate unique watermarking patterns

## Watermark Robustness

The watermarking implementation has been tested for robustness against:

- Volume changes (scaling)
- Addition of background noise
- Low-pass and high-pass filtering
- Resampling
- Combinations of multiple transformations
- Clipping
- Short segments of corrupted audio

## Performance Considerations

- Watermarking and detection operations scale linearly with audio length
- For real-time applications, consider using smaller audio segments
- Memory usage is proportional to the frame size and audio length

## Testing the Watermarking System

A dedicated test script is provided to verify the watermarking functionality:

```bash
# Run basic watermarking tests
./test_watermarking.sh

# Run with coverage analysis
./test_watermarking.sh --coverage
```

## Limitations

- Very short audio segments (less than 0.5s) may not reliably carry watermarks
- Extreme audio processing (severe compression, significant pitch shifting) may cause watermark loss
- Watermark payload capacity depends on audio length and complexity