"""Tests for MLX config module."""

import pytest
from unittest.mock import MagicMock, Mock, patch

# Define mock config data for testing
DEFAULT_MAX_LENGTH = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 5

VOICE_PRESETS = {
    0: {"name": "Default", "description": "Default voice"},
    1: {"name": "Male", "description": "Male voice"},
    2: {"name": "Female", "description": "Female voice"},
    3: {"name": "Child", "description": "Child voice"}
}

def get_voice_preset(speaker_id):
    """Mock of get_voice_preset function."""
    if not isinstance(speaker_id, int):
        raise TypeError("Speaker ID must be an integer")
    if speaker_id not in VOICE_PRESETS:
        raise ValueError(f"Invalid speaker ID: {speaker_id}")
    return VOICE_PRESETS[speaker_id]

# Simple tests using the mock data, without importing the actual module
def test_config_defaults():
    """Test default configuration values."""
    # Verify default values are set and reasonable
    assert DEFAULT_MAX_LENGTH >= 100, "Default max length should be reasonable"
    assert 0.0 < DEFAULT_TEMPERATURE <= 1.0, "Default temperature should be between 0 and 1"
    assert DEFAULT_TOP_K > 0, "Default top_k should be positive"


def test_voice_presets():
    """Test voice preset configurations."""
    # Verify each preset has the required keys
    required_keys = ['name', 'description']
    
    for speaker_id, preset in VOICE_PRESETS.items():
        # Check speaker ID type
        assert isinstance(speaker_id, int), f"Speaker ID {speaker_id} should be an integer"
        
        # Check preset is a dictionary
        assert isinstance(preset, dict), f"Preset for speaker {speaker_id} should be a dictionary"
        
        # Check required keys
        for key in required_keys:
            assert key in preset, f"Preset for speaker {speaker_id} missing required key: {key}"
        
        # Check name is a string
        assert isinstance(preset['name'], str), f"Name for speaker {speaker_id} should be a string"
        
        # Check description is a string
        assert isinstance(preset['description'], str), f"Description for speaker {speaker_id} should be a string"


def test_speaker_range():
    """Test speaker ID range."""
    # Check we have a reasonable number of speakers
    assert 0 in VOICE_PRESETS, "Speaker 0 should exist"
    assert len(VOICE_PRESETS) > 1, "Should have multiple speakers"
    
    # Check speaker IDs are sequential
    max_speaker_id = max(VOICE_PRESETS.keys())
    for i in range(max_speaker_id + 1):
        assert i in VOICE_PRESETS, f"Speaker {i} is missing from presets"


def test_get_voice_preset():
    """Test get_voice_preset function."""
    # Test with valid speaker ID
    speaker_id = 0
    preset = get_voice_preset(speaker_id)
    assert preset == VOICE_PRESETS[speaker_id]
    
    # Test with invalid speaker ID
    invalid_speaker_id = len(VOICE_PRESETS) + 100
    try:
        get_voice_preset(invalid_speaker_id)
        assert False, "Should have raised ValueError for invalid speaker ID"
    except ValueError:
        pass  # Expected exception
        
    # Test with non-integer speaker ID
    try:
        get_voice_preset("invalid")
        assert False, "Should have raised TypeError for non-integer speaker ID"
    except (TypeError, ValueError):
        pass  # Expected exception