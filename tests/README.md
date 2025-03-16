# CSM Test Suite

This directory contains tests for the CSM (Conversational Speech Model) package. The tests are organized into the following structure:

- `unit/`: Unit tests for individual components/functions
- `integration/`: Tests for interactions between components

## Running Tests

### Basic Usage

To run all tests with coverage reports:

```bash
# Run from project root
python -m pytest

# Run with verbose output
python -m pytest -v

# Run a specific test file
python -m pytest tests/unit/test_model_args.py

# Run a specific test function
python -m pytest tests/unit/test_model_args.py::test_model_args_initialization
```

### Coverage Reports

The test suite is configured to generate coverage reports automatically when you run pytest. Reports are generated in the following formats:

- **Terminal report**: Shows missing lines in the terminal
- **HTML report**: Detailed interactive HTML report in `htmlcov/` directory
- **XML report**: XML report in `coverage.xml` for CI integration

To view the HTML coverage report:

```bash
# Run tests with coverage
python -m pytest

# Open the HTML report
open htmlcov/index.html  # On macOS
# or
firefox htmlcov/index.html  # On Linux
```

## Writing Tests

When adding new functionality to CSM, please also add appropriate tests:

1. **Unit tests**: Test individual functions or classes in isolation
   - Place in `tests/unit/`
   - Name files as `test_*.py`
   - Focus on testing one component thoroughly

2. **Integration tests**: Test interactions between components
   - Place in `tests/integration/`
   - Test realistic usage scenarios
   - May involve multiple components working together

Follow these best practices:

- Keep tests small and focused
- Use descriptive names for test functions
- Include docstrings explaining what each test is checking
- Use proper assertions with helpful error messages
- Parameterize tests to check multiple scenarios when appropriate

## Testing Patterns

Here are a few common patterns used in the test suite:

### Fixtures

Use pytest fixtures to set up common test prerequisites:

```python
@pytest.fixture
def sample_model_args():
    return ModelArgs(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128256,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
    )

def test_something(sample_model_args):
    # Use sample_model_args in your test
    assert sample_model_args.backbone_flavor == "llama-1B"
```

### Parameterized Tests

Test multiple scenarios with parameterized tests:

```python
@pytest.mark.parametrize("input_value,expected", [
    (1, 1),
    (2, 4),
    (3, 9),
])
def test_square(input_value, expected):
    assert input_value ** 2 == expected
```