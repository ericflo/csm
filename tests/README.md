# CSM Test Suite

This directory contains tests for the CSM (Conversational Speech Model) package. The tests are organized into the following structure:

- `unit/`: Unit tests for individual components/functions
- `integration/`: Tests for interactions between components
- `mocks/`: Mock implementations for testing

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

### MLX-Specific Tests

The CSM codebase includes tests for MLX-accelerated components that require the MLX library, which is only available on Apple Silicon. For systems without MLX, or to test the non-MLX codepaths, you can skip MLX-specific tests using:

```bash
SKIP_MLX_TESTS=1 python -m pytest
```

Alternatively, you can use the command-line option:
```bash
python -m pytest --skip-mlx
```

#### MLX Test Configuration

MLX tests are automatically marked with the `requires_mlx` marker. These tests will be skipped when:
1. MLX is not available on the system
2. `--skip-mlx` flag is used
3. `SKIP_MLX_TESTS=1` environment variable is set

To manually mark a test that requires MLX:
```python
@pytest.mark.requires_mlx
def test_that_needs_mlx():
    # Test implementation
```

#### Running Tests on Systems with MLX

When running on Apple Silicon with MLX installed, you should use the real MLX implementation without any mocking:

```bash
# Run all tests, including MLX tests
python -m pytest
```

Each test file handles MLX imports with a try/except pattern that checks MLX availability:

```python
# Check if MLX is available
try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    pytest.skip("MLX is not available", allow_module_level=True)
```

For better results when running specific MLX tests, try running them in separate groups:

```bash
# Run transformer tests
python -m pytest tests/unit/test_mlx_transformer.py

# Run embedding tests
python -m pytest tests/unit/test_mlx_embedding.py

# Run model wrapper tests
python -m pytest tests/unit/test_mlx_model_wrapper.py
```

#### Running Tests on Systems without MLX

On systems without MLX or when you want to test non-MLX codepaths, use the SKIP_MLX_TESTS environment variable:

```bash
SKIP_MLX_TESTS=1 python -m pytest
```

#### Known Test Issues and Solutions

1. **Test Conflicts**: Some tests may fail when run as part of the full test suite but pass when run individually. This is usually due to cross-test dependencies or shared state.

2. **Mock vs. Real MLX**: Most MLX tests have two execution paths:
   - Using the real MLX library on supported hardware (M1/M2/M3 Mac)
   - Using mocks when MLX is not available or skipped

3. **Mock/Real MLX Conflict**: When testing on systems with real MLX installed, conflicts can occur between:
   - Tests using the real MLX library
   - Tests that attempt to create mocks of MLX modules
   
   This causes errors like:
   ```
   AssertionError: assert <MagicMock name='mock.MLXEmbedding().text_embeddings' id='1325120'> is None
   ```

4. **Cross-Test Contamination**: When running all tests together, there may be interference due to shared state, particularly around modules imported or mocked in one test affecting other tests.

If you encounter MLX test failures:

1. First try running the specific failing tests individually
2. If that passes, try running related groups of tests separately
3. Use `SKIP_MLX_TESTS=1` when testing non-MLX functionality

#### Recent Fixes

The following fixes were applied to improve MLX test stability:

1. **conftest.py**: Updated to properly handle the SKIP_MLX_TESTS environment variable

2. **Import Pattern**: Standardized the MLX import pattern in test files:
   ```python
   # These tests require MLX
   pytestmark = pytest.mark.requires_mlx
   
   # Check if MLX is available
   try:
       import mlx.core as mx
       import mlx.nn as nn
       HAS_MLX = True
   except ImportError:
       HAS_MLX = False
       pytest.skip("MLX is not available", allow_module_level=True)
   ```

3. **Eliminated Mocking Conflicts**: Removed complex mock implementations in multiple test files:
   - test_mlx_wrapper.py
   - test_mlx_generation.py
   - test_mlx_model_wrapper.py
   - test_mlx_ops.py
   - test_mlx_sample_exact.py
   - test_mlx_layers.py
   - test_mlx_generator.py
   - test_mlx_kvcache.py
   - test_mlx_sampling.py
   - test_components_sampling.py

4. **Fixed Module-Level Mocking**: Removed problematic module-level mocking that was interfering with real MLX:
   ```python
   # BEFORE (removed this)
   sys.modules['mlx'] = mx
   sys.modules['mlx.core'] = mx.core
   sys.modules['mlx.nn'] = mx.nn
   ```

5. **Test-Level Patching**: Switched to test-level patching rather than module-level:
   ```python
   # Instead of module-level mocks, use function-level patches
   def test_something():
       with patch('csm.mlx_accel.mlx_ops.torch_to_mlx') as mock_torch_to_mlx:
           # Test code here
   ```

6. **Improved Test Documentation**: Added guidance for running MLX tests effectively

The MLX test configuration is managed in `tests/conftest.py`.

#### Troubleshooting Remaining Issues

If you encounter test failures with MLX-related tests:

1. **Use Specific Test Selection**:
   ```bash
   # Run just one test
   python -m pytest tests/unit/test_mlx_embedding.py::test_mlx_embedding_initialization
   ```

2. **Run Tests in Groups by File**:
   ```bash
   # Run all tests in a specific file
   python -m pytest tests/unit/test_mlx_transformer.py
   ```

3. **Skip MLX for CI/CD**: For CI/CD systems without MLX, use:
   ```bash
   SKIP_MLX_TESTS=1 python -m pytest
   ```

4. **Check for Mock Conflicts**: If tests fail with errors like:
   ```
   AssertionError: assert <MagicMock name='mock.MLXEmbedding().text_embeddings'> is None
   ```

   This indicates a conflict between mocking and real MLX. For these cases:
   - Make sure the test file follows the standardized import pattern above
   - Use function-level patches instead of module-level mocks
   - Consider running the test file separately from other tests

Remember that `pytestmark = pytest.mark.requires_mlx` combined with the environment variable `SKIP_MLX_TESTS=1` will skip all MLX tests, which is useful for running on systems without MLX.

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