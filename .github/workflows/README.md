# CSM GitHub Actions Workflows

This directory contains GitHub Actions workflows for the CSM (Conversational Speech Model) project.

## Workflows

### `test.yml` - Run Tests

This workflow runs the test suite and code quality checks for the CSM project.

It includes two jobs:

1. **`test`** - Runs on Ubuntu
   - Runs tests (with MLX tests skipped)
   - Uploads coverage reports
   - Uploads coverage to Codecov (if configured)
   - Note: Linting and type checking are commented out by default

2. **`test-macos`** - Runs on macOS with Apple Silicon
   - Only runs on the main branch or PRs labeled with "run-mlx-tests"
   - Runs all tests including MLX tests
   - Uploads MLX-specific coverage reports

## Running MLX Tests

The macOS job is conditional and will only run in these cases:
- On the main branch
- On pull requests labeled with "run-mlx-tests"

To run MLX tests on a pull request, add the "run-mlx-tests" label to the PR.

## Notes

- The tests use the `SKIP_MLX_TESTS=1` environment variable to skip MLX tests on non-Apple hardware
- Coverage reports are generated in both XML and HTML formats
- The workflows use caching to speed up dependency installation