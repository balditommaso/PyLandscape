# PyLandscape Developer Guide

This guide provides comprehensive instructions for developers working on the PyLandscape project, from setting up the development environment to publishing new versions.

## Table of Contents

- [Installation](#installation)
- [Development Environment](#development-environment)
- [Project Structure](#project-structure)
- [Adding New Features](#adding-new-features)
- [Writing Tests](#writing-tests)
- [Running Tests](#running-tests)
- [Code Quality](#code-quality)
- [Publishing a New Version](#publishing-a-new-version)

## Installation

### Prerequisites

- Python 3.9 or higher
- Poetry (Python dependency management)
- Git
- (Optional) CUDA for GPU-based testing

### Installing Poetry

If you don't have Poetry installed, you can install it via:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```
<!-- 
Add Poetry to your PATH:

```bash
echo "$HOME/.local/bin" >> $GITHUB_PATH
``` -->

### Installing the Project

Clone the repository and install dependencies:

```bash
git clone https://github.com/balditommaso/PyLandscape.git
cd PyLandscape

# (if needed)
poetry lock

# Install the project in development mode
poetry install --with dev

# Or install all dependencies including extras
poetry install --all-extras --no-interaction
```

### Setting Up a Virtual Environment (Optional)

If you prefer not to use Poetry's built-in virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

## Development Environment

### Project Dependencies

The main dependencies are defined in `pyproject.toml`:

- **Core**: Python, NumPy, Pandas, SciPy, PyTorch, tqdm, PyHessian
- **Development**: pytest, pytest-cov, pytest-mock, torchvision

### Verifying Installation

Test that PyLandscape is properly installed:

```bash
python -c "import pylandscape; print('PyLandscape imported successfully')"
```

## Project Structure

```
pylandscape/
├── pylandscape/           # Main package
│   ├── __init__.py
│   ├── cka.py            # Centered Kernel Alignment
│   ├── hessian.py        # Hessian-based metrics
│   ├── metric.py         # General metrics
│   ├── mode_connectivity.py  # Mode connectivity analysis
│   └── surface.py        # Loss surface metrics
├── models/               # Neural network models
│   ├── autoencoder.py
│   ├── image_classification.py
│   ├── quantization/
│   └── utils.py
├── datamodule/           # Data loading modules
│   ├── vision_datamodule.py
│   └── utils.py
├── benchmarks/           # Benchmark implementations
│   ├── jacobian.py
│   ├── noise.py
│   └── lipschitz.py
├── tests/                # Test files
│   ├── test_*.py
│   └── conftest.py       # Pytest fixtures
├── scripts/              # Training and testing scripts
├── config/               # Configuration files
├── notebooks/            # Jupyter notebooks
├── pyproject.toml        # Poetry configuration
└── README.md             # User documentation
```

## Adding New Features

### Feature Development Workflow

1. **Design the Feature**
   - Review existing code structure
   - Determine which module the feature belongs to
   - Consider backward compatibility

2. **Implement the Feature**
   - Place code in appropriate module
   - Follow existing code style and patterns
   - Add docstrings and type hints

3. **Write Tests First** (Test-Driven Development)
   - Create test file in `tests/`
   - Write tests for the new feature
   - See [Writing Tests](#writing-tests)

4. **Run Tests**
   - Ensure all tests pass locally
   - See [Running Tests](#running-tests)

5. **Update Documentation**
   - Update README if needed
   - Add docstrings to public APIs
   - Document any breaking changes

### Example: Adding a New Metric

```python
# In pylandscape/metric.py

def your_new_metric(model, dataloader, **kwargs):
    """
    Calculate your new metric.

    Args:
        model: The neural network model
        dataloader: DataLoader for evaluation
        **kwargs: Additional arguments

    Returns:
        float: The computed metric value
    """
    # Implementation here
    pass
```

## Writing Tests

### Test File Structure

Test files should follow this pattern:

```python
# tests/test_your_feature.py

import pytest
import torch
import numpy as np
import pylandscape

class TestYourFeature:
    """Tests for your new feature."""

    def test_basic_functionality(self):
        """Test basic functionality of the feature."""
        # Setup
        model = create_model()
        data = create_data()

        # Execute
        result = your_feature(model, data)

        # Assert
        assert result is not None
        assert isinstance(result, (int, float, np.ndarray))

    def test_with_invalid_input(self):
        """Test feature with invalid input."""
        # Setup
        model = create_model()

        # Assert
        with pytest.raises(ValueError):
            your_feature(model, invalid_data)

    def test_different_dtypes(self):
        """Test feature with different data types."""
        # Test with various input types
        pass

    @pytest.mark.slow
    def test_large_dataset(self):
        """Test feature on larger datasets (slow test)."""
        # Large dataset test
        pass
```

### Test Naming Convention

- Files: `test_*.py`
- Classes: `Test*` (e.g., `TestHessianMetrics`)
- Test functions: `test_*` (e.g., `test_centered_kernel_alignment`)

### Fixtures

Use the provided fixtures in `conftest.py`:

- `torch_seed`: Sets random seed for reproducibility
- `dummy_model`: Creates a simple test model
- `dummy_data`: Creates test data
- `device`: Provides GPU if available, else CPU
- `hessian_model`: Model with Hessian capabilities
- `ckable_models`: Multiple models for CKA testing
- `temp_dir`: Temporary directory for test outputs

### Test Markers

Use pytest markers for categorizing tests:

```python
@pytest.mark.slow  # Tests that take a long time
@pytest.mark.cuda  # Tests requiring GPU (skip on CPU)
@pytest.mark.integration  # Integration tests
```

Run tests excluding specific markers:

```bash
# Skip slow tests
pytest -m "not slow"

# Skip CUDA tests
pytest -m "not cuda"

# Run integration tests only
pytest -m "integration"
```

## Running Tests

### Running All Tests

```bash
# Run all tests with coverage
poetry run pytest tests/ -v --tb=short --cov=pylandscape --cov-report=term-missing

# Run without coverage
poetry run pytest tests/ -v
```

### Running Specific Tests

```bash
# Run a specific test file
poetry run pytest tests/test_hessian.py -v

# Run a specific test function
poetry run pytest tests/test_hessian.py::TestHessian::test_centered_kernel_alignment -v

# Run tests matching a pattern
poetry run pytest tests/ -k "kernel" -v
```

### Running Tests with Custom Markers

```bash
# Skip slow tests
poetry run pytest tests/ -m "not slow" -v

# Run only CUDA tests (if available)
poetry run pytest tests/ -m "cuda" -v

# Run only integration tests
poetry run pytest tests/ -m "integration" -v
```

### Coverage Requirements

The project enforces a minimum 50% test coverage. The CI workflow will fail if coverage drops below this threshold:

```bash
# Check coverage locally
poetry run pytest tests/ --cov=pylandscape --cov-report=term-missing

# Run with coverage threshold check
poetry run pytest tests/ --cov=pylandscape --cov-fail-under=50
```

### Running Tests on Different Environments

```bash
# CPU tests only
poetry run pytest tests/ -m "not cuda" -v

# CUDA tests (if available)
poetry run pytest tests/ -m "cuda" -v
```

## Code Quality

### Code Style

Follow the project's code style. The CI workflow includes flake8 checks:

```bash
# Run flake8 checks
poetry run flake8 pylandscape/ --count --select=E9,F63,F7,F82 --show-source --statistics

# Run flake8 with max score
poetry run flake8 pylandscape/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

### Linting Rules

The project uses flake8 with the following defaults:
- Max complexity: 10
- Max line length: 127
- Critical errors only: E9, F63, F7, F82

### Type Hints

Use type hints for function signatures and variables:

```python
def train_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
    """Train a model and return metrics."""
    pass
```

### Docstrings

Add docstrings to all public functions and classes:

```python
def compute_cka(layer1: torch.Tensor, layer2: torch.Tensor) -> float:
    """
    Compute the Centered Kernel Alignment (CKA) between two layers.

    Args:
        layer1: Output of first layer (B x D)
        layer2: Output of second layer (B x D)

    Returns:
        float: CKA similarity score between layers

    Raises:
        ValueError: If input tensors have mismatched shapes
    """
    pass
```

## Publishing a New Version

### Versioning

PyLandscape uses semantic versioning (MAJOR.MINOR.PATCH). The version is managed in `pyproject.toml`:

```toml
version = "0.0.0"
```

### Preparing a Release

1. **Update the version number** in `pyproject.toml`

```bash
# Increase version (patch)
poetry version patch  # 0.0.1 -> 0.0.2

# Minor version
poetry version minor  # 0.0.2 -> 0.1.0

# Major version
poetry version major  # 0.1.0 -> 1.0.0
```

2. **Write a changelog** (optional but recommended)

3. **Run all tests locally**

```bash
# Run all tests with coverage
poetry run pytest tests/ -v --tb=short --cov=pylandscape --cov-report=term-missing

# Check coverage threshold
poetry run pytest tests/ --cov=pylandscape --cov-fail-under=50
```

4. **Push changes to git**

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "Bump version to X.Y.Z"
git push origin main
```

### Creating a Release Tag

Create a new tag with the version number:

```bash
# Create a tag
git tag v1.0.0

# Push the tag
git push origin v1.0.0
```

The GitHub Actions workflow will automatically:
1. Run tests on multiple Python/PyTorch versions
2. Check test coverage (minimum 50%)
3. Build the package
4. Publish to PyPI
5. Create a GitHub release

### Managing PyPI

1. **Configure PyPI token** in GitHub repository secrets
   - Go to: Repository Settings > Secrets and variables > Actions
   - Add `PYPI_API_TOKEN` with your PyPI API token

2. **Access the release on PyPI**
   - The package will be automatically published after the workflow completes
   - Verify at https://pypi.org/project/pylandscape/

### Creating a Pre-release

For pre-releases (alpha, beta, rc), use appropriate version suffixes:

```bash
# Alpha release
poetry version patch  # 1.0.0 -> 1.0.1
git tag v1.0.1a1
git push origin v1.0.1a1

# Beta release
poetry version patch  # 1.0.1 -> 1.0.2
git tag v1.0.2b1
git push origin v1.0.2b1

# Release candidate
poetry version patch  # 1.0.2 -> 1.0.3
git tag v1.0.3rc1
git push origin v1.0.3rc1
```

### Troubleshooting

**Tests failing locally:**
- Ensure you have the correct Python version (3.9+)
- Check that all dependencies are installed: `poetry install`
- Clear poetry cache: `poetry cache clear --all`

**Coverage threshold not met:**
- Write tests for uncovered code paths
- Use `--cov-report=html` to view detailed coverage report

**PyPI publish failing:**
- Verify PYPI_API_TOKEN is correctly set in GitHub secrets
- Check that version number is unique (not already published)
- Ensure pyproject.toml version is updated

## Common Tasks

### Install in Development Mode

```bash
poetry install --with dev
```

### Update Dependencies

```bash
# Update all dependencies
poetry update

# Update specific dependency
poetry update numpy
```

### View Dependency Tree

```bash
poetry show
```

### Clean Up

```bash
# Remove poetry cache
poetry cache clear --all

# Remove build artifacts
rm -rf dist/ build/
```

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyHessian Documentation](https://github.com/amirgholami/PyHessian)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [Pytest Documentation](https://docs.pytest.org/)
- [CKA Paper](https://arxiv.org/pdf/2010.15327)
- [Hessian Metrics Paper](https://arxiv.org/pdf/1912.07145)
- [Mode Connectivity Paper](https://arxiv.org/pdf/1802.10026)