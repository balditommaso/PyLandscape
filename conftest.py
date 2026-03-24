"""
Pytest configuration and fixtures for PyLandscape.
"""
import os
import sys
import tempfile
import torch
import pytest
import numpy as np
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

import pylandscape


class TorchSeedContext:
    """Context manager to set a random seed for torch."""
    def __init__(self, seed=42):
        self.seed = seed

    def __enter__(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture(scope="function")
def torch_seed():
    """Fixture that sets a random seed for torch."""
    return TorchSeedContext()


@pytest.fixture(scope="function")
def temp_dir():
    """Fixture that creates a temporary directory for test outputs."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    # Cleanup after test
    import shutil
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)


@pytest.fixture(scope="function")
def dummy_model():
    """Fixture that creates a simple dummy model for testing."""
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 2)

        def forward(self, x):
            return self.linear(x)

    return SimpleModel()


@pytest.fixture(scope="function")
def dummy_data():
    """Fixture that creates dummy data for testing."""
    # Create dummy input and target
    input_data = torch.randn(32, 10)
    target = torch.randint(0, 2, (32,))
    return input_data, target


@pytest.fixture(scope="function")
def device():
    """Fixture that provides a torch device (GPU if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture(scope="function")
def hessian_model(dummy_model, dummy_data):
    """Fixture that creates a model with Hessian capabilities for testing."""
    # Add a criterion to the model
    dummy_model.criterion = torch.nn.CrossEntropyLoss()
    dummy_model.to("cpu")

    # Ensure model is in eval mode for Hessian computation
    dummy_model.eval()

    return dummy_model


@pytest.fixture(scope="function")
def ckable_models():
    """Fixture that creates multiple models for CKA testing."""
    class Model1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Linear(10, 5)
            self.layer2 = torch.nn.Linear(5, 2)

        def forward(self, x):
            x = torch.relu(self.layer1(x))
            return self.layer2(x)

    class Model2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Linear(10, 5)
            self.layer2 = torch.nn.Linear(5, 2)

        def forward(self, x):
            x = torch.relu(self.layer1(x))
            return self.layer2(x)

    return Model1(), Model2()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "cuda: marks tests that require GPU (deselect with '-m \"not cuda\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )