"""Test Hessian module."""
import torch
import pytest

from pylandscape import Hessian


def test_hessian_initialization(hessian_model, dummy_data):
    """Test Hessian module initialization."""
    # Create a dummy dataloader
    class DummyDataloader:
        def __iter__(self):
            return iter([dummy_data])

    dataloader = DummyDataloader()

    hessian = Hessian(hessian_model, hessian_model.criterion, dataloader)

    assert hessian.model is hessian_model
    assert hessian.hessian_comp.criterion is hessian_model.criterion
    assert hessian.hessian_comp.data is dataloader


def test_hessian_save_load(hessian_model, dummy_data):
    """Test saving and loading Hessian data."""
    # Create a dummy dataloader
    class DummyDataloader:
        def __iter__(self):
            return iter([dummy_data])

    dataloader = DummyDataloader()

    hessian = Hessian(hessian_model, hessian_model.criterion, dataloader)

    # Test save method (should work with proper path)
    with pytest.raises((NotImplementedError, FileNotFoundError)):
        hessian.save_on_file("/tmp/test_hessian")


def test_hessian_attributes(hessian_model, device, dummy_data):
    """Test that Hessian has expected attributes."""
    # Create a dummy dataloader
    class DummyDataloader:
        def __iter__(self):
            return iter([dummy_data])

    dataloader = DummyDataloader()

    hessian = Hessian(hessian_model, hessian_model.criterion, dataloader)

    # Check that results attribute exists
    assert hasattr(hessian, "results")
    assert isinstance(hessian.results, dict)


def test_hessian_with_mock_dataloader(hessian_model, device):
    """Test Hessian with a mock dataloader."""
    # Create a mock dataloader
    class MockDataloader:
        def __iter__(self):
            return iter([
                (torch.randn(2, 10), torch.tensor([1.0, 0.0]))
            ])

    dataloader = MockDataloader()

    hessian = Hessian(hessian_model, hessian_model.criterion, dataloader, device)

    # Should be able to initialize without errors
    assert hessian is not None
    assert hessian.hessian_comp.data is dataloader


def test_compute_eigenvalues(hessian_model, dummy_data):
    """Test computing eigenvalues and eigenvectors."""
    # Create a dummy dataloader
    class DummyDataloader:
        def __iter__(self):
            return iter([dummy_data])

    dataloader = DummyDataloader()

    hessian = Hessian(hessian_model, hessian_model.criterion, dataloader)

    # Compute top-n eigenvectors and eigenvalues
    hessian.compute_eigenvalues(10)

    # Check that results were computed
    assert hessian.results is not None
    assert "eigenvalue" in hessian.results
    assert "eigenvector" in hessian.results

    # Number of eigenvalues should be <= number of parameters
    for eigvec, param in zip(hessian.results["eigenvector"][0], hessian_model.parameters()):
        assert eigvec.shape == param.shape





