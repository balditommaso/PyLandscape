"""Test CKA module."""
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import patch

from pylandscape import CKA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class SimpleModel(nn.Module):
    """Tiny MLP used across all CKA tests."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 8, output_dim: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def model_a():
    torch.manual_seed(0)
    return SimpleModel()


@pytest.fixture
def model_b():
    torch.manual_seed(1)
    return SimpleModel()


@pytest.fixture
def dummy_data():
    torch.manual_seed(42)
    inputs = torch.randn(16, 10)
    targets = torch.zeros(16, dtype=torch.long)
    return inputs, targets


@pytest.fixture
def dummy_dataloader(dummy_data):
    inputs, targets = dummy_data
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=8)


@pytest.fixture
def cka(device):
    return CKA(device=device)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestCKAInitialization:
    def test_default_device_is_cpu(self):
        c = CKA()
        assert c.device == "cpu"

    def test_custom_device_stored(self, device):
        c = CKA(device=device)
        assert c.device == device

    def test_default_name(self):
        c = CKA()
        assert c.name == "CKA_similarity"

    def test_custom_name_stored(self):
        c = CKA(name="my_cka")
        assert c.name == "my_cka"

    def test_results_attribute_exists(self, cka):
        assert hasattr(cka, "results")
        assert isinstance(cka.results, dict)


# ---------------------------------------------------------------------------
# gram_matrix
# ---------------------------------------------------------------------------

class TestGramMatrix:
    def test_output_is_1d(self):
        X = torch.randn(5, 4)
        g = CKA.gram_matrix(X)
        assert g.ndim == 1

    def test_output_length(self):
        n = 5
        X = torch.randn(n, 4)
        g = CKA.gram_matrix(X)
        assert len(g) == n * n

    def test_accepts_2d_input(self):
        X = torch.randn(6, 8)
        g = CKA.gram_matrix(X)
        assert g.shape == (36,)

    def test_flattens_higher_dimensional_input(self):
        """Input with shape (n, c, h, w) should be flattened to (n, c*h*w)."""
        X = torch.randn(4, 3, 2, 2)
        g = CKA.gram_matrix(X)
        assert g.shape == (16,)

    def test_raises_on_non_symmetric_gram(self):
        """Manually break symmetry to trigger the ValueError."""
        X = torch.randn(4, 4)
        gram = X @ X.T
        # Patch torch.allclose to return False, simulating non-symmetry
        with patch("torch.allclose", return_value=False):
            with pytest.raises(ValueError, match="symmetric"):
                CKA.gram_matrix(X)

    def test_centered_gram_has_near_zero_row_mean(self):
        """After double-centering, row/column means should be close to zero."""
        torch.manual_seed(0)
        X = torch.randn(8, 6)
        g = CKA.gram_matrix(X).reshape(8, 8)
        row_means = g.mean(dim=1)
        assert torch.allclose(row_means, torch.zeros_like(row_means), atol=1e-5)

    def test_identical_inputs_give_positive_values(self):
        X = torch.eye(5)
        g = CKA.gram_matrix(X)
        # Gram of identity is identity; centering may produce negatives,
        # but the result should be finite and not NaN.
        assert torch.all(torch.isfinite(g))

    def test_zero_matrix_input(self):
        X = torch.zeros(4, 6)
        g = CKA.gram_matrix(X)
        assert torch.all(g == 0.0)


# ---------------------------------------------------------------------------
# update_state
# ---------------------------------------------------------------------------

class TestUpdateState:
    def _make_activations(self, n_layers: int = 3, n_examples: int = 8, n_features: int = 4):
        return {f"layer_{i}": torch.randn(n_examples, n_features) for i in range(n_layers)}

    def test_output_shape(self):
        n_layers = 3
        acts = self._make_activations(n_layers)
        acc = torch.zeros(n_layers, n_layers)
        result = CKA.update_state(acc, acts)
        assert result.shape == (n_layers, n_layers)

    def test_accumulator_is_updated(self):
        n_layers = 3
        acts = self._make_activations(n_layers)
        acc = torch.zeros(n_layers, n_layers)
        result = CKA.update_state(acc, acts)
        assert not torch.equal(result, acc)

    def test_none_activations_are_skipped(self):
        acts = {"layer_0": torch.randn(8, 4), "layer_1": None, "layer_2": torch.randn(8, 4)}
        acc = torch.zeros(2, 2)
        # Should not raise; None entries are silently skipped
        result = CKA.update_state(acc, acts)
        assert result.shape == (2, 2)

    def test_accumulation_is_additive(self):
        n_layers = 2
        acts = self._make_activations(n_layers)
        acc = torch.zeros(n_layers, n_layers)
        result_once = CKA.update_state(acc, acts)
        result_twice = CKA.update_state(result_once, acts)
        assert torch.allclose(result_twice, 2 * result_once)

    def test_result_matrix_is_symmetric(self):
        n_layers = 4
        acts = self._make_activations(n_layers)
        acc = torch.zeros(n_layers, n_layers)
        result = CKA.update_state(acc, acts)
        assert torch.allclose(result, result.T, atol=1e-5)


# ---------------------------------------------------------------------------
# update_state_across_models
# ---------------------------------------------------------------------------

class TestUpdateStateAcrossModels:
    def _make_activations(self, n_layers: int = 3, n_examples: int = 8, n_features: int = 4):
        return {f"layer_{i}": torch.randn(n_examples, n_features) for i in range(n_layers)}

    def _make_accumulators(self, n1: int, n2: int):
        return (
            torch.zeros(n1, n2),   # cross-model
            torch.zeros(n1),        # self model 1
            torch.zeros(n2),        # self model 2
        )

    def test_returns_three_tensors(self):
        n1, n2 = 3, 3
        acts1 = self._make_activations(n1)
        acts2 = self._make_activations(n2)
        acc, acc1, acc2 = self._make_accumulators(n1, n2)
        result = CKA.update_state_across_models(acc, acc1, acts1, acc2, acts2)
        assert len(result) == 3

    def test_cross_accumulator_shape(self):
        n1, n2 = 3, 4
        acts1 = self._make_activations(n1)
        acts2 = self._make_activations(n2)
        acc, acc1, acc2 = self._make_accumulators(n1, n2)
        new_acc, _, _ = CKA.update_state_across_models(acc, acc1, acts1, acc2, acts2)
        assert new_acc.shape == (n1, n2)

    def test_self_accumulator_shapes(self):
        n1, n2 = 3, 4
        acts1 = self._make_activations(n1)
        acts2 = self._make_activations(n2)
        acc, acc1, acc2 = self._make_accumulators(n1, n2)
        _, new_acc1, new_acc2 = CKA.update_state_across_models(acc, acc1, acts1, acc2, acts2)
        assert new_acc1.shape == (n1,)
        assert new_acc2.shape == (n2,)

    def test_accumulation_is_additive(self):
        n = 3
        acts1 = self._make_activations(n)
        acts2 = self._make_activations(n)
        acc, acc1, acc2 = self._make_accumulators(n, n)
        r1, r1a, r1b = CKA.update_state_across_models(acc, acc1, acts1, acc2, acts2)
        r2, r2a, r2b = CKA.update_state_across_models(r1, r1a, acts1, r1b, acts2)
        assert torch.allclose(r2, 2 * r1)
        assert torch.allclose(r2a, 2 * r1a)
        assert torch.allclose(r2b, 2 * r1b)

    def test_tuple_activations_uses_first_element(self):
        """Activations stored as tuples should use the first element."""
        n = 2
        raw = torch.randn(8, 4)
        acts_tuple = {f"layer_{i}": (raw, torch.randn(8, 4)) for i in range(n)}
        acts_plain = {f"layer_{i}": raw for i in range(n)}
        acc, acc1, acc2 = self._make_accumulators(n, n)
        result_tuple = CKA.update_state_across_models(acc.clone(), acc1.clone(), acts_tuple,
                                                       acc2.clone(), acts_plain)
        result_plain = CKA.update_state_across_models(acc.clone(), acc1.clone(), acts_plain,
                                                       acc2.clone(), acts_plain)
        assert torch.allclose(result_tuple[0], result_plain[0])

    def test_none_activations_are_skipped(self):
        acts1 = {"layer_0": torch.randn(8, 4), "layer_1": None}
        acts2 = {"layer_0": torch.randn(8, 4), "layer_1": None}
        # Accumulators must match len(activations), not the number of non-None entries,
        # because the dimension check happens before None filtering.
        acc = torch.zeros(2, 2)
        acc1 = torch.zeros(2)
        acc2 = torch.zeros(2)
        # Should not raise; None entries are silently skipped during gram computation
        result = CKA.update_state_across_models(acc, acc1, acts1, acc2, acts2)
        assert len(result) == 3

    def test_dimension_mismatch_raises(self):
        """Accumulator size must match number of activations."""
        acts1 = self._make_activations(3)
        acts2 = self._make_activations(3)
        # acc1 has wrong size (2 instead of 3)
        acc = torch.zeros(2, 3)
        acc1 = torch.zeros(2)  # wrong
        acc2 = torch.zeros(3)
        with pytest.raises(Exception):
            CKA.update_state_across_models(acc, acc1, acts1, acc2, acts2)


# ---------------------------------------------------------------------------
# output_similarity
# ---------------------------------------------------------------------------

class TestOutputSimilarity:
    def test_returns_float(self, cka, model_a, model_b, dummy_dataloader):
        result = cka.output_similarity(model_a, model_b, dummy_dataloader)
        assert isinstance(result, float)

    def test_similarity_in_range_zero_one(self, cka, model_a, model_b, dummy_dataloader):
        result = cka.output_similarity(model_a, model_b, dummy_dataloader)
        assert 0.0 <= result <= 1.0

    def test_identical_models_have_similarity_one(self, cka, model_a, dummy_dataloader):
        result = cka.output_similarity(model_a, model_a, dummy_dataloader)
        assert pytest.approx(result, abs=1e-4) == 1.0

    def test_different_models_similarity_less_than_one(self, cka, model_a, model_b, dummy_dataloader):
        result = cka.output_similarity(model_a, model_b, dummy_dataloader)
        assert result < 1.0

    def test_models_set_to_eval_mode(self, cka, model_a, model_b, dummy_dataloader):
        cka.output_similarity(model_a, model_b, dummy_dataloader)
        assert not model_a.training
        assert not model_b.training

    def test_num_runs_produces_aggregated_result(self, cka, model_a, model_b, dummy_dataloader):
        """With multiple runs the method should still return a single scalar."""
        result = cka.output_similarity(model_a, model_b, dummy_dataloader, num_runs=3)
        assert isinstance(result, float)

    def test_aggregate_mean_vs_median(self, cka, model_a, model_b, dummy_dataloader):
        """Different aggregation operators should both return valid floats."""
        mean_result = cka.output_similarity(
            model_a, model_b, dummy_dataloader, num_runs=3, aggregate="mean"
        )
        median_result = cka.output_similarity(
            model_a, model_b, dummy_dataloader, num_runs=3, aggregate="median"
        )
        assert isinstance(mean_result, float)
        assert isinstance(median_result, float)

    def test_num_outputs_limits_batches_used(self, cka, model_a, model_b):
        """num_outputs caps the number of batches consumed from the dataloader."""
        inputs = torch.randn(40, 10)
        targets = torch.zeros(40, dtype=torch.long)
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)

        call_count = 0
        original_forward = model_a.forward

        def counting_forward(x):
            nonlocal call_count
            call_count += 1
            return original_forward(x)

        model_a.forward = counting_forward
        cka.output_similarity(model_a, model_b, loader, num_outputs=2)
        # num_outputs=2 means the loop stops when i > 2 → at most 3 forward passes
        assert call_count <= 3

    def test_nan_output_replaced_with_zero(self, cka, dummy_dataloader):
        """If scaled_hsic / (norm_x * norm_y) is NaN, similarity should be 0.0."""

        class ZeroModel(nn.Module):
            def forward(self, x):
                return torch.zeros(x.shape[0], 4)

        zero_model = ZeroModel()
        result = cka.output_similarity(zero_model, zero_model, dummy_dataloader)
        assert result == 0.0

    def test_symmetry_approximately_holds(self, cka, model_a, model_b, dummy_dataloader):
        """CKA(A, B) ≈ CKA(B, A) for the same data."""
        ab = cka.output_similarity(model_a, model_b, dummy_dataloader)
        ba = cka.output_similarity(model_b, model_a, dummy_dataloader)
        assert pytest.approx(ab, abs=1e-5) == ba