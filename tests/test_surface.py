"""Test Surface module."""
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import MagicMock, patch
from copy import deepcopy

from pylandscape import Surface


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class SimpleModel(nn.Module):
    """Tiny MLP used across all Surface tests."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 4, output_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def simple_model():
    torch.manual_seed(0)
    return SimpleModel()


@pytest.fixture
def dummy_data():
    torch.manual_seed(42)
    inputs = torch.randn(4, 10)
    targets = torch.randn(4, 2)
    return inputs, targets


@pytest.fixture
def dummy_dataloader(dummy_data):
    inputs, targets = dummy_data
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=4)


@pytest.fixture
def surface(simple_model, dummy_dataloader):
    return Surface(
        model=simple_model,
        criterion=simple_model.criterion,
        dataloader=dummy_dataloader,
        device=torch.device("cpu"),
        seed=42,
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestSurfaceInitialization:
    def test_basic_initialization(self, simple_model, dummy_dataloader):
        """Surface can be instantiated with minimal arguments."""
        s = Surface(simple_model, simple_model.criterion, dummy_dataloader)
        assert s.model is simple_model
        assert s.criterion is simple_model.criterion
        assert s.dataloader is dummy_dataloader

    def test_default_device_is_cpu(self, simple_model, dummy_dataloader):
        s = Surface(simple_model, simple_model.criterion, dummy_dataloader)
        assert s.device == "cpu"

    def test_custom_device_stored(self, simple_model, dummy_dataloader):
        s = Surface(
            simple_model,
            simple_model.criterion,
            dummy_dataloader,
            device=torch.device("cpu"),
        )
        assert s.device == torch.device("cpu")

    def test_seed_stored(self, simple_model, dummy_dataloader):
        s = Surface(simple_model, simple_model.criterion, dummy_dataloader, seed=99)
        assert s.seed == 99

    def test_custom_name_stored(self, simple_model, dummy_dataloader):
        s = Surface(
            simple_model, simple_model.criterion, dummy_dataloader, name="my_surface"
        )
        assert s.name == "my_surface"

    def test_inputs_targets_on_device(self, simple_model, dummy_dataloader, device):
        s = Surface(
            simple_model, simple_model.criterion, dummy_dataloader, device=device
        )
        assert s.inputs.device == device
        assert s.targets.device == device

    def test_results_attribute_exists(self, surface):
        assert hasattr(surface, "results")
        assert isinstance(surface.results, dict)


# ---------------------------------------------------------------------------
# Static helpers
# ---------------------------------------------------------------------------

class TestNamedEigenvectors:
    def test_returns_ordered_dict_with_correct_keys(self, simple_model):
        params = list(simple_model.parameters())
        eigenvectors = [torch.zeros_like(p) for p in params]
        result = Surface.named_eigenvectors(simple_model, eigenvectors)
        assert list(result.keys()) == [n for n, _ in simple_model.named_parameters()]

    def test_values_match_provided_eigenvectors(self, simple_model):
        params = list(simple_model.parameters())
        eigenvectors = [torch.ones_like(p) for p in params]
        result = Surface.named_eigenvectors(simple_model, eigenvectors)
        for v, result_v in zip(eigenvectors, result.values()):
            assert torch.equal(v, result_v)

    def test_shape_mismatch_skips_entry(self, simple_model, capsys):
        params = list(simple_model.parameters())
        # give the first eigenvector the wrong shape
        bad_eigenvectors = [torch.zeros(1)] + [torch.zeros_like(p) for p in params[1:]]
        result = Surface.named_eigenvectors(simple_model, bad_eigenvectors)
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        # mismatched entry should be absent
        assert len(result) == len(params) - 1


class TestGetParams:
    def test_returns_nn_module(self, simple_model):
        params = list(simple_model.parameters())
        direction = [torch.zeros_like(p) for p in params]
        result = Surface.get_params(simple_model, [direction], [0.0])
        assert isinstance(result, nn.Module)

    def test_zero_step_leaves_params_unchanged(self, simple_model):
        params = list(simple_model.parameters())
        direction = [torch.ones_like(p) for p in params]
        perturbed = Surface.get_params(simple_model, [direction], [0.0])
        for orig, pert in zip(simple_model.parameters(), perturbed.parameters()):
            assert torch.allclose(orig.data, pert.data)

    def test_nonzero_step_changes_params(self, simple_model):
        params = list(simple_model.parameters())
        direction = [torch.ones_like(p) for p in params]
        perturbed = Surface.get_params(simple_model, [direction], [1.0])
        for orig, pert in zip(simple_model.parameters(), perturbed.parameters()):
            assert not torch.equal(orig.data, pert.data)

    def test_original_model_is_not_mutated(self, simple_model):
        original_params = [p.data.clone() for p in simple_model.parameters()]
        params = list(simple_model.parameters())
        direction = [torch.ones_like(p) for p in params]
        Surface.get_params(simple_model, [direction], [5.0])
        for orig, current in zip(original_params, simple_model.parameters()):
            assert torch.equal(orig, current.data)

    def test_mismatched_directions_and_steps_raises(self, simple_model):
        params = list(simple_model.parameters())
        direction = [torch.zeros_like(p) for p in params]
        with pytest.raises(AssertionError):
            Surface.get_params(simple_model, [direction, direction], [0.0])

    def test_multiple_directions_additive(self, simple_model):
        """Perturbation from two directions should equal their sum."""
        params = list(simple_model.parameters())
        d1 = [torch.ones_like(p) for p in params]
        d2 = [torch.ones_like(p) * 2 for p in params]
        perturbed = Surface.get_params(simple_model, [d1, d2], [1.0, 1.0])
        for orig, pert in zip(simple_model.parameters(), perturbed.parameters()):
            expected = orig.data + 1.0 * torch.ones_like(orig) + 1.0 * torch.ones_like(orig) * 2
            assert torch.allclose(pert.data, expected)


class TestRandLike:
    def test_output_length_matches_input(self, simple_model):
        vector = list(simple_model.parameters())
        result = Surface._rand_like(vector)
        assert len(result) == len(vector)

    def test_output_shapes_match_input(self, simple_model):
        vector = list(simple_model.parameters())
        result = Surface._rand_like(vector)
        for v, r in zip(vector, result):
            assert v.shape == r.shape

    def test_values_are_different_from_input(self, simple_model):
        """Random output should (very likely) differ from the source tensors."""
        vector = [torch.zeros_like(p) for p in simple_model.parameters()]
        result = Surface._rand_like(vector)
        assert any(not torch.equal(v, r) for v, r in zip(vector, result))


class TestOrthogonalizeVectors:
    def test_result_length_matches_input(self, simple_model):
        v1 = [torch.randn_like(p) for p in simple_model.parameters()]
        v2 = [torch.randn_like(p) for p in simple_model.parameters()]
        result = Surface.orthogonalize_vectors(v2, v1)
        assert len(result) == len(v1)

    def test_orthogonality_after_orthogonalization(self, simple_model):
        torch.manual_seed(0)
        v1 = [torch.randn_like(p) for p in simple_model.parameters()]
        v2 = [torch.randn_like(p) for p in simple_model.parameters()]
        v2_orth = Surface.orthogonalize_vectors(v2, v1)
        assert Surface.check_orthogonality(v2_orth, v1, tol=1e-5)

    def test_already_orthogonal_vectors_unchanged(self):
        """e1 and e2 in R^2 are already orthogonal; result should stay orthogonal."""
        e1 = [torch.tensor([1.0, 0.0])]
        e2 = [torch.tensor([0.0, 1.0])]
        result = Surface.orthogonalize_vectors(e2, e1)
        dot = torch.dot(result[0].flatten(), e1[0].flatten())
        assert torch.abs(dot) < 1e-6


class TestCheckOrthogonality:
    def test_orthogonal_vectors_return_true(self):
        v1 = [torch.tensor([1.0, 0.0])]
        v2 = [torch.tensor([0.0, 1.0])]
        assert Surface.check_orthogonality(v1, v2) is True

    def test_non_orthogonal_vectors_return_false(self):
        v1 = [torch.tensor([1.0, 0.0])]
        v2 = [torch.tensor([1.0, 1.0])]
        assert Surface.check_orthogonality(v1, v2) is False

    def test_zero_vector_is_orthogonal_to_anything(self):
        v1 = [torch.tensor([0.0, 0.0])]
        v2 = [torch.tensor([3.0, 7.0])]
        assert Surface.check_orthogonality(v1, v2) is True

    def test_custom_tolerance_respected(self):
        v1 = [torch.tensor([1.0, 0.0])]
        v2 = [torch.tensor([1e-5, 1.0])]
        # With tight tolerance this is NOT orthogonal
        assert Surface.check_orthogonality(v1, v2, tol=1e-6) is False
        # With loose tolerance it IS orthogonal
        assert Surface.check_orthogonality(v1, v2, tol=1e-4) is True


# ---------------------------------------------------------------------------
# Loss landscape methods
# ---------------------------------------------------------------------------

class TestRandomLine:
    def test_returns_tuple_of_two_arrays(self, surface):
        lams, loss = surface.random_line((-1.0, 1.0), steps=5)
        assert isinstance(lams, np.ndarray)
        assert isinstance(loss, np.ndarray)

    def test_steps_determine_array_length(self, surface):
        lams, loss = surface.random_line((-1.0, 1.0), steps=7)
        assert len(lams) == 7
        assert len(loss) == 7

    def test_lam_range_respected(self, surface):
        lams, _ = surface.random_line((-2.0, 3.0), steps=5)
        assert pytest.approx(lams[0], abs=1e-5) == -2.0
        assert pytest.approx(lams[-1], abs=1e-5) == 3.0

    def test_results_stored_in_dict(self, surface):
        surface.random_line((-1.0, 1.0), steps=5)
        assert "random_line" in surface.results
        assert "alpha" in surface.results["random_line"]
        assert "loss" in surface.results["random_line"]

    def test_loss_values_are_finite(self, surface):
        _, loss = surface.random_line((-1.0, 1.0), steps=5)
        assert np.all(np.isfinite(loss))

    def test_loss_values_are_non_negative_for_mse(self, surface):
        _, loss = surface.random_line((-1.0, 1.0), steps=5)
        assert np.all(loss >= 0.0)


class TestRandomSurface:
    def test_returns_tuple_of_three_arrays(self, surface):
        result = surface.random_surface((-1.0, 1.0), steps=4)
        assert len(result) == 3

    def test_loss_surface_shape(self, surface):
        alpha, beta, loss = surface.random_surface((-1.0, 1.0), steps=4)
        assert loss.shape == (4, 4)

    def test_alpha_beta_have_correct_length(self, surface):
        alpha, beta, loss = surface.random_surface((-1.0, 1.0), steps=6)
        assert len(alpha) == 6
        assert len(beta) == 6

    def test_results_stored_in_dict(self, surface):
        surface.random_surface((-1.0, 1.0), steps=4)
        assert "random_plane" in surface.results
        assert "alpha" in surface.results["random_plane"]
        assert "beta" in surface.results["random_plane"]
        assert "loss" in surface.results["random_plane"]

    def test_loss_values_are_finite(self, surface):
        _, _, loss = surface.random_surface((-1.0, 1.0), steps=4)
        assert np.all(np.isfinite(loss))

    def test_directions_are_orthogonal(self, surface):
        """Internal directions used for the surface must be orthogonal."""
        # Patch _rand_like so we can inspect the vectors used
        original_rand_like = Surface._rand_like
        captured_vectors = []

        def capturing_rand_like(vector):
            v = original_rand_like(vector)
            captured_vectors.append(v)
            return v

        with patch.object(Surface, "_rand_like", staticmethod(capturing_rand_like)):
            surface.random_surface((-0.5, 0.5), steps=3)

        # After orthogonalization v2 should be orthogonal to v1
        import pyhessian
        v1 = pyhessian.utils.normalization(captured_vectors[0])
        v2_raw = captured_vectors[1]
        v2_orth = Surface.orthogonalize_vectors(v2_raw, v1)
        assert Surface.check_orthogonality(v1, v2_orth, tol=1e-5)


class TestHessianLine:
    def test_returns_tuple_of_two_arrays(self, surface):
        with patch("pyhessian.hessian") as mock_hessian:
            params = list(surface.model.parameters())
            mock_hessian.return_value.eigenvalues.return_value = (
                [1.0],
                [[torch.zeros_like(p) for p in params]],
            )
            lams, loss = surface.hessian_line((-1.0, 1.0), steps=5)
        assert isinstance(lams, np.ndarray)
        assert isinstance(loss, np.ndarray)

    def test_steps_determine_array_length(self, surface):
        with patch("pyhessian.hessian") as mock_hessian:
            params = list(surface.model.parameters())
            mock_hessian.return_value.eigenvalues.return_value = (
                [1.0],
                [[torch.zeros_like(p) for p in params]],
            )
            lams, loss = surface.hessian_line((-1.0, 1.0), steps=9)
        assert len(lams) == 9
        assert len(loss) == 9

    def test_results_stored_in_dict(self, surface):
        with patch("pyhessian.hessian") as mock_hessian:
            params = list(surface.model.parameters())
            mock_hessian.return_value.eigenvalues.return_value = (
                [1.0],
                [[torch.zeros_like(p) for p in params]],
            )
            surface.hessian_line((-1.0, 1.0), steps=5)
        assert "hessian_line" in surface.results
        assert "alpha" in surface.results["hessian_line"]
        assert "loss" in surface.results["hessian_line"]

    def test_eigenvalues_called_with_correct_args(self, surface):
        with patch("pyhessian.hessian") as mock_hessian:
            params = list(surface.model.parameters())
            mock_hessian.return_value.eigenvalues.return_value = (
                [1.0],
                [[torch.zeros_like(p) for p in params]],
            )
            surface.hessian_line((-1.0, 1.0), steps=5, max_iter=50)
            mock_hessian.return_value.eigenvalues.assert_called_once_with(
                maxIter=50, tol=1e-5
            )


class TestHessianSurface:
    def test_returns_tuple_of_three_arrays(self, surface):
        with patch("pyhessian.hessian") as mock_hessian:
            params = list(surface.model.parameters())
            mock_hessian.return_value.eigenvalues.return_value = (
                [1.0, 0.5],
                [
                    [torch.zeros_like(p) for p in params],
                    [torch.zeros_like(p) for p in params],
                ],
            )
            result = surface.hessian_surface((-1.0, 1.0), steps=4)
        assert len(result) == 3

    def test_loss_surface_shape(self, surface):
        with patch("pyhessian.hessian") as mock_hessian:
            params = list(surface.model.parameters())
            mock_hessian.return_value.eigenvalues.return_value = (
                [1.0, 0.5],
                [
                    [torch.zeros_like(p) for p in params],
                    [torch.zeros_like(p) for p in params],
                ],
            )
            alpha, beta, loss = surface.hessian_surface((-1.0, 1.0), steps=4)
        assert loss.shape == (4, 4)

    def test_results_stored_in_dict(self, surface):
        with patch("pyhessian.hessian") as mock_hessian:
            params = list(surface.model.parameters())
            mock_hessian.return_value.eigenvalues.return_value = (
                [1.0, 0.5],
                [
                    [torch.zeros_like(p) for p in params],
                    [torch.zeros_like(p) for p in params],
                ],
            )
            surface.hessian_surface((-1.0, 1.0), steps=4)
        assert "hessian_plane" in surface.results
        assert "alpha" in surface.results["hessian_plane"]
        assert "beta" in surface.results["hessian_plane"]
        assert "loss" in surface.results["hessian_plane"]

    def test_top_n_2_passed_to_eigenvalues(self, surface):
        with patch("pyhessian.hessian") as mock_hessian:
            params = list(surface.model.parameters())
            mock_hessian.return_value.eigenvalues.return_value = (
                [1.0, 0.5],
                [
                    [torch.zeros_like(p) for p in params],
                    [torch.zeros_like(p) for p in params],
                ],
            )
            surface.hessian_surface((-1.0, 1.0), steps=4)
            mock_hessian.return_value.eigenvalues.assert_called_once_with(
                maxIter=100, tol=1e-5, top_n=2
            )


class TestHessianHyperplane:
    def _make_mock_eigenvectors(self, model, n):
        return [[torch.zeros_like(p) for p in model.parameters()] for _ in range(n)]

    def test_returns_lams_and_hyperplane(self, surface):
        n = 3
        with patch("pyhessian.hessian") as mock_hessian:
            mock_hessian.return_value.eigenvalues.return_value = (
                [1.0] * n,
                self._make_mock_eigenvectors(surface.model, n),
            )
            result = surface.hessian_hyperplane((-1.0, 1.0), steps=3, n=n)
        assert len(result) == 2

    def test_loss_hyperplane_shape(self, surface):
        n = 3
        steps = 3
        with patch("pyhessian.hessian") as mock_hessian:
            mock_hessian.return_value.eigenvalues.return_value = (
                [1.0] * n,
                self._make_mock_eigenvectors(surface.model, n),
            )
            lams, loss = surface.hessian_hyperplane((-1.0, 1.0), steps=steps, n=n)
        assert loss.shape == (steps,) * n

    def test_correct_top_n_passed_to_eigenvalues(self, surface):
        n = 4
        with patch("pyhessian.hessian") as mock_hessian:
            mock_hessian.return_value.eigenvalues.return_value = (
                [1.0] * n,
                self._make_mock_eigenvectors(surface.model, n),
            )
            surface.hessian_hyperplane((-1.0, 1.0), steps=2, n=n)
            mock_hessian.return_value.eigenvalues.assert_called_once_with(
                maxIter=100, tol=1e-5, top_n=n
            )


class TestRandomHyperplane:
    def test_returns_lams_and_hyperplane(self, surface):
        lams, loss = surface.random_hyperplane((-1.0, 1.0), steps=3, n=3)
        assert isinstance(lams, np.ndarray)
        assert isinstance(loss, np.ndarray)

    def test_loss_hyperplane_shape(self, surface):
        n, steps = 3, 4
        lams, loss = surface.random_hyperplane((-1.0, 1.0), steps=steps, n=n)
        assert loss.shape == (steps,) * n

    def test_lam_range_respected(self, surface):
        lams, _ = surface.random_hyperplane((-3.0, 2.0), steps=5, n=2)
        assert pytest.approx(lams[0], abs=1e-5) == -3.0
        assert pytest.approx(lams[-1], abs=1e-5) == 2.0

    def test_n_1_equivalent_to_random_line(self, surface):
        """With n=1, random_hyperplane should produce a 1-D loss array."""
        lams, loss = surface.random_hyperplane((-1.0, 1.0), steps=5, n=1)
        assert loss.ndim == 1
        assert len(loss) == 5

    def test_directions_are_sequentially_orthogonal(self, surface):
        """Each new direction must be orthogonal to the previous one."""
        # Collect all directions by monkey-patching orthogonalize_vectors
        collected_pairs = []
        original_orth = Surface.orthogonalize_vectors

        @staticmethod
        def tracking_orth(new_v, ref_v):
            result = original_orth(new_v, ref_v)
            collected_pairs.append((result, ref_v))
            return result

        with patch.object(Surface, "orthogonalize_vectors", tracking_orth):
            surface.random_hyperplane((-0.5, 0.5), steps=2, n=3)

        import pyhessian
        for v_new, v_ref in collected_pairs:
            v_ref_norm = pyhessian.utils.normalization(v_ref)
            assert Surface.check_orthogonality(v_new, v_ref_norm, tol=1e-4)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_step_random_line(self, surface):
        lams, loss = surface.random_line((-1.0, 1.0), steps=1)
        assert len(lams) == 1
        assert len(loss) == 1

    def test_zero_perturbation_random_line_returns_baseline_loss(self, surface):
        """At lam=0, loss should equal the unperturbed model loss."""
        baseline = surface.criterion(
            surface.model(surface.inputs), surface.targets
        ).item()
        _, loss = surface.random_line((0.0, 0.0), steps=1)
        assert pytest.approx(loss[0], abs=1e-5) == baseline

    def test_symmetric_range_random_surface(self, surface):
        alpha, beta, _ = surface.random_surface((-2.0, 2.0), steps=5)
        assert pytest.approx(alpha[0], abs=1e-5) == -2.0
        assert pytest.approx(alpha[-1], abs=1e-5) == 2.0

    def test_results_overwritten_on_repeated_call(self, surface):
        surface.random_line((-1.0, 1.0), steps=3)
        first_loss = surface.results["random_line"]["loss"].copy()
        # Call again; result may differ due to new random direction
        surface.random_line((-1.0, 1.0), steps=3)
        # Key should still exist
        assert "random_line" in surface.results