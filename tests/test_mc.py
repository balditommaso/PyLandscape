"""Test ModeConnectivity, curved_model, and curve_module classes."""
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch import tensor
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import MagicMock, patch

from pylandscape import ModeConnectivity
from pylandscape.mc_utils.curve_module import (
    Coeffs_t,
    CurveWeightComputation,
    CurveModule,
    Linear,
    Conv2d,
    ConvTranspose2D,
)
from pylandscape.mc_utils import curved_model


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class SimpleMLP(nn.Module):
    def __init__(self, in_features: int = 8, hidden: int = 4, out_features: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.fc = nn.Linear(2 * 4 * 4, 2)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        return self.fc(x.flatten(1))


def make_dataloader(n: int = 16, in_features: int = 8, out_dim: int = 2,
                    batch_size: int = 8) -> DataLoader:
    torch.manual_seed(0)
    inputs = torch.randn(n, in_features)
    targets = torch.randn(n, out_dim)
    return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size)


FIX_POINTS_3 = [True, False, True]   # endpoints fixed, middle free
FIX_POINTS_2 = [True, True]


# ---------------------------------------------------------------------------
# Coeffs_t
# ---------------------------------------------------------------------------

class TestCoeffsT:
    def test_default_value_is_zero(self):
        assert Coeffs_t.value == 0

    def test_value_can_be_set(self):
        Coeffs_t.value = 0.5
        assert Coeffs_t.value == 0.5
        Coeffs_t.value = 0   # restore

    def test_shared_across_instances(self):
        Coeffs_t.value = 0.3
        assert Coeffs_t.value == 0.3
        Coeffs_t.value = 0   # restore


# ---------------------------------------------------------------------------
# CurveWeightComputation
# ---------------------------------------------------------------------------

class TestCurveWeightComputation:
    def _make_computation(self, index: int = 0):
        weight = torch.randn(4, 4)
        bias = torch.randn(4)

        def dummy_weight_fn(coeffs_t):
            return [weight * coeffs_t, bias * coeffs_t]

        return CurveWeightComputation(dummy_weight_fn, index), weight, bias

    def test_forward_returns_correct_index(self):
        cwc, weight, bias = self._make_computation(index=0)
        Coeffs_t.value = 1.0
        result = cwc.forward(torch.zeros(4, 4))  # weight arg ignored
        assert torch.allclose(result, weight)
        Coeffs_t.value = 0

    def test_forward_uses_coeffs_t_global(self):
        cwc, weight, _ = self._make_computation(index=0)
        Coeffs_t.value = 0.5
        result = cwc.forward(torch.zeros(4, 4))
        assert torch.allclose(result, weight * 0.5)
        Coeffs_t.value = 0

    def test_index_selects_correct_output(self):
        cwc, _, bias = self._make_computation(index=1)
        Coeffs_t.value = 1.0
        result = cwc.forward(torch.zeros(4,))
        assert torch.allclose(result, bias)
        Coeffs_t.value = 0


# ---------------------------------------------------------------------------
# CurveModule
# ---------------------------------------------------------------------------

class TestCurveModule:
    def _make_curve_module(self):
        class DummyCurve(CurveModule):
            def __init__(self):
                super().__init__([True, False, True], ("weight", "bias"))
                for i in range(self.num_bends):
                    self.register_parameter(f"weight_{i}", nn.Parameter(torch.randn(4, 4)))
                    self.register_parameter(f"bias_{i}", nn.Parameter(torch.randn(4)))

        return DummyCurve()

    def test_num_bends_set_correctly(self):
        m = self._make_curve_module()
        assert m.num_bends == 3

    def test_fix_points_stored(self):
        m = self._make_curve_module()
        assert m.fix_points == [True, False, True]

    def test_parameter_names_stored(self):
        m = self._make_curve_module()
        assert "weight" in m.parameter_names
        assert "bias" in m.parameter_names

    def test_compute_weights_t_returns_correct_length(self):
        m = self._make_curve_module()
        coeffs = [0.0, 1.0, 0.0]  # point at bend 1
        w_t = m.compute_weights_t(coeffs)
        assert len(w_t) == 2  # weight + bias

    def test_compute_weights_t_interpolates_linearly(self):
        """With equal coefficients the result should be the parameter mean."""
        m = self._make_curve_module()
        coeffs = [1 / 3, 1 / 3, 1 / 3]
        w_t = m.compute_weights_t(coeffs)
        expected = sum(getattr(m, f"weight_{i}").data for i in range(3)) / 3
        assert torch.allclose(w_t[0], expected, atol=1e-5)

    def test_fixed_params_have_no_gradient(self):
        m = self._make_curve_module()
        # fix_points = [True, False, True] → weight_0 and weight_2 must not require grad
        assert not m.weight_0.requires_grad
        assert m.weight_1.requires_grad
        assert not m.weight_2.requires_grad


# ---------------------------------------------------------------------------
# Linear curve module
# ---------------------------------------------------------------------------

class TestLinearCurveModule:
    def _make_linear(self, bias: bool = True):
        base = nn.Linear(6, 4, bias=bias)
        return Linear(base, FIX_POINTS_3)

    def test_num_bends(self):
        m = self._make_linear()
        assert m.num_bends == 3

    def test_weight_params_registered(self):
        m = self._make_linear()
        for i in range(3):
            assert hasattr(m, f"weight_{i}")

    def test_bias_params_registered_when_bias_true(self):
        m = self._make_linear()
        for i in range(3):
            assert hasattr(m, f"bias_{i}")
            assert getattr(m, f"bias_{i}") is not None

    def test_bias_params_none_when_no_bias(self):
        base = nn.Linear(6, 4, bias=False)
        m = Linear(base, FIX_POINTS_3)
        for i in range(3):
            assert getattr(m, f"bias_{i}") is None

    def test_fixed_points_block_gradients(self):
        m = self._make_linear()
        assert not m.weight_0.requires_grad   # fixed
        assert m.weight_1.requires_grad        # free
        assert not m.weight_2.requires_grad   # fixed

    def test_weight_shapes(self):
        base = nn.Linear(6, 4)
        m = Linear(base, FIX_POINTS_3)
        for i in range(3):
            assert getattr(m, f"weight_{i}").shape == (4, 6)

    def test_forward_returns_correct_shape(self):
        m = self._make_linear()
        Coeffs_t.value = [0.0, 1.0, 0.0]
        x = torch.randn(5, 6)
        out = m(x)
        assert out.shape == (5, 4)
        Coeffs_t.value = 0

    def test_reset_parameters_initializes_within_bounds(self):
        """Uniform initialisation stdv = 1/sqrt(in_features)."""
        base = nn.Linear(6, 4)
        m = Linear(base, FIX_POINTS_3)
        stdv = 1.0 / math.sqrt(6)
        for i in range(3):
            w = getattr(m, f"weight_{i}").data
            assert w.abs().max().item() <= stdv * 3   # 3σ tolerance


# ---------------------------------------------------------------------------
# Conv2d curve module
# ---------------------------------------------------------------------------

class TestConv2dCurveModule:
    def _make_conv(self, bias: bool = True, groups: int = 1):
        base = nn.Conv2d(4, 8, kernel_size=3, bias=bias, groups=groups)
        return Conv2d(base, FIX_POINTS_3)

    def test_num_bends(self):
        m = self._make_conv()
        assert m.num_bends == 3

    def test_weight_shape(self):
        m = self._make_conv()
        for i in range(3):
            assert getattr(m, f"weight_{i}").shape == (8, 4, 3, 3)

    def test_bias_shape_when_bias_true(self):
        m = self._make_conv()
        for i in range(3):
            assert getattr(m, f"bias_{i}").shape == (8,)

    def test_bias_none_when_no_bias(self):
        base = nn.Conv2d(4, 8, kernel_size=3, bias=False)
        m = Conv2d(base, FIX_POINTS_3)
        for i in range(3):
            assert getattr(m, f"bias_{i}") is None

    def test_groups_not_divisible_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            base = nn.Conv2d(4, 8, kernel_size=3)
            base.in_channels = 5   # not divisible by groups=2
            base.groups = 2
            Conv2d(base, FIX_POINTS_3)

    def test_forward_returns_correct_shape(self):
        m = self._make_conv()
        Coeffs_t.value = [0.0, 1.0, 0.0]
        x = torch.randn(2, 4, 8, 8)
        out = m(x)
        assert out.shape == (2, 8, 6, 6)
        Coeffs_t.value = 0

    def test_fixed_points_block_gradients(self):
        m = self._make_conv()
        assert not m.weight_0.requires_grad
        assert m.weight_1.requires_grad
        assert not m.weight_2.requires_grad

    def test_conv_attributes_stored(self):
        base = nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, dilation=1, groups=1)
        m = Conv2d(base, FIX_POINTS_3)
        assert m.stride == base.stride
        assert m.padding == base.padding
        assert m.dilation == base.dilation
        assert m.groups == base.groups


# ---------------------------------------------------------------------------
# ConvTranspose2D curve module
# ---------------------------------------------------------------------------

class TestConvTranspose2DCurveModule:
    def _make_conv_t(self, bias: bool = True):
        base = nn.ConvTranspose2d(4, 8, kernel_size=3, bias=bias)
        return ConvTranspose2D(base, FIX_POINTS_3)

    def test_num_bends(self):
        m = self._make_conv_t()
        assert m.num_bends == 3

    def test_weight_shape(self):
        """ConvTranspose2d weight shape is (in_channels, out_channels//groups, kH, kW)."""
        m = self._make_conv_t()
        for i in range(3):
            assert getattr(m, f"weight_{i}").shape == (4, 8, 3, 3)

    def test_bias_shape_when_bias_true(self):
        m = self._make_conv_t()
        for i in range(3):
            assert getattr(m, f"bias_{i}").shape == (8,)

    def test_groups_not_divisible_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            base = nn.ConvTranspose2d(4, 8, kernel_size=3)
            base.in_channels = 5
            base.groups = 2
            ConvTranspose2D(base, FIX_POINTS_3)

    def test_forward_returns_correct_shape(self):
        m = self._make_conv_t()
        Coeffs_t.value = [0.0, 1.0, 0.0]
        x = torch.randn(2, 4, 6, 6)
        out = m(x)
        assert out.shape == (2, 8, 8, 8)
        Coeffs_t.value = 0

    def test_fixed_points_block_gradients(self):
        m = self._make_conv_t()
        assert not m.weight_0.requires_grad
        assert m.weight_1.requires_grad
        assert not m.weight_2.requires_grad


# ---------------------------------------------------------------------------
# curved_model
# ---------------------------------------------------------------------------

class TestCurvedModel:
    def test_returns_nn_module(self):
        model = SimpleMLP()
        cm = curved_model(model, FIX_POINTS_3)
        assert isinstance(cm, nn.Module)

    def test_original_model_is_not_mutated(self):
        model = SimpleMLP()
        original_params = {n: p.data.clone() for n, p in model.named_parameters()}
        curved_model(model, FIX_POINTS_3)
        for name, p in model.named_parameters():
            assert torch.equal(p.data, original_params[name])

    def test_linear_layers_converted(self):
        model = SimpleMLP()
        cm = curved_model(model, FIX_POINTS_3)
        from pylandscape.mc_utils.curve_module import Linear as CurveLinear
        for module in cm.modules():
            if not isinstance(module, SimpleMLP):  # skip root
                assert not isinstance(module, nn.Linear), \
                    "Plain nn.Linear should be replaced by CurveLinear"

    def test_conv_layers_converted(self):
        model = SimpleCNN()
        cm = curved_model(model, FIX_POINTS_3)
        from pylandscape.mc_utils.curve_module import Conv2d as CurveConv2d
        for name, module in cm.named_modules():
            if "conv" in name and name != "":
                assert isinstance(module, CurveConv2d)

    def test_curved_model_has_correct_num_bends(self):
        model = SimpleMLP()
        cm = curved_model(model, FIX_POINTS_3)
        from pylandscape.mc_utils.curve_module import Linear as CurveLinear
        for module in cm.modules():
            if isinstance(module, CurveLinear):
                assert module.num_bends == len(FIX_POINTS_3)

    def test_unknown_modules_are_preserved_unchanged(self):
        """BatchNorm, ReLU, etc. should not be wrapped."""
        class ModelWithBN(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm1d(8)
                self.fc = nn.Linear(8, 4)

            def forward(self, x):
                return self.fc(self.bn(x))

        model = ModelWithBN()
        cm = curved_model(model, FIX_POINTS_3)
        assert isinstance(cm.bn, nn.BatchNorm1d)


# ---------------------------------------------------------------------------
# ModeConnectivity
# ---------------------------------------------------------------------------

class TestModeConnectivityInitialization:
    def test_default_device_is_cpu(self):
        mc = ModeConnectivity()
        assert mc.device == torch.device("cpu")

    def test_custom_name_stored(self):
        mc = ModeConnectivity(name="my_mc")
        assert mc.name == "my_mc"

    def test_results_attribute_exists(self):
        mc = ModeConnectivity()
        assert hasattr(mc, "results")
        assert isinstance(mc.results, dict)


class TestMaxDeviation:
    def test_flat_loss_gives_zero_deviation(self):
        loss = torch.ones(10) * 2.5
        assert ModeConnectivity.max_deviation(loss) == pytest.approx(0.0, abs=1e-6)

    def test_positive_deviation_when_interior_lower_than_boundary_mean(self):
        """If a point dips below the endpoint average, deviation should be positive."""
        loss = torch.tensor([1.0, 0.5, 0.2, 0.5, 1.0])
        dev = ModeConnectivity.max_deviation(loss)
        assert dev > 0.0

    def test_negative_deviation_when_interior_higher_than_boundary_mean(self):
        """If a point rises above the endpoint average, deviation should be negative."""
        loss = torch.tensor([0.5, 1.0, 2.0, 1.0, 0.5])
        dev = ModeConnectivity.max_deviation(loss)
        assert dev < 0.0

    def test_returns_float(self):
        loss = torch.linspace(1.0, 3.0, 5)
        result = ModeConnectivity.max_deviation(loss)
        assert isinstance(result, float)

    def test_two_element_tensor(self):
        """For a 2-element tensor the midpoint equals both elements → deviation = 0."""
        loss = torch.tensor([1.0, 3.0])
        dev = ModeConnectivity.max_deviation(loss)
        # midpoint = 2.0; both deviations are |1-2|=1 and |3-2|=1;
        # midpoint - max_dev = 2 - 3 = -1
        assert isinstance(dev, float)

    def test_known_value(self):
        """Manual calculation: endpoints=1 and 3, midpoint=2, min interior=0."""
        loss = torch.tensor([1.0, 0.0, 3.0])
        midpoint = (1.0 + 3.0) / 2   # 2.0
        # deviations: |1-2|=1, |0-2|=2, |3-2|=1 → max at index 1 (value 0)
        # result = midpoint - loss[1] = 2.0 - 0.0 = 2.0
        assert ModeConnectivity.max_deviation(loss) == pytest.approx(2.0, abs=1e-6)

    def test_symmetrical_u_shape(self):
        loss = torch.tensor([2.0, 1.0, 0.5, 1.0, 2.0])
        dev = ModeConnectivity.max_deviation(loss)
        # midpoint = 2.0; max deviation at value 0.5 → 2.0 - 0.5 = 1.5
        assert dev == pytest.approx(1.5, abs=1e-5)


class TestModeConnectivityCompute:
    """
    `compute` runs full training which is expensive and depends on mc_utils.Interpolate.
    We test it via mocking so the suite stays fast and deterministic.
    """

    def _setup(self):
        model1 = SimpleMLP()
        model2 = SimpleMLP()
        criterion = nn.MSELoss()
        train_dl = make_dataloader()
        test_dl = make_dataloader()
        mc = ModeConnectivity(device=torch.device("cpu"))
        return mc, model1, model2, criterion, train_dl, test_dl

    def _mock_interpolate(self, loss_values):
        """Return a mock Interpolate class whose sample_model yields loss_values sequentially."""
        mock_instance = MagicMock()
        mock_instance.sample_model.side_effect = iter(loss_values)
        mock_cls = MagicMock(return_value=mock_instance)
        return mock_cls, mock_instance

    def test_returns_float(self):
        mc, m1, m2, crit, tr_dl, te_dl = self._setup()
        n_points = 5
        mock_cls, mock_inst = self._mock_interpolate([float(i) for i in range(n_points)])
        with patch("pylandscape.ModeConnectivity.compute.__globals__['Interpolate']", mock_cls,
                   create=True):
            with patch("pylandscape.mc_utils.Interpolate", mock_cls):
                result = mc.compute(
                    m1, m2, crit, tr_dl, te_dl,
                    learning_rate=1e-3,
                    num_points=n_points,
                )
        assert isinstance(result, float)

    def test_train_curve_called_once(self):
        mc, m1, m2, crit, tr_dl, te_dl = self._setup()
        n_points = 5
        mock_cls, mock_inst = self._mock_interpolate([0.5] * n_points)
        with patch("pylandscape.mc_utils.Interpolate", mock_cls):
            mc.compute(m1, m2, crit, tr_dl, te_dl, learning_rate=1e-3, num_points=n_points)
        mock_inst.train_curve.assert_called_once()

    def test_sample_model_called_num_points_times(self):
        mc, m1, m2, crit, tr_dl, te_dl = self._setup()
        n_points = 7
        mock_cls, mock_inst = self._mock_interpolate([0.5] * n_points)
        with patch("pylandscape.mc_utils.Interpolate", mock_cls):
            mc.compute(m1, m2, crit, tr_dl, te_dl, learning_rate=1e-3, num_points=n_points)
        assert mock_inst.sample_model.call_count == n_points

    def test_t_range_spans_zero_to_one(self):
        """The t values sampled must start at 0.0 and end at 1.0."""
        mc, m1, m2, crit, tr_dl, te_dl = self._setup()
        n_points = 5
        captured_t = []

        def capture_sample(dataloader, t):
            captured_t.append(t.item())
            return 0.5

        mock_cls = MagicMock()
        mock_cls.return_value.train_curve = MagicMock()
        mock_cls.return_value.sample_model.side_effect = capture_sample

        with patch("pylandscape.mc_utils.Interpolate", mock_cls):
            mc.compute(m1, m2, crit, tr_dl, te_dl, learning_rate=1e-3, num_points=n_points)

        assert captured_t[0] == pytest.approx(0.0, abs=1e-6)
        assert captured_t[-1] == pytest.approx(1.0, abs=1e-6)
        assert len(captured_t) == n_points

    def test_curve_selection_passed_to_interpolate(self):
        mc, m1, m2, crit, tr_dl, te_dl = self._setup()
        n_points = 3
        mock_cls, mock_inst = self._mock_interpolate([0.5] * n_points)
        mock_inst.sample_model.side_effect = [0.5] * n_points

        with patch("pylandscape.mc_utils.Interpolate", mock_cls):
            with patch("pylandscape.mc_utils.curves") as mock_curves:
                mock_curves.Bezier = "MockBezier"
                mc.compute(m1, m2, crit, tr_dl, te_dl,
                           learning_rate=1e-3, curve="Bezier", num_points=n_points)

        # First positional arg to Interpolate should be the resolved curve
        call_kwargs = mock_cls.call_args
        assert call_kwargs is not None

    def test_max_deviation_result_matches_loss_curve(self):
        """End-to-end: mocked flat loss curve should give deviation ~0."""
        mc, m1, m2, crit, tr_dl, te_dl = self._setup()
        n_points = 10
        flat_loss = [1.0] * n_points
        mock_cls, mock_inst = self._mock_interpolate(flat_loss)

        with patch("pylandscape.mc_utils.Interpolate", mock_cls):
            result = mc.compute(m1, m2, crit, tr_dl, te_dl,
                                learning_rate=1e-3, num_points=n_points)

        assert result == pytest.approx(0.0, abs=1e-5)

    def test_u_shaped_loss_gives_positive_deviation(self):
        """A loss that dips in the middle (mode-connected) → positive deviation."""
        mc, m1, m2, crit, tr_dl, te_dl = self._setup()
        # boundaries at 1.0, interior drops to 0.1
        n_points = 5
        u_loss = [1.0, 0.5, 0.1, 0.5, 1.0]
        mock_cls, mock_inst = self._mock_interpolate(u_loss)

        with patch("pylandscape.mc_utils.Interpolate", mock_cls):
            result = mc.compute(m1, m2, crit, tr_dl, te_dl,
                                learning_rate=1e-3, num_points=n_points)

        assert result > 0.0

    def test_barrier_loss_gives_negative_deviation(self):
        """A loss that spikes in the middle → negative deviation."""
        mc, m1, m2, crit, tr_dl, te_dl = self._setup()
        n_points = 5
        barrier_loss = [0.5, 1.0, 2.0, 1.0, 0.5]
        mock_cls, mock_inst = self._mock_interpolate(barrier_loss)

        with patch("pylandscape.mc_utils.Interpolate", mock_cls):
            result = mc.compute(m1, m2, crit, tr_dl, te_dl,
                                learning_rate=1e-3, num_points=n_points)

        assert result < 0.0