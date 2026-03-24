"""Test PyLandscape module imports and basic functionality."""
import sys
from pathlib import Path

import pytest


def test_module_import():
    """Test that main module can be imported."""
    import pylandscape
    assert pylandscape is not None


def test_hessian_import():
    """Test that Hessian module can be imported."""
    from pylandscape import Hessian
    assert Hessian is not None


def test_cka_import():
    """Test that CKA module can be imported."""
    from pylandscape import CKA
    assert CKA is not None


def test_mode_connectivity_import():
    """Test that ModeConnectivity module can be imported."""
    from pylandscape import ModeConnectivity
    assert ModeConnectivity is not None


def test_surface_import():
    """Test that Surface module can be imported."""
    from pylandscape import Surface
    assert Surface is not None


def test_metric_import():
    """Test that metric module can be imported."""
    from pylandscape import metric
    assert metric is not None


def test_mc_utils_import():
    """Test that mc_utils module can be imported."""
    from pylandscape import mc_utils
    assert mc_utils is not None


def test_package_structure():
    """Test that package has expected structure."""
    import pylandscape

    # Check for expected modules
    assert hasattr(pylandscape, "Hessian")
    assert hasattr(pylandscape, "CKA")
    assert hasattr(pylandscape, "ModeConnectivity")
    assert hasattr(pylandscape, "Surface")
    assert hasattr(pylandscape, "metric")
    assert hasattr(pylandscape, "mc_utils")


def test_version():
    """Test that package has a version attribute."""
    import pylandscape
    # Version should be a string or semver-like
    assert isinstance(pylandscape.__version__, str)


def test_import_paths():
    """Test that modules can be imported from top level."""
    # These should all import without errors
    import pylandscape.hessian as hessian
    import pylandscape.cka as cka
    import pylandscape.mode_connectivity as mc
    import pylandscape.surface as surface
    import pylandscape.metric as metric
    import pylandscape.mc_utils as mc_utils

    assert hessian is not None
    assert cka is not None
    assert mc is not None
    assert surface is not None
    assert metric is not None
    assert mc_utils is not None