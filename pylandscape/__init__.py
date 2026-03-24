try:
    from importlib.metadata import version
    __version__ = version("pylandscape")
except ImportError:
    # Fallback for development when package not installed
    import toml
    with open(Path(__file__).parent.parent / "pyproject.toml") as f:
        pyproject = toml.load(f)
        __version__ = pyproject["tool"]["poetry"]["version"]

from .metric import Metric
from .cka import CKA
from .hessian import Hessian
from .surface import Surface
from .mode_connectivity import ModeConnectivity
