from deepmem import data, model, networks, utils
from deepmem._version import version as __version__

__all__ = ["__version__", "data", "model", "networks", "utils"]


def __dir__():
    return __all__
