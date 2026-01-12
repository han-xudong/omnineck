"""
Enable `omnineck.__version__` to be imported.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("omnineck")
except PackageNotFoundError:
    __version__ = "unknown"
