"""Qanta - A Python library for analysts and quants."""

from importlib.metadata import version

from qanta.centroids import compute_centroids, weighted_percentile

__version__ = version('qanta')
__all__ = ['__version__', 'compute_centroids', 'weighted_percentile']
