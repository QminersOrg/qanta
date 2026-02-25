"""Qanta - A Python library for analysts and quants."""

from importlib.metadata import version

from qanta.binning import binned_means, weighted_percentile

__version__ = version('qanta')
__all__ = ['__version__', 'binned_means', 'weighted_percentile']
