"""Prediction and betting recommendation modules."""

from .monte_carlo import MonteCarloSimulator
from .kelly import KellyCalculator

__all__ = [
    "MonteCarloSimulator",
    "KellyCalculator",
]
