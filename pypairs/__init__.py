"""
PyPairs: A Statistical Arbitrage System for Market Neutral Trading

This package implements a pairs trading strategy that identifies correlated
stocks and trades the spread between them for potential profit regardless
of market direction.
"""

from .data_engine import DataEngine

__all__ = ["DataEngine"]
__version__ = "0.1.0"
