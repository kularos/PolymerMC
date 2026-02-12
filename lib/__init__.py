"""
Polymer Monte Carlo Simulation Package

A modular framework for simulating polymer chains using Markov Chain
Monte Carlo with Von Mises torsion angle distributions.

Main Components:
---------------
config.SimulationConfig - Central configuration management
samplers.VonMisesSampler - Torsion angle sampling
markov.TorsionMCMC - Chain generation via MCMC
analysis.PolymerAnalyzer - Statistical analysis tools
plotting.* - Visualization classes

Module Structure:
----------------
├── run.py                  # Main simulation script
├── local/                  # Cache Venv and output isolation
└── lib/                    # Module library files
    ├── config.py           # Configuration management
    ├── core.py             # Geometric utilities (Rodrigues rotation, etc.)
    ├── samplers.py         # Von Mises sampler with caching
    ├── markov.py           # MCMC chain generation
    ├── analysis.py         # Chain analysis tools
    ├── plotting.py         # Visualization classes
    └── __init__.py         # This file

For detailed usage, see run.py or the documentation.
"""

from .config import SimulationConfig
from .samplers import VonMisesSampler
from .markov import TorsionMCMC
from .analysis import PolymerAnalyzer
from .core import rodrigues_rotation, align_geodesic
from .plotting import (
    GlobPlotter,
    PhasePlotter,
    HistogramPlotter,
    DashboardPlotter,
    BasePlotter
)

__version__ = "2.0.0"
__author__ = "Polymer Simulation Team"

__all__ = [
    # Configuration
    'SimulationConfig',

    # Core functionality
    'VonMisesSampler',
    'TorsionMCMC',
    'PolymerAnalyzer',

    # Geometric utilities
    'rodrigues_rotation',
    'align_geodesic',

    # Visualization
    'GlobPlotter',
    'PhasePlotter',
    'HistogramPlotter',
    'DashboardPlotter',
    'BasePlotter',
]
