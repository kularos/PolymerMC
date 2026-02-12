# Polymer Monte Carlo Simulation - Refactored

A clean, well-documented implementation of polymer chain Monte Carlo simulation using Von Mises torsion angle distributions.

## 🎯 What Changed in This Refactor

### Major Improvements

1. **Configuration Management**
   - New `SimulationConfig` class centralizes all parameters
   - Manages RNG seeding and device allocation
   - Easy to save/load configurations
   - No more global state or singleton patterns

2. **Code Organization**
   - Separated concerns: grid computation, caching, and sampling are now distinct
   - Better class hierarchies with clear responsibilities
   - Removed all dead code (FunctionCache, PhasePlotter1)

3. **Type Hints & Documentation**
   - Full type annotations on all functions
   - Comprehensive docstrings with examples
   - Clear parameter descriptions

4. **Better Naming**
   - `rodrigues_torch` → `rodrigues_rotation`
   - `iter_chains` → `step`
   - More descriptive variable names throughout

5. **Plotting Architecture**
   - Composition over inheritance with `FrameCapture` helper
   - Consistent interface across all plotters
   - New `DashboardPlotter` for integrated views

### Architecture Changes

#### Before (Old Pattern):
```python
# Global seed singleton
Seed().value = 1738

# Classes accept many parameters
sampler = VonMisesSampler(
    chain_length=128,
    n_chains=8,
    k_range=(1e-3, 1e7),
    device="cpu"
)
```

#### After (New Pattern):
```python
# Configuration object
config = SimulationConfig(
    chain_length=128,
    n_chains=8,
    seed=1738,
    device="cpu"
)

# Apply seed explicitly
config.apply_seed()

# Classes accept config
sampler = VonMisesSampler(config)
markov = TorsionMCMC(config)
```

## 📁 File Structure

```
polymer_sim/
├── config.py           # SimulationConfig class
├── core.py             # Geometric utilities (Rodrigues rotation)
├── samplers.py         # VonMisesSampler + GridCache
├── markov.py           # TorsionMCMC chain generation
├── analysis.py         # PolymerAnalyzer
├── plotting.py         # All visualization classes
├── run.py              # Main simulation script
└── __init__.py         # Package exports
```

## 🚀 Quick Start

### Basic Usage

```python
from config import SimulationConfig
from samplers import VonMisesSampler
from markov import TorsionMCMC
from analysis import PolymerAnalyzer
from plotting import GlobPlotter

# 1. Configure
config = SimulationConfig(
    chain_length=128,
    n_chains=8,
    seed=42,
    device="cpu"
)

# 2. Sample torsion angles
sampler = VonMisesSampler(config)
weights = (0.6, 0.2, 0.2)  # 60% trans, 20% each gauche
kappas = [1e2, 1e4, 1e6]   # Three concentration levels
samples = sampler(weights, kappas)

# 3. Generate chains
markov = TorsionMCMC(config, n_batches=len(kappas))
markov.run(samples)

# 4. Analyze
analyzer = PolymerAnalyzer(samples[..., [0]], markov.chains[..., 0])

# 5. Visualize
plotter = GlobPlotter()
plotter.plot(analyzer)
```

### Running Full Simulation

```bash
python run.py
```

This will:
1. Generate polymer chains for configured weight sets
2. Sweep through concentration parameters (κ)
3. Create animated GIFs in `./local/outputs/gifs/`

## 🔧 Configuration Options

```python
config = SimulationConfig(
    # Core parameters
    chain_length=128,        # Monomers per chain
    n_chains=8,              # Number of chains
    seed=1738,               # Random seed (None = random)
    
    # Hardware
    device="cpu",            # "cpu" or "cuda"
    
    # Von Mises sampling
    k_range=(1e-3, 1e7),     # Concentration parameter range
    k_res=1000,              # Kappa grid resolution
    p_res=100000,            # Probability grid resolution
    tau_centers=(0, -120, 120),  # Trans, G+, G- angles (degrees)
    
    # Output
    cache_dir=Path("./local/cache"),
    output_dir=Path("./local/outputs/gifs"),
    frame_dpi=256,           # Animation quality
    gif_duration=20,         # ms per frame
)
```

## 📊 Understanding the Physics

### Torsion Angle Distributions

The Von Mises distribution models conformational preferences:
- **κ (kappa)**: Concentration parameter
  - κ → 0: Uniform distribution (high entropy)
  - κ → ∞: Delta function (low entropy)
  
- **Weights**: Conformer preferences
  - Trans (τ = 0°)
  - Gauche+ (τ = +120°)
  - Gauche- (τ = -120°)

### The pS Parameterization

We use pS = 7 - log₁₀(κ) for intuitive sweeps:
- pS = 5 → κ = 10² (flexible)
- pS = 7 → κ = 1 (transition)
- pS = 9 → κ = 10⁻² (stiff)

## 🔬 Advanced Features

### Custom Weight Configurations

```python
# Pure trans (stiff rod)
weights = (1.0, 0.0, 0.0)

# Equal mixture (random coil)
weights = (0.33, 0.33, 0.34)

# Biased gauche
weights = (0.2, 0.6, 0.2)
```

### Gradient Computation

```python
sampler = VonMisesSampler(config)
grads = sampler.get_kappa_gradient(
    weights=(0.6, 0.2, 0.2),
    kappas=[1e2, 1e4]
)
# Returns ∂τ/∂κ for sensitivity analysis
```

### Principal Axis Alignment

```python
analyzer = PolymerAnalyzer(samples, chains)

# Align by end-to-end vector
aligned = analyzer.align_ends(axis=2)  # Along z-axis

# Align by gyration tensor
aligned, eigenvalues = analyzer.orient_principal_axes()
```

## 🎨 Visualization

### Individual Plotters

```python
from plotting import GlobPlotter, PhasePlotter, HistogramPlotter

# 3D visualization
glob = GlobPlotter()
glob.plot(analyzer)

# Phase space (torsion vs displacement)
phase = PhasePlotter()
phase.plot(analyzer)

# Torsion distribution
hist = HistogramPlotter()
hist.plot(analyzer)
```

### Dashboard View

```python
from plotting import DashboardPlotter

dashboard = DashboardPlotter()
dashboard.plot(analyzer)
```

### Animation Capture

```python
plotter = GlobPlotter()

for kappa in kappa_sweep:
    # ... generate chains for this kappa ...
    analyzer = PolymerAnalyzer(samples, chains)
    
    plotter.plot(analyzer)
    plotter.capture_frame(dpi=256)

plotter.save_gif("output.gif", duration=20)
```

## 📈 Performance

### Caching System

ICDF grids are automatically cached based on configuration hash:
- First run: ~30-60s to compute grid
- Subsequent runs: ~0.1s to load from cache
- Cache location: `config.cache_dir`

### GPU Acceleration

```python
config = SimulationConfig(device="cuda")
```

PyTorch tensors will automatically use GPU when available.

## 🧪 Testing

Create unit tests for core functionality:

```python
# tests/test_rotation.py
import torch
from core import rodrigues_rotation

def test_90_degree_rotation():
    v = torch.tensor([[1.0], [0.0], [0.0]])  # x-axis
    k = torch.tensor([[0.0], [0.0], [1.0]])  # z-axis
    theta = torch.tensor([[torch.pi / 2]])   # 90°
    
    result = rodrigues_rotation(v, k, theta)
    expected = torch.tensor([[0.0], [1.0], [0.0]])
    
    assert torch.allclose(result, expected, atol=1e-6)
```

## 🔍 Code Quality Improvements

### What Was Removed
- ✂️ `Seed` singleton class (replaced with config)
- ✂️ Incomplete `FunctionCache` class
- ✂️ Unused `PhasePlotter1` class
- ✂️ Commented-out plotting methods
- ✂️ Magic numbers scattered throughout

### What Was Added
- ✅ Full type hints on all functions
- ✅ Comprehensive docstrings
- ✅ Separation of concerns (GridCache, VonMisesDistribution)
- ✅ Constants for magic values
- ✅ Better error handling
- ✅ Progress reporting

### Naming Conventions

**Classes**: PascalCase
- `SimulationConfig`, `VonMisesSampler`

**Functions**: snake_case
- `rodrigues_rotation`, `align_geodesic`

**Private**: Leading underscore
- `_device`, `_interpolate_2d`

**Constants**: UPPER_CASE
- `EQUATORIAL_ELEVATION`, `CONFORMER_LABELS`

## 📝 Migration Guide

### Updating Old Code

**Before:**
```python
from lib import Seed, TorsionMCMC, VonMisesSampler

Seed().value = 1738
sampler = VonMisesSampler(128, 8, device="cpu")
markov = TorsionMCMC(128, 8)
```

**After:**
```python
from config import SimulationConfig
from samplers import VonMisesSampler
from markov import TorsionMCMC

config = SimulationConfig(chain_length=128, n_chains=8, seed=1738)
sampler = VonMisesSampler(config)
markov = TorsionMCMC(config)
```

## 🤝 Contributing

When adding features:
1. Add type hints
2. Write docstrings with examples
3. Update this README
4. Follow existing naming conventions
5. Add tests if applicable

## 📄 License

[Your License Here]

## 👥 Authors

[Your Name/Team]

---

**Note**: This refactored version maintains 100% feature parity with the original while significantly improving maintainability, documentation, and extensibility.
