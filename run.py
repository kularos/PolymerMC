"""
Main simulation runner for polymer Monte Carlo.

This script orchestrates the full simulation workflow:
1. Configuration setup
2. Sampler and MCMC initialization
3. Batch processing over weight configurations
4. Visualization and animation generation
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple

from lib import SimulationConfig, VonMisesSampler, TorsionMCMC, PolymerAnalyzer
from lib import GlobPlotter, PhasePlotter, HistogramPlotter


class Timer:
    """Simple timer for performance monitoring."""

    def __init__(self):
        """Initialize timer."""
        self.start_time = time.perf_counter()
        self.last_lap = time.perf_counter()

    def lap(self) -> float:
        """
        Record a lap time.

        Returns:
            Time since last lap in seconds
        """
        now = time.perf_counter()
        delta = now - self.last_lap
        self.last_lap = now
        return delta

    @property
    def total(self) -> float:
        """Get total elapsed time in seconds."""
        return time.perf_counter() - self.start_time


def setup_plotters() -> Tuple[GlobPlotter, PhasePlotter, HistogramPlotter]:
    """
    Initialize plotting windows.

    Returns:
        Tuple of (glob_plotter, phase_plotter, histogram_plotter)
    """
    # Create separate figures for each plotter
    fig_glob = plt.figure(0)
    fig_phase = plt.figure(1)
    fig_hist = plt.figure(2)

    # Initialize plotters
    glob = GlobPlotter(fig_glob)
    glob.set_size(6, 6)

    phase = PhasePlotter(fig_phase)
    phase.set_size(8, 6)

    hist = HistogramPlotter(fig_hist)
    hist.set_size(6, 6)

    return glob, phase, hist


def generate_kappa_sweep(
        p_min: float = 5.0,
        p_max: float = 9.0,
        n_points: int = 101
) -> Tuple[np.ndarray, List[float]]:
    """
    Generate logarithmic sweep of concentration parameters.

    Uses the pS parameterization: κ = 10^(7 - pS)
    where pS ∈ [5, 9] gives κ ∈ [100, 0.01]

    Args:
        p_min: Minimum pS value (higher concentration)
        p_max: Maximum pS value (lower concentration)
        n_points: Number of points in sweep

    Returns:
        ps_values: Array of pS values
        kappas: List of corresponding κ values
    """
    ps_values = np.linspace(p_min, p_max, n_points)
    kappas = [10 ** (7 - pS) for pS in ps_values]
    return ps_values, kappas


def run_weight_batch(
        config: SimulationConfig,
        sampler: VonMisesSampler,
        markov: TorsionMCMC,
        plotters: Tuple,
        weights: Tuple[float, float, float],
        ps_values: np.ndarray,
        kappas: List[float],
        progress_interval: int = 50
) -> None:
    """
    Run simulation for one weight configuration.

    Args:
        config: Simulation configuration
        sampler: Von Mises sampler instance
        markov: MCMC chain builder
        plotters: Tuple of (glob, phase, hist) plotters
        weights: (T, Gp, Gn) weight configuration
        ps_values: Array of pS parameterization values
        kappas: List of concentration parameters
        progress_interval: Print progress every N frames
    """
    T, Gp, Gn = weights
    glob_plotter, phase_plotter, hist_plotter = plotters
    all_plotters = [glob_plotter, phase_plotter, hist_plotter]
    plotter_names = ["glob", "phase", "hist"]

    timer = Timer()

    # Generate samples for all kappa values
    print(f"🔄 Preparing data for W=({T},{Gp},{Gn})... ", end="", flush=True)
    all_samples = sampler(weights=weights, kappas=kappas)

    # Build chains via MCMC
    markov.run(all_samples)
    print(f"Done! ({timer.lap():.2f}s)")

    # Generate animation frames
    try:
        print(f"🎬 Generating animation frames...")

        for i, pS in enumerate(ps_values):
            # Extract data for this kappa value
            chains_i = markov.chains[..., i]
            sample_i = all_samples[..., [i]]
            kappa_i = kappas[i]

            # Analyze chains
            analyzer = PolymerAnalyzer(sample_i, chains_i)

            # Update all plots
            for plotter in all_plotters:
                # Set title with current parameters
                title = f"W=({T},{Gp},{Gn}), κ={kappa_i:.2e}, pS={pS:.2f}"
                plotter.fig.suptitle(title, x=0.05, ha="left")

                # Plot and capture frame
                plotter.plot(analyzer)
                plotter.capture_frame(dpi=config.frame_dpi)

            # Progress reporting
            frame_time = timer.lap()

            if (i + 1) % progress_interval == 0:
                total_time = timer.total
                print(
                    f"\r  ✅ Progress: {i + 1}/{len(ps_values)} frames | "
                    f"Total: {total_time:.2f}s",
                    end=""
                )

            # Per-frame status (overwrites)
            print(
                f"\r  ↳ [Frame {i + 1}/{len(ps_values)}] "
                f"κ: {kappa_i:.2e} | "
                f"Last: {frame_time * 1000:.1f}ms | "
                f"Total: {timer.total:.2f}s",
                end=""
            )

        print()  # Newline after progress

    finally:
        # Save GIFs even if interrupted
        print(f"\n💾 Saving GIFs for W=({T},{Gp},{Gn})...")

        output_dir = config.output_path(weights)

        for plotter, name in zip(all_plotters, plotter_names):
            gif_name = f"{name}_W({T},{Gp},{Gn})_{config.chain_length}x{config.n_chains}.gif"
            output_file = output_dir / gif_name

            print(f"  ", end="")
            plotter.save_gif(
                output_file,
                duration=config.gif_duration,
                loop=0
            )

        print(f"✅ Batch complete! Total time: {timer.total:.2f}s")
        print("-" * 60)


def run_simulation(
        config: SimulationConfig,
        weight_batches: List[Tuple[float, float, float]],
        ps_range: Tuple[float, float] = (5.0, 9.0),
        n_kappa_points: int = 101
) -> None:
    """
    Main simulation entry point.

    Args:
        config: Simulation configuration
        weight_batches: List of (T, Gp, Gn) weight configurations
        ps_range: (min, max) for pS parameter sweep
        n_kappa_points: Number of concentration parameters to sample
    """
    # Apply random seed
    config.apply_seed()

    # Initialize components
    print(f"\n{'=' * 60}")
    print(f"Polymer MCMC Simulation")
    print(f"{'=' * 60}")
    print(f"Chain length: {config.chain_length}")
    print(f"Number of chains: {config.n_chains}")
    print(f"Device: {config.device}")
    print(f"Seed: {config.seed}")
    print(f"{'=' * 60}\n")

    sampler = VonMisesSampler(config)
    markov = TorsionMCMC(config, n_batches=n_kappa_points)

    # Setup visualization
    plotters = setup_plotters()

    # Generate parameter sweep
    ps_values, kappas = generate_kappa_sweep(
        ps_range[0],
        ps_range[1],
        n_kappa_points
    )

    print(f"κ range: {kappas[0]:.2e} → {kappas[-1]:.2e}")
    print(f"Number of κ values: {len(kappas)}\n")

    # Process each weight configuration
    for weights in weight_batches:
        run_weight_batch(
            config=config,
            sampler=sampler,
            markov=markov,
            plotters=plotters,
            weights=weights,
            ps_values=ps_values,
            kappas=kappas
        )

    print(f"\n{'=' * 60}")
    print(f"Simulation complete!")
    print(f"{'=' * 60}\n")


def main():
    """Main entry point for the simulation."""

    # Create configuration
    config = SimulationConfig(
        chain_length=128,
        n_chains=8,
        seed=1738,
        device="cpu",  # Change to "cuda" if GPU available
        frame_dpi=256,
        gif_duration=20
    )

    # Define weight configurations to simulate
    # (Trans, Gauche+, Gauche-) - should sum to 1.0
    weight_batches = [
        (1.0, 0.0, 0.0),  # Pure trans
        (0.0, 0.0, 1.0),  # Pure gauche-
    ]

    # Run simulation
    run_simulation(
        config=config,
        weight_batches=weight_batches,
        ps_range=(5.0, 9.0),
        n_kappa_points=101
    )


if __name__ == "__main__":
    main()
