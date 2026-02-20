"""
Main simulation runner for polymer Monte Carlo.

This script orchestrates the full simulation workflow:
1. Configuration setup
2. Sampler and MCMC initialization
3. Batch processing over weight configurations
4. Visualization and animation generation

Animation structure (stacked loops, pS increases fastest):
    Outer loop: pGamma in 0 .. pL  (bisection scale)
    Inner loop: pS values          (entropy / concentration)

    Frame index: i = pGamma_idx * n_pS + pS_idx
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
        self.start_time = time.perf_counter()
        self.last_lap = time.perf_counter()

    def lap(self) -> float:
        now = time.perf_counter()
        delta = now - self.last_lap
        self.last_lap = now
        return delta

    @property
    def total(self) -> float:
        return time.perf_counter() - self.start_time


def setup_plotters() -> Tuple[GlobPlotter, PhasePlotter, HistogramPlotter]:
    """
    Initialize plotting windows.

    Returns:
        Tuple of (glob_plotter, phase_plotter, histogram_plotter)
    """
    fig_glob  = plt.figure(0)
    fig_phase = plt.figure(1)
    fig_hist  = plt.figure(2)

    glob = GlobPlotter(fig_glob)
    glob.set_size(6, 6)

    phase = PhasePlotter(fig_phase)
    phase.set_size(8, 6)

    hist = HistogramPlotter(fig_hist)
    hist.set_size(6, 6)

    return glob, phase, hist


def generate_pS_sweep(
    pS_min: float = 5.0,
    pS_max: float = 9.0,
    n_points: int = 101
) -> np.ndarray:
    """
    Generate linear sweep of entropy parameters.

    pS = 7 - log10(κ):  higher pS → higher entropy → more flexible chains.

    Args:
        pS_min:   Minimum pS value (stiffer,   pS=5 → κ=100)
        pS_max:   Maximum pS value (more flex., pS=9 → κ=0.01)
        n_points: Number of points in sweep

    Returns:
        Array of pS values
    """
    return np.linspace(pS_min, pS_max, n_points)


def run_weight_batch(
    config: SimulationConfig,
    sampler: VonMisesSampler,
    markov: TorsionMCMC,
    plotters: Tuple,
    weights: Tuple[float, float, float],
    pS_values: np.ndarray,
) -> None:
    """
    Run simulation for one weight configuration.

    Generates a single PolymerAnalyzer (one long chain, all pS values),
    then sweeps over (pGamma × pS) to produce animation frames.

    Frame order: pS increases fastest (inner), pGamma increases slowest
    (outer), so frame index = pGamma_idx * n_pS + pS_idx.

    Args:
        config:     Simulation configuration
        sampler:    Von Mises sampler instance
        markov:     MCMC chain builder
        plotters:   Tuple of (glob, phase, hist) plotters
        weights:    (T, Gp, Gn) weight configuration
        pS_values:  Array of entropy parameters (length K)
    """
    T, Gp, Gn = weights
    glob_plotter, phase_plotter, hist_plotter = plotters
    all_plotters  = [glob_plotter, phase_plotter, hist_plotter]
    plotter_names = ["glob", "phase", "hist"]

    timer = Timer()
    n_pS    = len(pS_values)
    pGamma_values = list(config.valid_pGamma_range)  # 0 .. pL inclusive
    n_pGamma = len(pGamma_values)
    n_frames_total = n_pGamma * n_pS

    # ------------------------------------------------------------------
    # 1. Sample torsion angles for all pS values at once
    # ------------------------------------------------------------------
    print(f"🔄 Preparing data for W=({T},{Gp},{Gn})... ", end="\n\t↳ ", flush=True)
    all_samples = sampler(weights=weights, pS_values=pS_values)
    # all_samples: (1, 2^pL, K)

    # ------------------------------------------------------------------
    # 2. Build single long chain via MCMC (one pass, all K pS values)
    # ------------------------------------------------------------------
    markov.run(all_samples)
    print(f"Done! ({timer.lap():.2f}s)")
    # markov.chains: (3, 2^pL, K)

    # ------------------------------------------------------------------
    # 3. Instantiate ONE analyzer for this entire weight batch
    # ------------------------------------------------------------------
    analyzer = PolymerAnalyzer(
        torsion_samples=all_samples,
        chains=markov.chains,
        pL=config.pL,
        device=config.device,
    )

    # ------------------------------------------------------------------
    # 4. Stacked animation loop: outer=pGamma, inner=pS (pS fastest)
    # ------------------------------------------------------------------
    try:
        print(f"🎬 Generating {n_frames_total} animation frames "
              f"({n_pGamma} pGamma × {n_pS} pS)...")

        frame_idx = 0
        for pGamma in pGamma_values:
            for pS_idx, pS in enumerate(pS_values):

                # Lazy bisection for this (pGamma, pS) coordinate
                view = analyzer.generate_assets(pGamma=pGamma, pS_idx=pS_idx)

                kappa = config.pS_to_kappa(pS)
                n_sub = view.n_chains
                L_sub = view.chain_length

                # Update all plots
                for plotter in all_plotters:
                    title = (
                        f"W=({T},{Gp},{Gn})  "
                        f"pGamma={pGamma} ({n_sub}×{L_sub})  "
                        f"pS={pS:.2f} (κ={kappa:.2e})"
                    )
                    plotter.fig.suptitle(title, x=0.05, ha="left")
                    plotter.plot(view)
                    plotter.capture_frame(dpi=config.frame_dpi)

                frame_time = timer.lap()
                frame_idx += 1

                print(
                    f"\r  ↳ [Frame {frame_idx}/{n_frames_total}] "
                    f"pGamma={pGamma} | pS={pS:.2f} (κ={kappa:.2e}) | "
                    f"Last: {frame_time * 1000:.1f}ms | "
                    f"Total: {timer.total:.2f}s",
                    end=""
                )

        print()  # newline after progress bar

    finally:
        # Save GIFs even if interrupted mid-sweep
        print(f"\n💾 Saving GIFs for W=({T},{Gp},{Gn})...")

        output_dir = config.output_path(weights)

        for plotter, name in zip(all_plotters, plotter_names):
            gif_name = (
                f"{name}_W({T},{Gp},{Gn})"
                f"_pL{config.pL}"
                f"_{n_pGamma}pGamma"
                f"_{n_pS}pS.gif"
            )
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
    pS_range: Tuple[float, float] = (5.0, 9.0),
    n_pS_points: int = 101
) -> None:
    """
    Main simulation entry point.

    Args:
        config:         Simulation configuration (pL determines pGamma sweep)
        weight_batches: List of (T, Gp, Gn) weight configurations
        pS_range:       (min, max) for pS parameter sweep
        n_pS_points:    Number of pS values to sample
    """
    config.apply_seed()

    n_pGamma = config.pL + 1  # 0 .. pL inclusive
    n_frames  = n_pGamma * n_pS_points

    print(f"\n{'='*60}")
    print(f"Polymer MCMC Simulation")
    print(f"{'='*60}")
    print(f"Total length:   2^{config.pL} = {config.total_length} monomers")
    print(f"pGamma sweep:   0 .. {config.pL} ({n_pGamma} levels)")
    print(f"pS sweep:       {pS_range[0]} → {pS_range[1]} ({n_pS_points} points)")
    print(f"Frames/batch:   {n_pGamma} × {n_pS_points} = {n_frames}")
    print(f"Device:         {config.device}")
    print(f"Seed:           {config.seed}")
    print(f"{'='*60}\n")

    sampler = VonMisesSampler(config)
    markov  = TorsionMCMC(config, n_batches=n_pS_points)

    plotters  = setup_plotters()
    pS_values = generate_pS_sweep(pS_range[0], pS_range[1], n_pS_points)

    kappa_min = config.pS_to_kappa(pS_range[1])
    kappa_max = config.pS_to_kappa(pS_range[0])
    print(f"pS range:  {pS_range[0]} → {pS_range[1]}")
    print(f"κ range:   {kappa_max:.2e} → {kappa_min:.2e}\n")

    for weights in weight_batches:
        run_weight_batch(
            config=config,
            sampler=sampler,
            markov=markov,
            plotters=plotters,
            weights=weights,
            pS_values=pS_values,
        )

    print(f"\n{'='*60}")
    print(f"Simulation complete!")
    print(f"{'='*60}\n")


def main(pL, pS_):
    """Main entry point for the simulation."""

    config = SimulationConfig(
        pL=pL,
        seed=1738,
        device="cpu",
        pS_range=(2.0, 12.0),
        frame_dpi=256,
        gif_duration=20
    )

    weight_batches = [
        (1, 0, 0),  # Pure trans
        (1, 1, 1),  # Maximum Entropy
        #(0, 0, 1),  # Pure gauche-
        #(0, 1, 0),  # Pure gauche+
        #(0, 1, 1),  # Degenerate Diamond collapse
    ]

    run_simulation(
        config=config,
        weight_batches=weight_batches,
        pS_range=pS_,
        n_pS_points=201
    )


if __name__ == "__main__":
    main(pL=8, pS_=(5.0, 9.0))