"""
Visualization tools for polymer chain analysis.

This module provides plotting classes for visualizing polymer chains
and their statistical properties, with built-in support for animation
frame capture and GIF generation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import io
from PIL import Image
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from typing import Tuple, Optional
from pathlib import Path

# Constants for visualization
EQUATORIAL_ELEVATION = 0
DEFAULT_AZIMUTH = 45
CONFORMER_LABELS = ['Trans', 'Ecl+', 'Gch+', 'Cis', 'Gch-', 'Ecl-']
CONFORMER_ANGLES = np.arange(0, 360, 60)


class FrameCapture:
    """
    Helper class for capturing animation frames and creating GIFs.

    Manages the frame buffer and provides methods for saving to
    animated GIF files.
    """

    def __init__(self):
        """Initialize empty frame buffer."""
        self.frames: list[Image.Image] = []

    def capture(
            self,
            fig: plt.Figure,
            dpi: int = 64,
            reverse: bool = True
    ) -> None:
        """
        Capture current figure state as an image frame.

        Args:
            fig: Matplotlib figure to capture
            dpi: Resolution in dots per inch
            reverse: If True, insert at beginning (for reverse playback)
        """
        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi)
        buf.seek(0)
        img = Image.open(buf)

        if reverse:
            self.frames.insert(0, img)
        else:
            self.frames.append(img)

    def save_gif(
            self,
            output_path: Path | str,
            duration: int = 40,
            loop: int = 0
    ) -> None:
        """
        Save captured frames as animated GIF.

        Args:
            output_path: Output file path
            duration: Duration per frame in milliseconds
            loop: Number of loops (0 = infinite)
        """
        if not self.frames:
            print("⚠️  No frames captured, skipping GIF save")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.frames[0].save(
            output_path,
            save_all=True,
            append_images=self.frames[1:],
            optimize=False,
            duration=duration,
            loop=loop
        )

        print(f"✅ Saved GIF: {output_path}")
        self.frames = []  # Clear buffer


class BasePlotter:
    """
    Base class for all plotters.

    Provides common functionality for figure management and frame capture.
    """

    def __init__(self, fig: Optional[plt.Figure] = None, ax=None):
        """
        Initialize plotter with figure and axes.

        Args:
            fig: Matplotlib figure (creates new if None)
            ax: Matplotlib axes (creates new if None)
        """
        self.fig = fig if fig else plt.figure()
        self.ax = ax if ax else self.fig.add_subplot(111)
        self.frame_capture = FrameCapture()

    def set_size(self, width: float, height: float) -> None:
        """Set figure size in inches."""
        self.fig.set_figwidth(width)
        self.fig.set_figheight(height)

    def capture_frame(self, dpi: int = 64, reverse: bool = True) -> None:
        """Capture current state as animation frame."""
        self.frame_capture.capture(self.fig, dpi=dpi, reverse=reverse)

    def save_gif(
            self,
            output_path: Path | str,
            duration: int = 40,
            loop: int = 0
    ) -> None:
        """Save captured frames as GIF."""
        self.frame_capture.save_gif(output_path, duration, loop)

    def plot(self, analyzer):
        """
        Plot analyzer data. Must be implemented by subclasses.

        Args:
            analyzer: PolymerAnalyzer instance
        """
        raise NotImplementedError("Subclasses must implement plot()")


class GlobPlotter(BasePlotter):
    """
    3D visualization of polymer chains.

    Shows chains aligned along their end-to-end vectors with
    equatorial viewing angle for maximum visual clarity.
    """

    def __init__(
            self,
            fig: Optional[plt.Figure] = None,
            ax=None,
            alpha: float = 0.8
    ):
        """
        Initialize 3D plotter.

        Args:
            fig: Matplotlib figure
            ax: 3D axes (will create if None)
            alpha: Transparency for chain rendering
        """
        if ax is None:
            fig = fig if fig else plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        super().__init__(fig, ax)
        self.alpha = alpha

    def plot(
            self,
            analyzer,
            max_dist: Optional[float] = None
    ) -> None:
        """
        Plot 3D polymer chains.

        Args:
            analyzer: PolymerAnalyzer instance
            max_dist: Maximum plot extent (auto-computed if None)
        """
        self.ax.clear()
        self.ax.set_axis_off()

        # Align chains along z-axis and center at median monomer
        chains = analyzer.align_ends().numpy()
        chains[2] = chains[2] - chains[2, -1] / 2

        # Set viewing limits
        if max_dist is None:
            max_dist = analyzer.chain_length * 0.8

        half_range = max_dist / 2
        self.ax.set_xlim(-half_range, half_range)
        self.ax.set_ylim(-half_range, half_range)
        self.ax.set_zlim(-half_range, half_range)
        self.ax.set_box_aspect([1, 1, 1])

        # Equatorial view
        self.ax.view_init(elev=EQUATORIAL_ELEVATION, azim=DEFAULT_AZIMUTH)

        # Plot each chain
        for i in range(chains.shape[2]):
            x, y, z = chains[:, :, i]
            self.ax.plot3D(x, y, z, alpha=self.alpha, lw=1.5)


class HistogramPlotter(BasePlotter):
    """
    Polar histogram of torsion angle distributions.

    Displays the conformational preferences with standard
    conformer labels (Trans, Gauche+, Gauche-, etc.).
    """

    def __init__(self, fig: Optional[plt.Figure] = None, ax=None):
        """
        Initialize polar histogram plotter.

        Args:
            fig: Matplotlib figure
            ax: Polar axes (will create if None)
        """
        if ax is None:
            fig = fig if fig else plt.figure()
            ax = fig.add_subplot(111, projection="polar")

        super().__init__(fig, ax)

    def plot(self, analyzer, bins: int = 81) -> None:
        """
        Plot torsion angle distribution.

        Args:
            analyzer: PolymerAnalyzer instance
            bins: Number of histogram bins
        """
        self.ax.clear()

        # Configure polar plot
        self.ax.set_title("Bond Torsion Distribution")
        self.ax.set_yticklabels([])
        self.ax.set_theta_zero_location("N")
        self.ax.set_thetagrids(CONFORMER_ANGLES, labels=CONFORMER_LABELS)

        # Compute histogram
        sample = analyzer.taus.flatten()
        counts, bin_edges = np.histogram(
            sample,
            bins=bins,
            range=[-np.pi, np.pi]
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Plot as polar bar chart
        self.ax.bar(
            bin_centers,
            counts,
            width=(2 * np.pi) / bins,
            bottom=0.0
        )


class PhasePlotter(BasePlotter):
    """
    Phase space plot: net torsion vs. displacement.

    Shows correlation between winding (cumulative torsion) and
    extension (end-to-end distance) with covariance ellipses.
    """

    def __init__(self, fig: Optional[plt.Figure] = None, ax=None):
        """Initialize phase space plotter."""
        super().__init__(fig, ax)

    def _compute_ensemble_stats(
            self,
            analyzer
    ) -> Tuple[Tuple[float, float], np.ndarray]:
        """
        Compute ensemble mean and covariance.

        Args:
            analyzer: PolymerAnalyzer instance

        Returns:
            mean: (mean_torsion, mean_displacement)
            cov: 2x2 covariance matrix
        """
        # Final state for all chains
        net_torsion = analyzer.torsion[-1, :] / (2 * np.pi)  # Convert to turns
        displacement = analyzer.dist[0, -1, :]

        mean = (np.mean(net_torsion), np.mean(displacement))
        cov = np.cov(net_torsion, displacement)

        return mean, cov

    def _plot_covariance_ellipse(
            self,
            mean: Tuple[float, float],
            cov: np.ndarray,
            n_std: float = 1.0,
            **kwargs
    ) -> None:
        """
        Draw covariance ellipse.

        Args:
            mean: Center point (x, y)
            cov: 2x2 covariance matrix
            n_std: Number of standard deviations for ellipse size
            **kwargs: Additional arguments for Ellipse patch
        """
        # Pearson correlation coefficient
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

        # Ellipse radii
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)

        # Create ellipse
        ellipse = Ellipse(
            (0, 0),
            width=ell_radius_x * 2,
            height=ell_radius_y * 2,
            **kwargs
        )

        # Transform: scale by std dev, rotate, translate to mean
        scale_x = np.sqrt(cov[0, 0]) * n_std
        scale_y = np.sqrt(cov[1, 1]) * n_std

        transform = (
            transforms.Affine2D()
            .rotate_deg(45)
            .scale(scale_x, scale_y)
            .translate(*mean)
        )

        ellipse.set_transform(transform + self.ax.transData)
        self.ax.add_patch(ellipse)

    def plot(
            self,
            analyzer,
            max_dist: Optional[float] = None,
            torsion_range: float = 0.5
    ) -> None:
        """
        Plot phase space trajectory.

        Args:
            analyzer: PolymerAnalyzer instance
            max_dist: Maximum displacement (auto if None)
            torsion_range: Fraction of chain_length for torsion axis
        """
        self.ax.clear()
        self.ax.set_title("Phase Space")
        self.ax.set_xlabel("Net Torsion (Turns)")
        self.ax.set_ylabel("Net Displacement (Bond Units)")

        # Plot individual chain trajectories
        for i in range(analyzer.shape[2]):
            torsion = analyzer.torsion[:, i] / (2 * np.pi)
            displacement = analyzer.dist[0, :, i]
            self.ax.plot(torsion, displacement, lw=1, alpha=0.7)

        # Compute and display ensemble statistics
        mean, cov = self._compute_ensemble_stats(analyzer)

        # 2-sigma and 1-sigma confidence ellipses
        self._plot_covariance_ellipse(
            mean, cov, n_std=2.0,
            facecolor='tab:red', alpha=0.1
        )
        self._plot_covariance_ellipse(
            mean, cov, n_std=1.0,
            facecolor='tab:red', alpha=0.3
        )

        # Set axis limits
        if max_dist is None:
            max_dist = analyzer.chain_length

        self.ax.set_xlim(-max_dist * torsion_range, max_dist * torsion_range)
        self.ax.set_ylim(0, max_dist)
        self.ax.grid(True, linestyle='--', alpha=0.3)


class DashboardPlotter(BasePlotter):
    """
    Composite dashboard with multiple synchronized plots.

    Combines 3D visualization, phase space, and histogram in
    a single figure layout.
    """

    def __init__(self):
        """Initialize dashboard with multi-panel layout."""
        super().__init__()

        self.fig = plt.figure(figsize=(19.20, 10.80))

        # Create grid layout: 2 rows, 2 columns
        gs = gridspec.GridSpec(
            2, 2,
            height_ratios=[3, 2],
            width_ratios=[1, 2]
        )

        # Initialize sub-plotters
        self.glob = GlobPlotter(
            self.fig,
            self.fig.add_subplot(gs[:, 0], projection='3d')
        )
        self.phase = PhasePlotter(
            self.fig,
            self.fig.add_subplot(gs[0, 1])
        )
        self.hist = HistogramPlotter(
            self.fig,
            self.fig.add_subplot(gs[1, 1], projection="polar")
        )

    def plot(self, analyzer) -> None:
        """
        Update all sub-plots.

        Args:
            analyzer: PolymerAnalyzer instance
        """
        self.glob.plot(analyzer)
        self.phase.plot(analyzer)
        self.hist.plot(analyzer)
        self.fig.tight_layout()

    def capture_frame(self, dpi: int = 256, reverse: bool = True) -> None:
        """Capture frame from main figure."""
        self.frame_capture.capture(self.fig, dpi=dpi, reverse=reverse)


# Export all plotter classes
__all__ = [
    'GlobPlotter',
    'HistogramPlotter',
    'PhasePlotter',
    'DashboardPlotter',
    'BasePlotter',
]
