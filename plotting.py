import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import io
from PIL import Image

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def get_ensemble_stats(analyzer):
    # Get the final state for all N chains
    # x: Net Torsion (Turns), y: Displacement
    x = analyzer.torsion[-1, :] / (2 * np.pi)
    y = analyzer.dist[0, -1, :]

    mu_x, mu_y = np.mean(x), np.mean(y)
    cov = np.cov(x, y)

    return (mu_x, mu_y), cov

class GifPlotter:
    """
    Base class that handles the memory buffer for animation frames.
    Allows for capturing frames and compiling them into GIFs.
    """

    def __init__(self, fig=None, ax=None):
        self.fig = fig if fig else plt.figure()
        self.ax = ax if ax else self.fig.add_subplot(111)
        self.frames = []

    def set_shape(self, width, height):
        self.fig.set_figheight(height)
        self.fig.set_figwidth(width)

    def capture_frame(self, fig, reversed=True, dpi=64):
        """Captures the current figure state into the frame buffer."""
        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi)
        buf.seek(0)
        img = Image.open(buf)
        if not reversed:
            self.frames.append(img)
        else:
            self.frames.insert(0, img)

    def save_gif(self, output_path, duration=40, loop=0):
        """Compiles the captured frames into an animated GIF."""
        if not self.frames:
            print("⚠️ No frames captured.")
            return
        self.frames[0].save(
            output_path, save_all=True, append_images=self.frames[1:],
            optimize=False, duration=duration, loop=loop
        )
        self.frames = []  # Clear memory after saving
        print(f"✅ Saved GIF: {output_path}")


class GlobPlotter(GifPlotter):
    """3D Equatorial visualization with Median-Monomer Centering."""

    def __init__(self, fig=None, ax=None, alpha=0.8):
        fig = fig if fig else plt.figure()
        ax = ax if ax else fig.add_subplot(111, projection="3d")
        super().__init__(fig, ax)

        self.alpha = alpha

    def plot(self, analyzer, max_dist=None):
        self.ax.clear()
        self.ax.set_axis_off()

        # 1. Align to Z-axis and convert to numpy
        chains = analyzer.align_ends().numpy()  # Shape: (3, L, N)


        chains[2] = chains[2] - chains[2, -1] / 2

        # 4. Set Dynamic Scaling & Equatorial View
        if max_dist is None:
            max_dist = analyzer.chain_length * 0.8

        self.ax.set_xlim(-max_dist/2, max_dist/2)
        self.ax.set_ylim(-max_dist/2, max_dist/2)
        self.ax.set_zlim(-max_dist/2, max_dist/2)
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.view_init(elev=0, azim=45)  # View from the equator

        for i in range(chains.shape[2]):
            c = chains[:, :, i]
            self.ax.plot3D(c[0], c[1], c[2], alpha=self.alpha, lw=1.5)


class HistogramPlotter(GifPlotter):
    """Polar distributions for torsion angles."""

    def __init__(self, fig, ax=None):
        fig = fig if fig else plt.figure()
        ax = ax if ax else fig.add_subplot(111, projection="polar")
        super().__init__(fig, ax)


    def plot(self, analyzer, bins=81):
        self.ax.clear()

        self.ax.set_title("Bond Torsion Distribution:")
        self.ax.set_yticklabels([])
        self.ax.set_theta_zero_location("N")
        angles = np.arange(0, 360, 60)
        labels = ['Trns', 'Ecl+', 'Gch+', 'Cis', 'Gch-', 'Ecl-']
        self.ax.set_thetagrids(angles, labels=labels)

        sample = analyzer.taus.flatten()
        counts, bin_edges = np.histogram(sample, bins=bins, range=[-np.pi, np.pi])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        self.ax.bar(bin_centers, counts, width=(2 * np.pi) / bins, bottom=0.0)


class GraphPlotter(GifPlotter):
    """2D Cartesian plots for physical properties."""

    def plot(self, analyzer):
        #self._plot_torsion(analyzer)
        #self._plot_dist(analyzer)
        self._plot_neighbors(analyzer)

    def _plot_torsion(self, analyzer):
        self.ax.clear()
        n = np.arange(0, analyzer.shape[1])
        torsion = analyzer.torsion / (2 * np.pi)
        for i in range(torsion.shape[1]):
            self.ax.plot(n, torsion[:, i], lw=0.5, alpha=0.8)

    def _plot_dist(self, analyzer):
        self.ax.clear()
        n = np.arange(0, analyzer.shape[1])
        dist = analyzer.dist
        for i in range(analyzer.shape[2]):
            self.ax.plot(n, dist[:, 0, i], lw=0.5, alpha=0.8)

    def _plot_neighbors(self, analyzer, max_dist=32):
        self.ax.clear()
        radii = np.linspace(0, max_dist, 64)
        neighbors = np.stack([analyzer.mean_neighbors(r) for r in radii], axis=1)
        for i in range(analyzer.shape[2]):
            self.ax.plot(radii, neighbors[i], lw=0.5, alpha=0.8)


class PhasePlotter(GifPlotter):
    """X-Y Scatter showing Correlation between Winding and Extension."""

    def plot_ellipse(self, mean, cov, n_std=1.0, **kwargs):
        """Draws a covariance ellipse showing the ensemble spread."""
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)

        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs)

        # Scaling and rotation based on eigenvalues
        scale_x = np.sqrt(cov[0, 0]) * n_std
        scale_y = np.sqrt(cov[1, 1]) * n_std

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean[0], mean[1])

        ellipse.set_transform(transf + self.ax.transData)
        return self.ax.add_patch(ellipse)

    def plot(self, analyzer,  max_dist=None, tau_slew=0.5):
        self.ax.clear()
        self.ax.set_title("Phase Space")
        self.ax.set_xlabel("Net Torsion (Turns)")
        self.ax.set_ylabel("Net Displacement (Bond Units)")


        for i in range(analyzer.shape[2]):
            x = analyzer.torsion[:, i] / (2 * np.pi)
            y = analyzer.dist[0, :, i]

            # Scatter with transparency to handle overlapping chain data
            self.ax.plot(x, y, lw=1)

        # 2. Calculate Ensemble Stats
        (mx, my), cov = get_ensemble_stats(analyzer)

        # 3. Display the spread (The "Statistics")
        # 2-Sigma (95% confidence) - Faint
        self.plot_ellipse((mx, my), cov, n_std=2.0, facecolor='tab:red', alpha=0.1)
        # 1-Sigma (68% confidence) - Stronger
        self.plot_ellipse((mx, my), cov, n_std=1.0, facecolor='tab:red', alpha=0.3)

        # Set dynamic limits to keep the "zoom" consistent
        if max_dist is None:
            max_dist = analyzer.chain_length
        self.ax.set_xlim(-max_dist * tau_slew, max_dist * tau_slew)
        self.ax.set_ylim(0, max_dist)

        self.ax.grid(True, linestyle='--', alpha=0.3)

class PhasePlotter1(GifPlotter):
    def __init__(self, ax=None, max_dist=100, max_turns=100):
        super().__init__()
        self.ax = ax if ax else plt.figure().add_subplot(111)
        self.max_dist = max_dist
        self.max_turns = max_turns


    def plot(self, analyzer):
        self.ax.clear()

        # 1. Plot individual chain traces (The "Raw" Data)
        for i in range(analyzer.shape[2]):
            x = analyzer.torsion[:, i] / (2 * np.pi)
            y = analyzer.dist[0, :, i]
            self.ax.plot(x, y, lw=0.5, color='gray', alpha=0.3)

        # 2. Calculate Ensemble Stats
        (mx, my), cov = get_ensemble_stats(analyzer)

        # 3. Display the spread (The "Statistics")
        # 2-Sigma (95% confidence) - Faint
        self.plot_ellipse((mx, my), cov, n_std=2.0, facecolor='tab:red', alpha=0.1)
        # 1-Sigma (68% confidence) - Stronger
        self.plot_ellipse((mx, my), cov, n_std=1.0, facecolor='tab:red', alpha=0.3)

        # 4. Display the Centroid
        self.ax.scatter(mx, my, color='red', s=50, edgecolors='white', zorder=5, label="Ensemble Mean")

        # Formatting
        self.ax.set_xlim(-self.max_turns, self.max_turns)
        self.ax.set_ylim(0, self.max_dist)
        self.ax.set_xlabel("Net Torsion (Turns)")
        self.ax.set_ylabel("Displacement")
        self.ax.grid(True, linestyle=':', alpha=0.6)


class WindowPlotter(GifPlotter):
    """Updated Dashboard with Phase Plotter integration."""

    def __init__(self):
        super().__init__()
        self.fig = plt.figure(figsize=(19.20, 10.80))
        # Adjusting GridSpec to 3x3 or reallocating space
        gs = gridspec.GridSpec(2, 2, height_ratios=[3, 2], width_ratios=[1, 2])

        self.glob = GlobPlotter(self.fig.add_subplot(gs[:, 0], projection='3d'))
        self.hist = HistogramPlotter(self.fig.add_subplot(gs[1, 1], projection="polar"))
        self.phase = PhasePlotter(self.fig.add_subplot(gs[0, 1]))
        #self.graphs = GraphPlotter(self.fig.add_subplot(gs[1, 1]))


    def plot(self, analyzer):
        self.glob.plot(analyzer)
        self.hist.plot(analyzer)
        self.phase.plot(analyzer)  # Execute new plot
        #self.graphs.plot(analyzer)
        self.fig.tight_layout()
