import os
import time
import numpy as np
import matplotlib.pyplot as plt

from lib import Seed, TorsionMCMC, VonMisesSampler, PolymerAnalyzer
from lib import GlobPlotter, PhasePlotter, WindowPlotter, GraphPlotter, HistogramPlotter

# Simulation Constants
N_CHAINS = 8
CHAIN_LENGTH = 128
Seed().value = 1738

# Weight parameters (T, Gp, Gn)
batch = [(1, 0, 0), (0, 0, 1)]


class Timer:
    def __init__(self):
        self.start_time = time.perf_counter()
        self.last_lap = time.perf_counter()

    def lap(self):
        now = time.perf_counter()
        delta = now - self.last_lap
        self.last_lap = now
        return delta

    @property
    def total_elapsed(self):
        return time.perf_counter() - self.start_time


def main(blocksize=50):
    sampler = VonMisesSampler(CHAIN_LENGTH, N_CHAINS)
    ps_values = np.linspace(5.0, 9.0, 201)
    kappas = [10 ** (7 - pS) for pS in ps_values]


    # Initialize GlobPlotter. It will create its own 3D figure internally.
    figs = [plt.figure(i) for i in range(3)]
    glob_window = GlobPlotter(figs[0])
    glob_window.set_shape(6, 6)

    phase_window = PhasePlotter(figs[1])
    phase_window.set_shape(8, 6)

    hist_window = HistogramPlotter(figs[2])
    hist_window.set_shape(6, 6)

    windows = [glob_window, phase_window, hist_window]

    for T, Gp, Gn in batch:
        clock = Timer()
        gif_root = f"./local/outputs/gifs"
        os.makedirs(gif_root, exist_ok=True)
        print(f"🔄 Preparing Data for W=({T},{Gp},{Gn})... ", end="", flush=True)
        ps_values = np.linspace(5.0, 9.0, 101)
        kappas = [10 ** (7 - pS) for pS in ps_values]

        all_samples = sampler(weights=(T, Gp, Gn), kappas=kappas)
        markov = TorsionMCMC(CHAIN_LENGTH, N_CHAINS, n_batches=len(kappas))
        markov.run(all_samples)

        print(f"Done! ({clock.lap():.2f}s)")

        # Frame iteration
        try:
            print(f"🎬 Generating Animation Frames...")
            for i, pS in enumerate(ps_values):
                chains_i = markov.chains[..., i]
                sample_i = all_samples[..., [i]]
                k_i = kappas[i]

                analyzer = PolymerAnalyzer(sample_i, chains_i)

                # Update the plot
                for window in windows:
                    run_string = f"W=({T},{Gp},{Gn}), κ={k_i:.2e}, pS={pS:.2f}"

                    window.plot(analyzer)
                    window.fig.suptitle(run_string, x=0.05, ha="left")
                    window.capture_frame(window.fig, dpi=256)

                frame_time = clock.lap()
                total = clock.total_elapsed

                if i % blocksize == 0 and i != 0:
                    print(f"\r\t✅ Block {i // blocksize}: {blocksize} frames in {total:.2f}s total.")

                print(f"\r\t↳ [Frame {i + 1}/{len(ps_values)}] Current κ: {k_i:.2e} | "
                      f"Last: {frame_time * 1000:.1f}ms | Total: {total:.2f}s", end="")

        finally:
            print(f"\n💾 Saving GIFs for W=({T},{Gp},{Gn})... ")
            gif_path = os.path.join(gif_root, f"W({T},{Gp},{Gn})")
            os.makedirs(gif_path, exist_ok=True)
            for window, name in zip(windows, ["glob", "phase", "hist"]):
                print("\t", end="")
                gif_name = f"{name}_W({T},{Gp},{Gn})_{CHAIN_LENGTH}x{N_CHAINS}.gif"
                output_file = os.path.join(gif_path, gif_name)
                window.save_gif(output_file, duration=20, loop=0)

            print(f"Saved! Total batch time: {clock.total_elapsed:.2f}s\n" + "-" * 50)


if __name__ == "__main__":
    main()