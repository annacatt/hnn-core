"""Plot several VIP C++ runs (one row each) and print simple features.

Each run is a folder inside cpp_runs/ containing the results.txt written by
the C++ program (e.g. VIP_one_sweep.cpp).

Usage (from any folder):
    python compare_cpp_runs.py one_sweep_iapp9.8 one_sweep_iapp9.8_gd2
In VS Code you can also edit RUNS below and press Run.
The figure is saved in cpp_runs/ as FIG_NAME (set SAVE_FIG = False
to only show it).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS = ['one_sweep_iapp5_gd3_only', 'one_sweep_iapp5_gd3_gm0_gt80', 'one_sweep_iapp5_gd3_gm0_gt40', 'one_sweep_iapp5_gd3_gm0_gt20']#, 'one_sweep_iapp8_gd3_gm1_gt80', 'one_sweep_iapp9.8_gd3_gm1_gt80', 'one_sweep_iapp10_gd3_gm1_gt80']   # folders in cpp_runs/
if len(sys.argv) > 1:
    RUNS = sys.argv[1:]
XLIM = None              # e.g. (0, 500) to zoom in on the first 500 ms
SAVE_FIG = True          # False: only show the figure
FIG_NAME = 'compare_runs.png'   # saved in cpp_runs/; .png, .pdf or .svg

try:
    CPP_RUNS = Path(__file__).resolve().parent
except NameError:        # running interactively
    CPP_RUNS = Path.cwd()


def load_results(path):
    """Return t, V from a results.txt (pandas if available: faster)."""
    try:
        import pandas as pd
        d = pd.read_csv(path, header=None, usecols=[0, 1]).to_numpy()
    except ImportError:
        d = np.loadtxt(path, delimiter=',', usecols=(0, 1))
    return d[:, 0], d[:, 1]


runs = []
for run in RUNS:
    path = CPP_RUNS / run / 'results.txt'
    if path.exists():
        runs.append((run, path))
    else:
        print(f'{run}: no results.txt in {path.parent}, skipped')
if not runs:
    sys.exit('Nothing to plot: check the folder names in RUNS.')

# one row per run, same time and voltage axes in every row
fig, axs = plt.subplots(len(runs), 1, figsize=(11, 1.8 * len(runs) + 0.6),
                        sharex=True, sharey=True, squeeze=False,
                        constrained_layout=True)
axs = axs[:, 0]
for ax, (run, path) in zip(axs, runs):
    t, v = load_results(path)
    up = (v[1:] > 0) & (v[:-1] <= 0)          # upward crossings of 0 mV
    spk = t[1:][up]
    first = f'{spk[0]:.1f} ms' if len(spk) else 'none'
    rate = len(spk) / (t[-1] - t[0]) * 1e3
    print(f'{run}: {len(spk)} spikes ({rate:.1f} Hz mean), '
          f'first spike {first}, V at end {v[-1]:.2f} mV')
    ax.plot(t, v, 'k', lw=0.5)
    ax.set_ylabel('V (mV)')
    ax.set_title(run, fontsize=9, loc='left')

axs[-1].set_xlabel('time (ms)')
fig.suptitle('VIP C++ runs')
if XLIM is not None:
    axs[-1].set_xlim(*XLIM)
if SAVE_FIG:
    fig_path = CPP_RUNS / FIG_NAME
    fig.savefig(fig_path, dpi=150)
    print(f'figure saved to {fig_path}')
plt.show()
