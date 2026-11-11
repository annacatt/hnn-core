"""Compare the C++ VIP model (VIPwithT.cpp) with its NEURON version (vip_cell.py).

Runs every simulation listed in SIMS below. Each one sets the applied current
Iapp (uA/cm2, constant from t = 0) and the D-current, M-current and T-current
strengths (gd, gm in mS/cm2; Pca in the C++ units), the same in both models.
`python compare_cpp_neuron.py 8.5` instead runs one simulation at that Iapp
with the default channels (gd = 3, gm = 1, Pca = 80).

For each simulation, with <TAG> = Iapp<x>_gd<x>_gm<x>_pca<x>:
1. Compiles and runs VIPwithT.cpp for one 5 s sweep and saves the voltage
   trace to sim_outputs/vip_cpp_<TAG>.csv. VIPwithT.cpp is not modified: a
   copy with these parameters and a single sweep is built in cpp_build/.
2. Runs the NEURON cell with the same parameters and protocol (constant Iapp
   from t = 0, V(0) = -63 mV) and saves sim_outputs/vip_neuron_<TAG>.csv,
   sampled every 0.05 ms like the C++ output.
3. Compares spike and burst features (table printed and saved to
   sim_outputs/comparison_summary_<TAG>.csv) and plots the two traces
   (figures/compare_cpp_neuron/compare_cpp_neuron_<TAG>.png).

Needs a C++ compiler for step 1: on macOS, `xcode-select --install`.
Saved outputs are reused on later runs; set RERUN_CPP / RERUN_NEURON to
True to regenerate them.

Split into cells (# %%) for step-by-step runs in PyCharm, VS Code or Jupyter.
"""

# %% imports, paths and settings
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    RT_DIR = Path(__file__).resolve().parent
except NameError:  # running interactively
    RT_DIR = Path.cwd()
sys.path.insert(0, str(RT_DIR))

CPP_SRC = RT_DIR / 'VIPwithT.cpp'
BUILD_DIR = RT_DIR / 'cpp_build'
OUT_DIR = RT_DIR / 'sim_outputs'
FIG_DIR = RT_DIR / 'figures' / 'compare_cpp_neuron'
for d in (BUILD_DIR, OUT_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Simulations to run, one dict each; the same values go into both models.
#   iapp: applied current density (uA/cm2), constant from t = 0
#   gd:   D-current conductance (mS/cm2)    VIPwithT.cpp default: 3
#   gm:   M-current conductance (mS/cm2)    VIPwithT.cpp default: 1
#   pca:  T-current permeability Pca        VIPwithT.cpp default: 80
SIMS = [
    {'iapp': 5., 'gd': 3., 'gm': 0., 'pca': 80.},
    {'iapp': 5., 'gd': 3., 'gm': 0., 'pca': 20.},
    {'iapp': 5., 'gd': 3., 'gm': 0., 'pca': 0.},
]
if len(sys.argv) > 1:          # optional: one Iapp, default channels
    try:
        SIMS = [{'iapp': float(sys.argv[1]), 'gd': 3., 'gm': 1., 'pca': 80.}]
    except ValueError:         # e.g. Jupyter's own arguments
        pass

VIP_AREA = 0.00029  # VIParea in VIPwithT.cpp (multiplies Pca)


def tag(sim):
    """Label used in file names, e.g. 'Iapp5_gd3_gm0_pca80'."""
    return (f'Iapp{sim["iapp"]:g}_gd{sim["gd"]:g}_gm{sim["gm"]:g}'
            f'_pca{sim["pca"]:g}')


def label(sim):
    """Label used in figure titles."""
    return (f'Iapp = {sim["iapp"]:g} uA/cm2, gd = {sim["gd"]:g}, '
            f'gm = {sim["gm"]:g}, Pca = {sim["pca"]:g}')


RERUN_CPP = False
RERUN_NEURON = False

TSTOP = 5000.       # ms, END in VIPwithT.cpp
CPP_DT = 0.05       # ms, TIMESTEP in VIPwithT.cpp (RK4)
BURST_GAP = 100.    # ms, ISIs longer than this separate bursts

# NEURON time step for this comparison. NEURON's fixed-step implicit Euler is
# first order, so spikes per burst converge on the C++ (RK4) as dt shrinks:
# 0.025 ms -> 23/burst, 0.01 -> 25, 0.005 -> 26, 0.0025 -> 27 (C++: 27-28).
# 0.0025 ms takes ~10-20 s for 5 s of simulation.
NEURON_DT = 0.0025


# %% helpers: run and save the C++ model

def run_cpp(sim, src=CPP_SRC, build_dir=BUILD_DIR):
    """Build a one-sweep copy of VIPwithT.cpp with the parameters in sim
    (iapp, gd, gm, pca), run it, return t, V."""
    code = src.read_text()
    # Each parameter is set by one line of VIPwithT.cpp; replace its number.
    # Iapp, Pca and gm are (re)set inside the cell loop of Derive(), which
    # overrides the values declared above it ("Iapp[z] = 9.8;", "Pca = 80;",
    # "gm = 1;"); gd is set once, where it is declared ("double gd = 3;").
    num = r'[-+0-9.eE]+'
    for name, pattern, value in (
            ('Iapp', rf'(Iapp\[z\]\s*=\s*){num}(\s*;)', sim['iapp']),
            ('Pca', rf'(^\s*Pca\s*=\s*){num}(\s*;)', sim['pca']),
            ('gm', rf'(^\s*gm\s*=\s*){num}(\s*;)', sim['gm']),
            ('gd', rf'(double\s+gd\s*=\s*){num}(\s*;)', sim['gd'])):
        code, n = re.subn(pattern, rf'\g<1>{float(value)!r}\g<2>', code,
                          flags=re.MULTILINE)
        if n != 1:
            raise RuntimeError(f'expected one line setting {name} in '
                               f'{src.name}, found {n}')
    # The C++ repeats the 5 s simulation for Ifreq = 5, 10, ..., 80 Hz, but
    # Ifreq only feeds the (commented-out) sinusoidal input, so every sweep
    # is the same run continued. Keep just the first one.
    loop = 'while (Ifreq <= 80)'
    if loop not in code:
        raise RuntimeError(f'could not find "{loop}" in {src.name}; '
                           'has the sweep loop changed?')
    one_sweep = build_dir / 'VIPwithT_one_sweep.cpp'
    one_sweep.write_text(code.replace(loop, 'while (Ifreq <= 5)'))

    compiler = next((c for c in ('c++', 'clang++', 'g++') if shutil.which(c)),
                    None)
    if compiler is None:
        raise RuntimeError('no C++ compiler found (macOS: xcode-select '
                           '--install)')
    exe = build_dir / 'vip_one_sweep'
    subprocess.run([compiler, '-O2', '-o', str(exe), str(one_sweep)],
                   check=True)
    subprocess.run([str(exe)], cwd=build_dir, check=True)  # writes results*.txt

    data = np.loadtxt(build_dir / 'results.txt', delimiter=',')
    # Each row prints the time *before* the RK4 step next to the voltage
    # *after* it, so the voltage belongs to t + dt.
    return data[:, 0] + CPP_DT, data[:, 1]


def save_trace(path, t, v, comment):
    np.savetxt(path, np.column_stack([t, v]), delimiter=',', fmt='%.4f',
               header=f'{comment}\nt_ms,V_mV')


def load_trace(path):
    d = np.loadtxt(path, delimiter=',')
    return d[:, 0], d[:, 1]


# %% helpers: run (or load) one simulation in each model

def get_cpp(sim):
    path = OUT_DIR / f'vip_cpp_{tag(sim)}.csv'
    if RERUN_CPP or not path.exists():
        t, v = run_cpp(sim)
        save_trace(path, t, v, f'VIPwithT.cpp, one 5 s sweep, {label(sim)}, '
                   'V(0) = -63 mV, RK4 dt = 0.05 ms')
        print(f'C++ run saved to {path}')
    else:
        t, v = load_trace(path)
        print(f'C++ run loaded from {path}')
    return t, v


def get_neuron(sim):
    path = OUT_DIR / f'vip_neuron_{tag(sim)}.csv'
    if RERUN_NEURON or not path.exists():
        import copy
        from vip_cell import (delete_cell, density_to_nA, get_vip_cell,
                              run_current_clamp, vip_params)
        params = copy.deepcopy(vip_params)
        soma_params = params['soma']
        soma_params['kd_golomb2007'] = sim['gd'] * 1e-3      # mS -> S/cm2
        soma_params['km_mccarthy2023'] = sim['gm'] * 1e-3    # mS -> S/cm2
        # Pca * VIParea, then uA/cm2 -> mA/cm2 (as in vip_cell.py)
        soma_params['cat_mccarthy2023'] = sim['pca'] * VIP_AREA * 1e-3
        cell = get_vip_cell(params)
        soma = cell._nrn_sections['soma']
        t, v = run_current_clamp(cell, density_to_nA(soma, sim['iapp']),
                                 delay=0, dur=1e9, tstop=TSTOP,
                                 dt=NEURON_DT, record_dt=CPP_DT)
        delete_cell()
        save_trace(path, t, v, f'vip_cell.py, {label(sim)}, V(0) = -63 mV, '
                   f'dt = {NEURON_DT} ms, sampled every {CPP_DT} ms')
        print(f'NEURON run saved to {path}')
    else:
        t, v = load_trace(path)
        print(f'NEURON run loaded from {path}')
    return t, v


# %% helpers: features

def spike_times(t, v, thresh=0.):
    idx = np.where((v[:-1] <= thresh) & (v[1:] > thresh))[0] + 1
    return t[idx]


def split_bursts(spk, gap=BURST_GAP):
    """List of spike-time arrays, one per burst."""
    if len(spk) == 0:
        return []
    breaks = np.where(np.diff(spk) > gap)[0] + 1
    return np.split(spk, breaks)


def features(t, v):
    spk = spike_times(t, v)
    bursts = split_bursts(spk)
    peaks, troughs = [], []
    for s in spk:   # AP peak within 1 ms of the threshold crossing
        w = (t >= s) & (t <= s + 1)
        peaks.append(v[w].max())
    for b in bursts:   # AHP trough between consecutive spikes of a burst
        for s0, s1 in zip(b[:-1], b[1:]):
            w = (t > s0) & (t < s1)
            troughs.append(v[w].min())
    # depolarised plateau between bursts, skipping 100 ms after each burst
    plateau = []
    for b0, b1 in zip(bursts[:-1], bursts[1:]):
        w = (t > b0[-1] + 100) & (t < b1[0] - 50)
        plateau.append(np.median(v[w]))

    def stat(x):
        return np.mean(x) if len(x) else np.nan

    feat = {
        'n_spikes': len(spk),
        'n_bursts': len(bursts),
        'spikes_per_burst': stat([len(b) for b in bursts]),
        'first_spike_ms': spk[0] if len(spk) else np.nan,
        'burst_period_ms': stat(np.diff([b[0] for b in bursts])),
        'burst_duration_ms': stat([b[-1] - b[0] for b in bursts]),
        'first_isi_in_burst_ms': stat([b[1] - b[0] for b in bursts
                                       if len(b) > 1]),
        'last_isi_in_burst_ms': stat([b[-1] - b[-2] for b in bursts
                                      if len(b) > 1]),
        'ap_peak_mV': stat(peaks),
        'ahp_trough_mV': stat(troughs),
        'interburst_plateau_mV': stat(plateau),
    }
    for i, b in enumerate(bursts):
        feat[f'burst{i + 1}_onset_ms'] = b[0]
    return feat, spk, bursts



# %% helpers: compare and plot one simulation

def compare_features(sim, t_cpp, v_cpp, t_nrn, v_nrn):
    """Print and save the feature table; return the spikes and bursts."""
    f_cpp, spk_cpp, bursts_cpp = features(t_cpp, v_cpp)
    f_nrn, spk_nrn, bursts_nrn = features(t_nrn, v_nrn)

    rows = []
    print(f'\n{"feature":<24}{"C++":>10}{"NEURON":>10}{"diff":>10}{"diff %":>9}')
    for k in sorted(set(f_cpp) | set(f_nrn), key=list({**f_cpp, **f_nrn}).index):
        a, b = f_cpp.get(k, np.nan), f_nrn.get(k, np.nan)
        pct = 100 * (b - a) / abs(a) if np.isfinite(a) and a != 0 else np.nan
        rows.append((k, a, b, b - a, pct))
        print(f'{k:<24}{a:>10.2f}{b:>10.2f}{b - a:>10.2f}{pct:>8.1f}%')

    with open(OUT_DIR / f'comparison_summary_{tag(sim)}.csv', 'w') as fh:
        fh.write('feature,cpp,neuron,diff,diff_percent\n')
        for r in rows:
            fh.write(','.join([r[0]] + [f'{x:.4f}' for x in r[1:]]) + '\n')

    return spk_cpp, bursts_cpp, spk_nrn, bursts_nrn


def plot_comparison(sim, t_cpp, v_cpp, spk_cpp, bursts_cpp,
                    t_nrn, v_nrn, spk_nrn, bursts_nrn):
    c_cpp, c_nrn = 'k', 'tab:red'
    fig, axs = plt.subplot_mosaic([['full', 'full'], ['onset', 'spike'],
                                   ['onset', 'isi']],
                                  figsize=(10, 7), constrained_layout=True)

    ax = axs['full']
    ax.plot(t_cpp, v_cpp, c_cpp, lw=0.5, label='C++ (RK4, dt 0.05 ms)')
    ax.plot(t_nrn, v_nrn, c_nrn, lw=0.5, alpha=0.7, label=f'NEURON (dt {NEURON_DT} ms)')
    ax.set(xlabel='time (ms)', ylabel='V (mV)',
           title=f'VIP cell, {label(sim)}')
    ax.legend(loc='upper right', fontsize=8)

    # second burst, each trace aligned to its own burst onset
    ax = axs['onset']
    for t, v, bursts, c in ((t_cpp, v_cpp, bursts_cpp, c_cpp),
                            (t_nrn, v_nrn, bursts_nrn, c_nrn)):
        if len(bursts) > 1:
            t0 = bursts[1][0]
            w = (t >= t0 - 100) & (t <= t0 + 150)
            ax.plot(t[w] - t0, v[w], c, lw=0.8)
    ax.set(xlabel='time from burst 2 onset (ms)', ylabel='V (mV)',
           title='burst onset (aligned)')

    # average spike waveform, aligned to the threshold crossing
    ax = axs['spike']
    for t, v, spk, c in ((t_cpp, v_cpp, spk_cpp, c_cpp),
                         (t_nrn, v_nrn, spk_nrn, c_nrn)):
        dt_s = t[1] - t[0]
        n_pre, n_post = int(2 / dt_s), int(6 / dt_s)
        idx = np.searchsorted(t, spk)
        idx = idx[(idx > n_pre) & (idx < len(v) - n_post)]
        if len(idx) == 0:
            continue
        waves = np.array([v[i - n_pre:i + n_post] for i in idx])
        ax.plot(np.arange(-n_pre, n_post) * dt_s, waves.mean(0), c, lw=1.2)
    ax.set(xlabel='time from 0 mV crossing (ms)', ylabel='V (mV)',
           title='mean spike')

    # ISI along each burst
    ax = axs['isi']
    for bursts, c, m in ((bursts_cpp, c_cpp, 'o'), (bursts_nrn, c_nrn, 's')):
        for b in bursts:
            ax.plot(np.arange(1, len(b)), np.diff(b), c, marker=m, ms=3, lw=0.6,
                    alpha=0.8)
    ax.set(xlabel='spike number in burst', ylabel='ISI (ms)',
           title='adaptation within bursts')

    fig.savefig(FIG_DIR / f'compare_cpp_neuron_{tag(sim)}.png', dpi=150)


# %% run every simulation in SIMS
for sim in SIMS:
    print(f'\n=== {label(sim)} ===')
    t_cpp, v_cpp = get_cpp(sim)
    t_nrn, v_nrn = get_neuron(sim)
    spikes = compare_features(sim, t_cpp, v_cpp, t_nrn, v_nrn)
    plot_comparison(sim, t_cpp, v_cpp, spikes[0], spikes[1],
                    t_nrn, v_nrn, spikes[2], spikes[3])
plt.show()

