"""Current-step and chirp recordings of human L2/3 VIP interneurons (Lee, Dalley 2023).

Plots the recordings extracted from the DANDI 000636 NWB files by
vip_nwb_extract.py and computes a few simple features from them, next to the
values the authors report in LeeDalley_ephys_fx.csv, so the extraction can be
checked.

1 s current steps (data/lee_dalley/nwb_steps/<specimen_id>.npz):
    baseline V, sag ratio (step closest to -100 pA), rebound spikes after
    hyperpolarising steps, rheobase, first-spike latency and firing rate at
    rheobase, and the f-I curve.
Chirps (data/lee_dalley/nwb_chirp/<specimen_id>.npz), when the cell has them:
a sinusoidal current whose frequency rises during the sweep. The voltage
response is plotted against the chirp's instantaneous frequency (measured
from the injected current of each sweep), so that sweeps with different
chirps line up: older protocols rotate a 40 s, a 20 s and a 10 s chirp, each
covering a different frequency range. From each sweep the script computes the
impedance profile |Z(f)| = |V(f)| / |I(f)| over the frequencies its chirp
covers, and averages the sweeps in log-spaced frequency bins. Features: low-frequency impedance (mean |Z| up to LOW_F), peak
impedance, peak (resonance) frequency, peak ratio (peak / low-frequency
impedance) and the 3 dB frequency (highest frequency where |Z| >= peak /
sqrt(2)). A peak ratio above 1 points to currents that oppose slow voltage
changes, such as the h- and M-currents; the T-current can also add to it.
The impedance assumes a subthreshold response, so chirp sweeps with spikes
(V above CHIRP_SPIKE_THRESH) are plotted in grey and left out of it.

For each cell: figures/vip_ephys_traces/cell_<specimen_id>.png (steps on the
top row, chirp response and impedance below when the cell has chirps)
If several cells are available: figures/vip_ephys_traces/fI_all_cells.png
and figures/vip_ephys_traces/impedance_all_cells.png
Table of features: data/lee_dalley/results/vip_L23_step_features.csv
(chirp features in the columns starting with chirp_)

Only numpy, pandas and matplotlib are needed. Run it from anywhere, or cell by
cell (# %%) in VS Code / PyCharm / Jupyter.

Feature definitions here are simple and not identical to the IPFX ones used by
the authors (e.g. IPFX's rheobase comes from a finer search, its peak and
steady-state voltages are measured slightly differently, and it smooths |Z|
with its own frequency bands); small differences from the table are expected.
"""

# %% imports, paths, settings
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    RT_DIR = Path(__file__).resolve().parent
except NameError:  # running interactively
    RT_DIR = Path.cwd()

DATA_DIR = RT_DIR / 'data' / 'lee_dalley'
STEP_DIR = DATA_DIR / 'nwb_steps'
CHIRP_DIR = DATA_DIR / 'nwb_chirp'
OUT_DIR = DATA_DIR / 'results'
FIG_DIR = RT_DIR / 'figures' / 'vip_ephys_traces'
for d in (OUT_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

CELLS = None             # None: every extracted cell; or a list of specimen ids
SPIKE_THRESH = 0.        # mV, upward crossing counted as a spike
SAG_TARGET = -100.       # pA: sag is measured on the step closest to this
                         # (as in IPFX; ties go to the smaller step)
REBOUND_WIN = 0.5        # s after step offset in which rebound spikes are counted
STEP_DUR = 1.0           # s
CHIRP_SPIKE_THRESH = -30.  # mV: chirp sweeps crossing this are left out of |Z|
BAND_FRAC = 0.1          # chirp band: frequencies where |I(f)| > this x max
N_BINS = 40              # log-spaced frequency bins for smoothing |Z|
LOW_F = 1.0              # Hz: low-frequency impedance is the mean up to here

BLUE_RAMP = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281']
INK, GRID, GREY = '#1f1f1d', '#e2e1dc', '#9a9993'

files = sorted(STEP_DIR.glob('*.npz'))
if CELLS is not None:
    files = [f for f in files if int(f.stem) in set(CELLS)]
if not files:
    sys.exit(f'No extracted cells in {STEP_DIR}: run vip_nwb_extract.py '
             'first (in an environment with h5py).')

fx = pd.read_csv(DATA_DIR / 'LeeDalley_ephys_fx.csv')
fx['specimen_id'] = fx['specimen_id'].astype('int64')
fx = fx.set_index('specimen_id')


# %% chirp helpers: instantaneous frequency, impedance, features
def inst_freq(t, i):
    """Instantaneous frequency of the chirp (Hz) against time (s), from the
    upward zero crossings of the injected current (and the chirp onset, when
    the chirp starts by rising)."""
    ok = np.isfinite(i)
    t, i = t[ok], i[ok] - i[ok & (t < 0)].mean()
    up = np.flatnonzero((i[1:] > 0) & (i[:-1] <= 0) & (t[1:] >= 0))
    tc = t[up + 1]
    first = i[(t >= 0) & (np.abs(i) > 0.5)]
    if len(first) and first[0] > 0 and (len(tc) == 0 or tc[0] > 0.05):
        tc = np.r_[0., tc]
    if len(tc) < 3:
        return None
    return (tc[:-1] + tc[1:]) / 2, np.maximum.accumulate(1 / np.diff(tc))


def time_to_freq(t, i):
    """Function mapping time (s) to the chirp's instantaneous frequency (Hz),
    interpolated in log(frequency) and extended linearly past both ends."""
    fr = inst_freq(t, i)
    if fr is None:
        return None
    tm, fi = fr
    keep = np.r_[True, np.diff(fi) > 0]                # strictly increasing
    tm, lf = tm[keep], np.log(fi[keep])
    s0 = (lf[1] - lf[0]) / (tm[1] - tm[0])
    s1 = (lf[-1] - lf[-2]) / (tm[-1] - tm[-2])
    tm = np.r_[tm[0] - 1e3, tm, tm[-1] + 1e3]
    lf = np.r_[lf[0] - 1e3 * s0, lf, lf[-1] + 1e3 * s1]
    return lambda x: np.exp(np.interp(x, tm, lf))


def sweep_impedance(t, v, i, dur):
    """|Z| (MOhm) against frequency (Hz) for one sweep, over the frequencies
    its chirp covers (from inst_freq): outside them the current has almost
    no power, and |Z| is mostly noise and slow drift."""
    ok = np.isfinite(v) & np.isfinite(i)
    t, v, i = t[ok], v[ok], i[ok]
    fr = inst_freq(t, i)
    if fr is None:
        return None
    pre, win = t < 0, (t >= 0) & (t < dur)
    dv, di = v[win] - v[pre].mean(), i[win] - i[pre].mean()
    f = np.fft.rfftfreq(len(dv), t[1] - t[0])
    V, I = np.fft.rfft(dv), np.fft.rfft(di)
    band = (np.abs(I) > BAND_FRAC * np.abs(I[1:]).max()) & (f > 0) \
        & (f >= fr[1].min()) & (f <= fr[1].max())
    return f[band], np.abs(V[band] / I[band]) * 1e3   # mV/pA = GOhm -> MOhm


def cell_impedance(t, V, I, durs):
    """Each sweep's |Z| averaged in common log-spaced bins (one row per
    sweep, NaN where its chirp does not reach), the bin centres, and the mean
    over the sweeps covering each bin."""
    per = [sweep_impedance(t, v, i, d) for v, i, d in zip(V, I, durs)]
    per = [x for x in per if x is not None and len(x[0])]
    if not per:
        return None
    edges = np.geomspace(min(f.min() for f, _ in per),
                         max(f.max() for f, _ in per), N_BINS + 1)
    fc = np.sqrt(edges[:-1] * edges[1:])
    Z = np.full((len(per), N_BINS), np.nan)
    for n, (f, z) in enumerate(per):
        idx = np.clip(np.digitize(f, edges) - 1, 0, N_BINS - 1)
        for k in np.unique(idx):
            Z[n, k] = z[idx == k].mean()
    with warnings.catch_warnings():          # bins no sweep reaches
        warnings.simplefilter('ignore', RuntimeWarning)
        zm = np.nanmean(Z, 0)
    keep = np.isfinite(zm)
    return fc, Z, fc[keep], zm[keep]


def chirp_features(f, z):
    low = z[f <= LOW_F].mean() if (f <= LOW_F).any() else z[0]
    k = int(np.argmax(z))
    above = np.flatnonzero(z >= z[k] / np.sqrt(2))
    return {'chirp_low_freq_impedance': low, 'chirp_peak_impedance': z[k],
            'chirp_peak_freq': f[k], 'chirp_peak_ratio': z[k] / low,
            'chirp_3db_freq': f[above.max()]}


FREQ_TICKS = [0.2, 0.5, 1, 2, 5, 10, 20, 40]     # Hz, labels on log axes


def freq_ticks(ax):
    """Label a log frequency axis with plain numbers (0.5, 1, 2, 5, ...),
    without changing its limits."""
    ax.xaxis.set_major_locator(plt.FixedLocator(FREQ_TICKS))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())


def load_chirp(sid):
    """The cell's chirp sweeps and impedance, or None if it has none."""
    path = CHIRP_DIR / f'{sid}.npz'
    if not path.exists():
        return None
    d = np.load(path)
    c = {k: d[k] for k in ('t', 'v', 'i', 'dur', 'sweep')}
    c['good'] = ~(np.nan_to_num(c['v'], nan=-np.inf)
                  > CHIRP_SPIKE_THRESH).any(1)
    c['features'] = {}
    g = c['good']
    imp = cell_impedance(c['t'], c['v'][g], c['i'][g], c['dur'][g]) \
        if g.any() else None
    if imp is not None:
        c['f_bins'], c['z_sweeps'], c['f'], c['z'] = imp
        c['features'] = chirp_features(c['f'], c['z'])
    return c


# %% feature extraction
def spike_times(t, v):
    up = np.flatnonzero((v[1:] > SPIKE_THRESH) & (v[:-1] <= SPIKE_THRESH))
    return t[up + 1]


def cell_features(z):
    t, V, amps = z['t'], z['v'], z['i_pA']
    base = V[:, t < 0].mean(axis=1)
    in_step = (t >= 0) & (t < STEP_DUR)
    rates, lat = [], []
    for v in V:
        spk = spike_times(t, v)
        spk = spk[(spk >= 0) & (spk < STEP_DUR)]
        rates.append(len(spk) / STEP_DUR)
        lat.append(spk[0] if len(spk) else np.nan)
    rates, lat = np.array(rates), np.array(lat)

    f = {'specimen_id': int(z['specimen_id']),
         'transcriptomic_type': str(z['transcriptomic_type']),
         'condition': str(z['condition']),
         'v_baseline': float(np.median(base))}

    neg = np.flatnonzero(amps < 0)
    if len(neg):
        d = np.abs(amps[neg] - SAG_TARGET)
        k = neg[np.lexsort((np.abs(amps[neg]), d))[0]]   # closest to -100 pA
        v = V[k]
        peak = v[in_step].min()                       # largest deflection
        steady = v[(t >= STEP_DUR - 0.1) & (t < STEP_DUR)].mean()
        f['sag_I_pA'] = float(amps[k])
        f['vm_for_sag'] = float(peak)
        f['sag'] = float((steady - peak) / (base[k] - peak))
        reb = [np.sum((s >= STEP_DUR) & (s < STEP_DUR + REBOUND_WIN))
               for s in (spike_times(t, V[j]) for j in neg)]
        f['rebound_spikes_max'] = int(max(reb))
        f['rebound_sweeps'] = int(np.sum(np.array(reb) > 0))

    pos = np.flatnonzero((amps > 0) & (rates > 0))
    if len(pos):
        k = pos[np.argmin(amps[pos])]
        f['rheobase_i'] = float(amps[k])
        f['latency_rheo_ms'] = float(lat[k] * 1e3)
        f['rate_rheo_Hz'] = float(rates[k])
        f['max_rate_Hz'] = float(rates.max())
    return f, rates


rows, fI, chirps = [], {}, {}
for path in files:
    z = np.load(path)
    f, rates = cell_features(z)
    sid = f['specimen_id']
    c = load_chirp(sid)
    if c is not None:
        chirps[sid] = c
        f['chirp_sweeps'] = len(c['v'])
        f['chirp_sweeps_used'] = int(c['good'].sum())
        f.update(c['features'])
    rows.append(f)
    fI[sid] = (z['i_pA'], rates)
chirp_only = sorted(set(int(p.stem) for p in CHIRP_DIR.glob('*.npz'))
                    - set(fI))
if chirp_only:
    print(f'cells with chirps but no step sweeps (not plotted): {chirp_only}')
feat = pd.DataFrame(rows).set_index('specimen_id')

# authors' IPFX values, for comparison
ref = fx.reindex(feat.index)
compare = pd.DataFrame({
    'v_baseline': feat['v_baseline'], 'v_baseline_ipfx': ref['v_baseline'],
    'sag': feat.get('sag'), 'sag_ipfx': ref['sag'],
    'vm_for_sag': feat.get('vm_for_sag'), 'vm_for_sag_ipfx': ref['vm_for_sag'],
    'rheobase_i': feat.get('rheobase_i'), 'rheobase_i_ipfx': ref['rheobase_i'],
    'latency_rheo_ms': feat.get('latency_rheo_ms'),
    'latency_rheo_ms_ipfx': ref['latency_rheo'] * 1e3,
    'rate_rheo_Hz': feat.get('rate_rheo_Hz'),
    'rate_rheo_Hz_ipfx': ref['avg_rate_rheo'],
})
for k in ('low_freq_impedance', 'peak_impedance', 'peak_freq', 'peak_ratio',
          '3db_freq'):
    if f'chirp_{k}' in feat:
        compare[f'chirp_{k}'] = feat[f'chirp_{k}']
        compare[f'chirp_{k}_ipfx'] = ref.get(f'{k}_chirp')
feat.join(compare.filter(like='_ipfx')).to_csv(
    OUT_DIR / 'vip_L23_step_features.csv')
print(f'{len(feat)} cells\n')
print(compare.round(2).T.to_string())
print('\nrebound spikes after hyperpolarising steps (max per sweep, '
      'number of sweeps with any):')
print(feat[['rebound_spikes_max', 'rebound_sweeps']].to_string())


# %% figure per cell: hyperpolarising steps, depolarising steps, f-I curve;
#    below, when the cell has chirps: chirp response and impedance
def style(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


for path in files:
    z = np.load(path)
    sid = int(z['specimen_id'])
    t_ms, V, amps = z['t'] * 1e3, z['v'], z['i_pA']
    neg, pos = np.flatnonzero(amps < 0), np.flatnonzero(amps > 0)

    ch = chirps.get(sid)
    layout = [['hyp', 'dep', 'fI']] + ([['chirp', 'chirp', 'Z']] if ch else [])
    fig, ax_d = plt.subplot_mosaic(
        layout, figsize=(12, 3.6 * len(layout)), constrained_layout=True,
        gridspec_kw={'width_ratios': [1.2, 1.2, 0.8]})
    axs = [ax_d['hyp'], ax_d['dep'], ax_d['fI']]
    # hyperpolarising family, light -> dark with step size
    for n, k in enumerate(neg[np.argsort(-amps[neg])]):
        c = BLUE_RAMP[min(n, len(BLUE_RAMP) - 1)]
        axs[0].plot(t_ms, V[k], color=c, lw=0.7, label=f'{amps[k]:g}')
    axs[0].set(xlabel='time from step onset (ms)', ylabel='V (mV)',
               title='hyperpolarising steps (pA)')
    axs[0].legend(fontsize=7, frameon=False, loc='lower right', ncol=2)

    # depolarising steps: rheobase, middle and strongest, offset vertically
    if len(pos):
        spiking = [k for k in pos if feat.loc[sid].get('rheobase_i', np.nan)
                   <= amps[k]]
        pick = pos if len(pos) <= 3 else \
            [pos[0]] + ([spiking[0]] if spiking else []) + [pos[-1]]
        pick = list(dict.fromkeys(pick))[:3]
        for n, k in enumerate(pick):
            axs[1].plot(t_ms, V[k] + 120 * n, color=INK, lw=0.5)
            axs[1].text(t_ms[-1], V[k][-1] + 120 * n + 8, f'{amps[k]:g} pA',
                        fontsize=7, ha='right', color='#55544f')
        axs[1].set(xlabel='time from step onset (ms)',
                   ylabel='V (mV, offset 120 mV)', yticks=[],
                   title='depolarising steps')

    # f-I curve
    I, r = fI[sid]
    axs[2].plot(I[I >= 0], r[I >= 0], 'o-', color=BLUE_RAMP[2], ms=5)
    axs[2].set(xlabel='step (pA)', ylabel='spikes / s in step', title='f-I')

    if ch:
        # chirp response against the chirp's instantaneous frequency, each
        # sweep mapped with its own current (sweeps can have different chirps)
        ax = ax_d['chirp']
        for n in range(len(ch['v'])):
            col = BLUE_RAMP[min(n + 1, len(BLUE_RAMP) - 1)] if ch['good'][n] \
                else GREY
            fmap = time_to_freq(ch['t'], ch['i'][n])
            w = (ch['t'] >= 0) & (ch['t'] < ch['dur'][n]) \
                & np.isfinite(ch['v'][n])
            if fmap is None or not w.any():
                continue
            ax.plot(fmap(ch['t'][w]), ch['v'][n][w], color=col, lw=0.4,
                    label=f'sweep {ch["sweep"][n]} ({ch["dur"][n]:.0f} s)'
                    + ('' if ch['good'][n] else ' spikes, not in |Z|'))
        ax.set_xscale('log')
        freq_ticks(ax)
        ax.set(xlabel='chirp frequency (Hz)', ylabel='V (mV)',
               title='chirp response')
        ax.legend(fontsize=7, frameon=False, loc='lower left',
                  ncol=min(3, len(ch['v'])))
        axs.append(ax)

        # impedance profile: each subthreshold sweep thin, their mean thick
        ax = ax_d['Z']
        if ch['features']:
            for zn in ch['z_sweeps']:
                ax.plot(ch['f_bins'], zn, color=BLUE_RAMP[1], lw=0.6,
                        alpha=0.6)
            cf = ch['features']
            ax.plot(ch['f'], ch['z'], color=BLUE_RAMP[4], lw=1.8,
                    label='mean of sweeps')
            ax.plot(cf['chirp_peak_freq'], cf['chirp_peak_impedance'], 'o',
                    color=BLUE_RAMP[4], ms=6)
            ax.axvline(cf['chirp_peak_freq'], color=GREY, lw=0.8, ls='--')
            ax.text(0.97, 0.97, f'peak {cf["chirp_peak_freq"]:.1f} Hz\n'
                    f'peak ratio {cf["chirp_peak_ratio"]:.2f}\n'
                    f'3 dB {cf["chirp_3db_freq"]:.1f} Hz',
                    transform=ax.transAxes, ha='right', va='top', fontsize=8)
            ax.set_xscale('log')
            ax.legend(fontsize=7, frameon=False, loc='lower left')
            ax.sharex(ax_d['chirp'])
            freq_ticks(ax)
        else:
            ax.text(0.5, 0.5, 'every chirp sweep has spikes:\nno impedance',
                    transform=ax.transAxes, ha='center', va='center',
                    color=GREY)
        ax.set(xlabel='frequency (Hz)', ylabel='|Z| (MOhm)', title='impedance')
        axs.append(ax)

    for ax in axs:
        style(ax)
        title = ax.get_title()
        ax.set_title('')
        ax.set_title(title, loc='left', fontsize=9)
    fig.suptitle(f'{sid}  {z["transcriptomic_type"]}  (layer {z["layer"]}, '
                 f'{z["condition"]})', fontsize=9, x=0.01, ha='left')
    fig.savefig(FIG_DIR / f'cell_{sid}.png', dpi=200)
    plt.show()


# %% all cells: f-I curves together
if len(files) > 1:
    fig, ax = plt.subplots(figsize=(5, 3.6), constrained_layout=True)
    for sid, (I, r) in fI.items():
        ax.plot(I[I >= 0], r[I >= 0], '-', color='#8a8a86', lw=0.8, alpha=0.7)
    ax.set(xlabel='step (pA)', ylabel='spikes / s in step')
    ax.set_title(f'f-I curves, L2/3 VIP (n={len(fI)})', loc='left',
                 fontsize=9)
    style(ax)
    fig.savefig(FIG_DIR / 'fI_all_cells.png', dpi=200)
    plt.show()

# %% all cells: impedance profiles together
Z = {sid: (c['f'], c['z']) for sid, c in chirps.items() if c['features']}
if len(Z) > 1:
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for sid, (f, z) in Z.items():
        low = z[f <= LOW_F].mean() if (f <= LOW_F).any() else z[0]
        axs[0].plot(f, z, color='#8a8a86', lw=0.8, alpha=0.7)
        axs[1].plot(f, z / low, color='#8a8a86', lw=0.8, alpha=0.7)
    axs[0].set(ylabel='|Z| (MOhm)')
    axs[0].set_title(f'impedance, L2/3 VIP (n={len(Z)})', loc='left',
                     fontsize=9)
    axs[1].set(ylabel=f'|Z| / mean |Z| below {LOW_F:g} Hz')
    axs[1].set_title('normalised to low-frequency impedance', loc='left',
                     fontsize=9)
    axs[1].axhline(1, color=GREY, lw=0.8, ls='--')
    for ax in axs:
        ax.set(xscale='log', xlabel='frequency (Hz)')
        freq_ticks(ax)
        style(ax)
    fig.savefig(FIG_DIR / 'impedance_all_cells.png', dpi=200)
    plt.show()

print(f'\nfigures saved in {FIG_DIR}\ntable saved in {OUT_DIR}')

#