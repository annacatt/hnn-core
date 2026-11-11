"""Extract the 1 s current steps and the chirps from Lee-Dalley NWB files (DANDI 000636).

The raw recordings of Lee, Dalley et al. (2023) are NWB files on DANDI
(https://dandiarchive.org/dandiset/000636), named
sub-<donor>_ses-<nwb_id>_icephys.nwb, where <nwb_id> is the nwb_id column of
LeeDalley_manuscript_metadata_v2.csv. This script reads those files and saves,
for each cell, its long current-step sweeps (hyperpolarising and
depolarising) in a small NumPy file that vip_ephys_traces.py plots with only
numpy, pandas and matplotlib, and its chirp sweeps (a sinusoidal current of
rising frequency, for subthreshold impedance and resonance) in a second file.

Needs h5py (NWB files are HDF5). It is meant to be run in a separate
environment or sandbox, so that your main environment stays unchanged:
    python vip_nwb_extract.py              # every .nwb file in NWB_DIR
    python vip_nwb_extract.py --download   # also fetch missing L2/3 VIP files
                                           # from DANDI first

Input:  data/lee_dalley/*.nwb  (and data/lee_dalley/nwb/*.nwb)
Output: data/lee_dalley/nwb_steps/<specimen_id>.npz with
    t      time relative to step onset (s), sampled at OUT_RATE
    v      membrane potential, one row per sweep (mV)
    i_pA   step amplitude of each sweep (pA)
    sweep  sweep number in the NWB file
    stim   stimulus name of each sweep
    specimen_id, nwb_id, cell_name, transcriptomic_type, layer, condition
and, for cells with chirp sweeps, data/lee_dalley/nwb_chirp/<specimen_id>.npz:
    t      time relative to chirp onset (s), sampled at OUT_RATE, from
           CHIRP_PAD before the chirp to up to CHIRP_PAD after it ends
    v      membrane potential, one row per sweep (mV)
    i      injected current, one row per sweep (pA)
    dur    duration of each chirp (s)
    Sweeps can have different chirps (older protocols rotate a 40 s, a 20 s
    and a 10 s chirp); shorter sweeps are padded with NaN at the end.
    sweep, stim and the same cell information as above

Step sweeps are recognised by their current: a single step lasting
0.9-1.1 s (the short test pulse at the start of each sweep is ignored).
Chirp sweeps are recognised by their stimulus name (containing "CHIRP", e.g.
C2CHIRP171103); the chirp itself is the longest stretch of nonzero current
in the sweep, allowing short gaps where it crosses zero. Voltage is
downsampled from 50 kHz to OUT_RATE by keeping every n-th sample: spike
counts and latencies are unaffected, but spike peaks can be slightly
underestimated, so use the original file for spike-shape analysis.

Sweeps with missing data are dropped, and listed when the script runs: a
sweep cut short (the kept window runs past its end), one with NaNs, or one
where V stays exactly constant for MAX_FLAT or longer. The last case is a
sweep that stopped recording early and was padded with a constant value
(usually 0 mV): a real recording always has noise, so V never repeats the
same value for that long.
"""
import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import h5py
except ImportError:
    sys.exit('This script needs h5py. Run it in a separate environment, e.g.\n'
             '    python3 -m venv ~/nwbenv && ~/nwbenv/bin/pip install h5py '
             'numpy pandas\n'
             '    ~/nwbenv/bin/python vip_nwb_extract.py')

RT_DIR = Path(__file__).resolve().parent
DATA_DIR = RT_DIR / 'data' / 'lee_dalley'
NWB_DIR = DATA_DIR / 'nwb'                 # downloads go here
OUT_DIR = DATA_DIR / 'nwb_steps'
CHIRP_DIR = DATA_DIR / 'nwb_chirp'
for d in (OUT_DIR, CHIRP_DIR):
    d.mkdir(parents=True, exist_ok=True)

OUT_RATE = 10_000          # Hz, sampling rate of the saved traces
T_BEFORE, T_AFTER = 0.25, 1.5   # s kept around step onset (step lasts 1 s)
LAYERS = ['2', '3', '2_3']
MAX_FLAT = 0.02            # s: V unchanged this long means missing data
CHIRP_PAD = 0.5            # s kept before and after each chirp

DANDI_API = 'https://api.dandiarchive.org/api'
DANDISET, VERSION = '000636', '0.241120.0510'


def l23_vip(meta):
    keep = (meta['Revised_subclass_label'] == 'VIP') \
        & meta['Cortical_layer'].astype(str).isin(LAYERS) \
        & meta['nwb_id'].notna()
    return meta[keep]


def download(nwb_id):
    """Find sub-*_ses-<nwb_id>_icephys.nwb on DANDI and download it."""
    query = urllib.parse.urlencode({'glob': f'*ses-{nwb_id}_*'})
    url = (f'{DANDI_API}/dandisets/{DANDISET}/versions/{VERSION}/assets/'
           f'?{query}')
    with urllib.request.urlopen(url, timeout=60) as r:
        res = json.load(r)['results']
    if len(res) != 1:
        print(f'  nwb_id {nwb_id}: {len(res)} matching files on DANDI, '
              'skipped')
        return None
    path = NWB_DIR / Path(res[0]['path']).name
    if not path.exists():
        NWB_DIR.mkdir(parents=True, exist_ok=True)
        print(f'  downloading {path.name} '
              f'({res[0].get("size", 0) / 1e6:.0f} MB)')
        tmp = path.with_suffix('.part')
        urllib.request.urlretrieve(
            f'{DANDI_API}/assets/{res[0]["asset_id"]}/download/', tmp)
        tmp.rename(path)
    return path


def step_of(i_pA, rate):
    """Onset (sample) and amplitude (pA) of a single 0.9-1.1 s step, else
    None."""
    on_mask = np.abs(i_pA) > 0.5
    edges = np.diff(np.r_[0, on_mask.astype(np.int8), 0])
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    long_ = [(a, b) for a, b in zip(starts, stops)
             if 0.9 * rate <= b - a <= 1.1 * rate]
    if len(long_) != 1:
        return None
    a, b = long_[0]
    return a, round(float(np.median(i_pA[a:b])), 1)


def chirp_span(i_pA, rate, max_gap=0.1, min_dur=1.0):
    """Start and stop (samples) of the chirp in a sweep, else None.

    Nonzero samples closer than max_gap (s) are joined into one stretch, so
    the chirp's zero crossings do not split it; the short test pulse at the
    start of the sweep is a separate stretch. The chirp is the longest
    stretch, if it lasts at least min_dur (s)."""
    on = np.flatnonzero(np.abs(i_pA) > 0.5)
    if len(on) == 0:
        return None
    breaks = np.flatnonzero(np.diff(on) > max_gap * rate)
    starts = np.r_[on[0], on[breaks + 1]]
    stops = np.r_[on[breaks], on[-1]] + 1
    k = int(np.argmax(stops - starts))
    if stops[k] - starts[k] < min_dur * rate:
        return None
    return int(starts[k]), int(stops[k])


def missing_data(v, rate):
    """Reason the trace has missing data (NaN, or V constant for MAX_FLAT or
    longer), else None."""
    if not np.isfinite(v).all():
        return 'contains NaN'
    flat = longest_flat(v)
    if flat >= MAX_FLAT * rate:
        last = np.flatnonzero(np.diff(v) == 0)[-1]
        return (f'V constant at {v[last]:.1f} mV for '
                f'{flat / rate * 1e3:.0f} ms')
    return None


def cell_info(cell):
    return dict(specimen_id=int(cell['specimen_id_x']),
                nwb_id=int(cell['nwb_id']), cell_name=str(cell['cell_name']),
                transcriptomic_type=str(cell['Transcriptomic_type']),
                layer=str(cell['Cortical_layer']),
                condition=str(cell['condition']))


def longest_flat(v):
    """Length (samples) of the longest run of identical consecutive values."""
    same = np.diff(v) == 0
    if not same.any():
        return 1
    edges = np.diff(np.r_[0, same.astype(np.int8), 0])
    runs = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return int(runs.max()) + 1


def extract(path, cell):
    """Save the cell's step and chirp sweeps. Return the step npz path (or
    None), the number of step sweeps kept, the chirp npz path (or None), the
    number of chirp sweeps kept, and the dropped sweeps with the reason."""
    with h5py.File(path, 'r') as f:
        acq, stim = f['acquisition'], f['stimulus/presentation']
        rows, chirps, dropped = [], [], []
        for name in sorted(acq):
            a = acq[name]
            if a.attrs.get('neurodata_type') != 'CurrentClampSeries':
                continue
            s_name = name.replace('_AD', '_DA')
            if s_name not in stim:
                continue
            rate = float(a['starting_time'].attrs['rate'])
            s = stim[s_name]
            i_pA = s['data'][()] * float(s['data'].attrs['conversion']) * 1e12
            desc = str(a.attrs['stimulus_description'])
            sweep = int(a.attrs['sweep_number'])
            every = int(round(rate / OUT_RATE))
            if 'CHIRP' in desc.upper():
                span = chirp_span(i_pA, rate)
                if span is None:
                    dropped.append(f'sweep {sweep} (chirp): no chirp found '
                                   'in the current')
                    continue
                v = a['data'][()] * float(a['data'].attrs['conversion']) * 1e3
                lo = span[0] - int(CHIRP_PAD * rate)
                hi = min(span[1] + int(CHIRP_PAD * rate), len(v))
                if lo < 0 or span[1] > len(v):
                    dropped.append(f'sweep {sweep} (chirp): cut short')
                    continue
                why = missing_data(v[lo:hi], rate)
                if why:
                    dropped.append(f'sweep {sweep} (chirp): {why}')
                    continue
                chirps.append((sweep, desc, (span[1] - span[0]) / rate,
                               v[lo:hi:every].astype(np.float32),
                               i_pA[lo:hi:every].astype(np.float32)))
                continue
            step = step_of(i_pA, rate)
            if step is None:
                continue
            onset, amp = step
            v = a['data'][()] * float(a['data'].attrs['conversion']) * 1e3
            lo, hi = onset - int(T_BEFORE * rate), onset + int(T_AFTER * rate)
            if lo < 0 or hi > len(v):
                dropped.append(f'sweep {sweep} ({amp:g} pA): cut short')
                continue
            why = missing_data(v[lo:hi], rate)
            if why:
                dropped.append(f'sweep {sweep} ({amp:g} pA): {why}')
                continue
            rows.append((amp, sweep, desc,
                         v[lo:hi:every].astype(np.float32)))
    info = cell_info(cell)
    out = chirp_out = None
    if rows:
        rows.sort(key=lambda r: (r[0], r[1]))
        n = min(len(r[3]) for r in rows)
        t = (np.arange(n) / OUT_RATE - T_BEFORE).astype(np.float32)
        out = OUT_DIR / f'{info["specimen_id"]}.npz'
        np.savez_compressed(
            out, t=t, v=np.stack([r[3][:n] for r in rows]),
            i_pA=np.array([r[0] for r in rows]),
            sweep=np.array([r[1] for r in rows]),
            stim=np.array([r[2] for r in rows]), **info)
    if chirps:
        chirps.sort(key=lambda r: r[0])
        # chirps can differ in length: pad the shorter ones with NaN
        n = max(len(r[3]) for r in chirps)
        t = (np.arange(n) / OUT_RATE - CHIRP_PAD).astype(np.float32)

        def pad(x):
            return np.r_[x, np.full(n - len(x), np.nan, np.float32)]
        chirp_out = CHIRP_DIR / f'{info["specimen_id"]}.npz'
        np.savez_compressed(
            chirp_out, t=t, v=np.stack([pad(r[3]) for r in chirps]),
            i=np.stack([pad(r[4]) for r in chirps]),
            dur=np.array([r[2] for r in chirps]),
            sweep=np.array([r[0] for r in chirps]),
            stim=np.array([r[1] for r in chirps]), **info)
    return out, len(rows), chirp_out, len(chirps), dropped


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--download', action='store_true',
                   help='download missing L2/3 VIP files from DANDI first')
    args = p.parse_args()

    meta = pd.read_csv(DATA_DIR / 'LeeDalley_manuscript_metadata_v2.csv')
    meta.columns = meta.columns.str.strip()
    by_nwb = meta.dropna(subset=['nwb_id']).set_index(
        meta['nwb_id'].dropna().astype('int64'))

    if args.download:
        cells = l23_vip(meta)
        print(f'{len(cells)} L2/3 VIP cells with an nwb_id')
        for nwb_id in cells['nwb_id'].astype('int64'):
            if not any(DATA_DIR.rglob(f'*ses-{nwb_id}_*.nwb')):
                download(nwb_id)

    files = sorted(set(DATA_DIR.glob('*.nwb')) | set(NWB_DIR.glob('*.nwb')))
    if not files:
        sys.exit(f'No .nwb files in {DATA_DIR} or {NWB_DIR}.')
    for path in files:
        try:
            nwb_id = int(path.name.split('_ses-')[1].split('_')[0])
        except (IndexError, ValueError):
            print(f'{path.name}: name not sub-*_ses-<id>_icephys.nwb, skipped')
            continue
        if nwb_id not in by_nwb.index:
            print(f'{path.name}: nwb_id {nwb_id} not in the metadata, skipped')
            continue
        cell = by_nwb.loc[nwb_id]
        out, n_steps, chirp_out, n_chirps, dropped = extract(path, cell)
        print(f'{path.name}: {n_steps} step sweeps, {n_chirps} chirp sweeps'
              + (f' -> {out.name}' if out else ''))
        for d in dropped:
            print(f'    dropped {d}')


if __name__ == '__main__':
    main()
