"""Single VIP interneuron: first protocols.

The model (parameters, mechanisms, helpers) is defined in vip_cell.py, a
NEURON port of VIPwithT.cpp. See compare_cpp_neuron.py for the direct
comparison with the C++ code.

Compile the new mechanisms once before running:
    cd runscripts-rt/mod
    nrnivmodl

The script is split into cells (# %%) so it can be run step by step in
PyCharm, VS Code or Jupyter.
"""

# %% imports and paths
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    RT_DIR = Path(__file__).resolve().parent
except NameError:  # running interactively
    RT_DIR = Path.cwd()
sys.path.insert(0, str(RT_DIR))

from vip_cell import (IAPP_CPP, delete_cell, density_to_nA,  # noqa: E402
                      get_vip_cell, run_current_clamp, spike_times)

FIG_DIR = RT_DIR / 'figures' / 'vip_single_cell'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %% 1. constant drive for 5 s (the C++ run uses IAPP_CPP = 9.8 uA/cm2)
IAPP = IAPP_CPP   # uA/cm2; change to try another drive
cell = get_vip_cell()
soma = cell._nrn_sections['soma']
amp = density_to_nA(soma, IAPP)
t, v = run_current_clamp(cell, amp, delay=0, dur=1e9, tstop=5000)
spk = spike_times(t, v)
delete_cell()

print(f'Iapp = {IAPP} uA/cm2 -> {amp:.4f} nA')
if len(spk):
    print(f'first spike: {spk[0]:.1f} ms, '
          f'{len(spk)} spikes, mean rate after onset '
          f'{(len(spk) - 1) / (spk[-1] - spk[0]) * 1e3:.1f} Hz')
else:
    print('no spikes')

fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
ax.plot(t, v, 'k', lw=0.6)
ax.set(xlabel='time (ms)', ylabel='V (mV)',
       title=f'VIP cell, constant Iapp = {IAPP} uA/cm2')
fig.savefig(FIG_DIR / f'constant_drive_Iapp{IAPP:g}.png', dpi=150)
plt.show()

# %% 2. resting membrane potential (no drive)
cell = get_vip_cell()
t, v = run_current_clamp(cell, 0., delay=0, dur=0, tstop=2000)
delete_cell()
print(f'RMP: {v[-1]:.2f} mV')

# %% 3. current steps (f-I curve)
# Steps are given as densities so they compare directly with the C++ Iapp.
# The D current makes this cell delay its first spike by up to seconds, so
# the steps are long.
step_densities = np.arange(0, 14.1, 1.0)   # uA/cm2
delay, dur = 200., 3000.
rates, first_spk = [], []
traces = {}
for i_d in step_densities:
    cell = get_vip_cell()
    soma = cell._nrn_sections['soma']
    t, v = run_current_clamp(cell, density_to_nA(soma, i_d),
                             delay=delay, dur=dur, tstop=delay + dur + 100)
    delete_cell()
    spk = spike_times(t, v)
    spk = spk[(spk >= delay) & (spk <= delay + dur)]
    rates.append(len(spk) / dur * 1e3)
    first_spk.append(spk[0] - delay if len(spk) else np.nan)
    traces[i_d] = (t, v)

fig, axs = plt.subplots(1, 3, figsize=(11, 3), constrained_layout=True)
for offset, i_d in enumerate((8., 10., 13.)):   # around threshold
    t_i, v_i = traces[i_d]
    axs[0].plot(t_i, v_i + 140 * offset, lw=0.6, label=f'{i_d:g}')
axs[0].set(xlabel='time (ms)', ylabel='V (mV, offset)',
           xlim=(delay - 50, delay + 400), yticks=[])
axs[0].legend(title='uA/cm2', fontsize=7)
axs[1].plot(step_densities, rates, 'o-k')
axs[1].set(xlabel='I (uA/cm2)', ylabel='rate (Hz)', title='f-I')
axs[2].plot(step_densities, first_spk, 'o-k')
axs[2].set(xlabel='I (uA/cm2)', ylabel='latency (ms)',
           title='first-spike latency')
fig.savefig(FIG_DIR / 'current_steps.png', dpi=150)
plt.show()

# %% 4. comparison with experimental data (to do)
# Once the recordings are in, load them here and extract the same features
# from data and model, as in single_cell_sim/cells/interneuron.ipynb:
#   RMP, input resistance and membrane time constant (hyperpolarising step),
#   rheobase, f-I curve, first-spike latency, AP threshold, height, width,
#   AHP depth, and spike-frequency adaptation.
# Two things to set from the data before tuning conductances:
#   - the injected currents in nA (use density_to_nA in reverse, or switch
#     the protocols above to nA),
#   - the cell size: soma area and cm set the capacitance, so match the
#     measured capacitance or input resistance (_get_inh_soma uses
#     L = diam = 20 um).
