"""VIP interneuron model in NEURON (Katharina's "hnn-tuning" format).

NEURON version of VIPwithT.cpp (Michelle's code): a single compartment VIP interneuron
with fast Na, delayed-rectifier K, D (slowly inactivating K), M (slow K)
and T-type Ca currents plus a leak.

This module only defines the model and simulation helpers.
Run vip_single_cell.py for simulation; the C++ comparison is in compare_cpp_neuron.py.

Mechanisms
----------
From hnn-core (hnn_core/mod, already used by hnn-tuning's interneuron):
    nas_golomb2007, kdr_golomb2007, kd_golomb2007
    Their equations are identical to the Na, K and D currents in the C++.
New, in runscripts-rt/mod (transcribed from VIPwithT.cpp):
    km_mccarthy2023   M current
    cat_mccarthy2023  T current (GHK, fixed [Ca]i and [Ca]o)

To compile new mechanisms:
    cd runscripts-rt/mod
    nrnivmodl

Unit conversions from the C++ (which uses uA/cm2, mS/cm2, uF/cm2):
    conductances  mS/cm2 -> S/cm2  (x 1e-3)
    T current     Pca * VIParea = 80 * 0.00029 -> gbar = 0.0232 * 1e-3
    applied current Iapp (uA/cm2) -> IClamp amplitude (nA) via the soma area,
    see density_to_nA().

Cell building uses local copies of two hnn-tuning helper files
(cell_builder_utils.py, inh_mechs_utils.py, from hnn-tuning commit fac61b4),
so hnn-tuning does not need to be installed
"""

import sys
from pathlib import Path

import neuron
import numpy as np
from neuron import h
from neuron.units import ms, mV

import hnn_core

RT_DIR = Path(__file__).resolve().parent

# local copies of the hnn-tuning helpers (see the header of each file)
sys.path.insert(0, str(RT_DIR))
from cell_builder_utils import Cell, _get_inh_soma  # noqa: E402
from inh_mechs_utils import Mechs as inhmechs  # noqa: E402

# hnn-core's mod folder, found the same way hnn-tuning's _paths.py does
HNN_CORE_MOD = Path(hnn_core.__file__).parent / 'mod'
MOD_DIR = RT_DIR / 'mod'

# hnn-core mechanisms (Golomb 2007 channels), then the new VIP mechanisms
mech_class = inhmechs(str(HNN_CORE_MOD))
neuron.load_mechanisms(str(MOD_DIR))
h.load_file('stdrun.hoc')

# --- model parameters (VIPwithT.cpp values, converted to NEURON units) ---

# keys are mechanism names; values are gbar_<mech> as Mechs.add_mechs expects
vip_params = {'soma': {
    'nas_golomb2007': 0.1125,             # gna = 112.5 mS/cm2
    'kdr_golomb2007': 0.225,              # gk  = 225 mS/cm2
    'kd_golomb2007': 0.003,               # gd  = 3 mS/cm2
    'km_mccarthy2023': 0.001,             # gm  = 1 mS/cm2
    'cat_mccarthy2023': 80 * 0.00029e-3,  # Pca * VIParea * 1e-3
    'pas': {'g': 0.25e-3,                 # gl  = 0.25 mS/cm2
            'e': -70.},                   # El  = -70 mV
}}

reversal = {'ena': 50., 'ek': -90.}       # Ena, Ek in the C++

IAPP_CPP = 9.8   # uA/cm2, constant drive used in VIPwithT.cpp
V_INIT = -63.    # mV, initial voltage in VIPwithT.cpp
CELSIUS = 37.    # the VIP mechanisms do not depend on temperature

# Time step: the C++ uses RK4 with dt = 0.05 ms. With NEURON's default
# implicit Euler, dt = 0.01 ms reproduces the C++ burst timing within ~2 %
# (0.025 ms drifts by ~5 %; spike shapes match at both).
DT = 0.01


# --- helper functions ---

def get_vip_cell(params=vip_params, rev=reversal):
    """Build a fresh single-compartment VIP cell with mechanisms inserted."""
    section = {'soma': _get_inh_soma()}   # L = diam = 20 um, cm = 1 uF/cm2
    cell = Cell(section, cell_tree=None, cell_type='VIP', gid=0)
    cell.build()
    cell = mech_class.add_mechs(cell, params)
    soma = cell._nrn_sections['soma']
    soma.ena = rev['ena']
    soma.ek = rev['ek']
    return cell


def density_to_nA(sec, i_density):
    """Convert a current density (uA/cm2) to an IClamp amplitude (nA)."""
    area_cm2 = sum(seg.area() for seg in sec) * 1e-8   # um2 -> cm2
    return i_density * area_cm2 * 1e3                   # uA -> nA


def run_current_clamp(cell, amp_nA, delay, dur, tstop, dt=DT,
                      v_init=V_INIT, celsius=CELSIUS, record_dt=None):
    """Inject a current step into the soma and return t, v (numpy arrays).

    record_dt : float | None
        Sampling interval of the returned traces (ms). None records every
        time step; e.g. 0.05 matches the C++ output.
    """
    soma = cell._nrn_sections['soma']
    stim = h.IClamp(soma(0.5))
    stim.delay, stim.dur, stim.amp = delay, dur, amp_nA

    if record_dt is None:
        t = h.Vector().record(h._ref_t)
        v = h.Vector().record(soma(0.5)._ref_v)
    else:
        t = h.Vector().record(h._ref_t, record_dt)
        v = h.Vector().record(soma(0.5)._ref_v, record_dt)

    h.dt = dt
    h.celsius = celsius
    h.finitialize(v_init * mV)
    h.continuerun(tstop * ms)
    return t.as_numpy().copy(), v.as_numpy().copy()


def spike_times(t, v, thresh=0.):
    """Times of upward crossings of `thresh` (mV)."""
    idx = np.where((v[:-1] <= thresh) & (v[1:] > thresh))[0] + 1
    return t[idx]


def delete_cell():
    h('forall delete_section()')
