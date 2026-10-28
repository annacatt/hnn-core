import numpy as np
from IPython.core.getipython import get_ipython
from matplotlib.lines import Line2D
import pickle
from hnn_core.parallel_backends import MPIBackend
import postproc_tm_currents_dipole_lfp_csd as tme
from hnn_core.extracellular import calculate_csd2d
#import Plotting_tools as plott
import matplotlib.pyplot as plt

from hnn_core import (
    JoblibBackend,
    jones_2009_model,
    simulate_dipole,
)
from hnn_core.cells_default import pyramidal
from hnn_core.network_builder import load_custom_mechanisms
from hnn_core.network_models import add_erp_drives_to_jones_model, duecker_ET_model

net = duecker_ET_model()
#net.use_dataframe = True

net.set_cell_positions(inplane_distance=30.)
#net.connectivity.clear() #<- clears everything, including drives
#net.clear_connectivity()  # this keeps drives, removes recurrent


# Laminar probe
depths = np.arange(-550, 2150, 100) #depths = np.arange(-625, 2150, 100)
electrode_pos = [(135, 135, z) for z in depths]
net.add_electrode_array('probe1', electrode_pos)

weights_ampa_p1 = {'L2_inhibitory': 0.015, 
                   'L2_pyramidal': 0.035,
                   'L5_inhibitory': 0, 
                   'L5_pyramidal': 0.015}
weights_nmda_p1 = {'L2_inhibitory': 0.01, 'L2_pyramidal': 0.025,
                   'L5_inhibitory': 0.0, 'L5_pyramidal': .01}
synaptic_delays_prox = {'L2_inhibitory': 0.1, 'L2_pyramidal': .1,
                        'L5_inhibitory': 1., 'L5_pyramidal': 1.}

net.add_evoked_drive('prox1', mu=15.13, sigma=3.1, numspikes=1, weights_ampa=weights_ampa_p1,
    weights_nmda=weights_nmda_p1, location='proximal', synaptic_delays=synaptic_delays_prox)

weights_ampa_d1 = {'L2_inhibitory': 3e-6, 'L2_pyramidal': 0.0001, 'L5_pyramidal': 2.5}
weights_nmda_d1 = {'L2_inhibitory': 8e-12, 'L2_pyramidal': 0.0001, 'L5_pyramidal': 0.04}
synaptic_delays_dist = {'L2_inhibitory': 0.5, 'L2_pyramidal': 0.5,'L5_pyramidal': .5}

net.add_evoked_drive('dist1', mu=60, sigma=3.7, numspikes=4, weights_ampa=weights_ampa_d1,
    weights_nmda=weights_nmda_d1, location='distal', synaptic_delays=synaptic_delays_dist)

weights_ampa_p2 = {'L2_inhibitory': 0.0, 'L2_pyramidal':0.014,
                   'L5_inhibitory': 2e-7, 'L5_pyramidal': 0.34}
weights_nmda_p2 = {'L2_inhibitory': 0.0, 'L2_pyramidal': 2e-5,
                   'L5_inhibitory': 5.7e-8, 'L5_pyramidal': 0.0027}
synaptic_delays_prox = {'L2_inhibitory': 0.1, 'L2_pyramidal': .1,
                        'L5_inhibitory': 1., 'L5_pyramidal': 1.}
net.add_evoked_drive('prox2', mu=122, sigma=13.4, numspikes=3, weights_ampa=weights_ampa_p2,
    weights_nmda=weights_nmda_p2, location='proximal', synaptic_delays=synaptic_delays_prox)


record_method = "all" # or "soma"

tm_currents={
    # aggregated (summed over all the mechanisms in the segment)
    "agg_i_mem": {
        "type": "derived",
        "mech": None,
        "ref": "_ref_i_membrane_",
        "per_segment": True,
        "recorded_sections": record_method,
    },
    "agg_i_cap": {
        "type": "derived",
        "mech": None,
        "ref": "_ref_i_cap",
        "per_segment": True,
        "recorded_sections": record_method,
    },
    "agg_ina": {
        "type": "ionic",
        "mech": None,
        "ref": "_ref_ina",
        "per_segment": True,
        "recorded_sections": record_method,
    },
    #"ina_hh2": {
    #    "type": "ionic",
    #    "mech": "hh2",
    #    "ref": "_ref_ina",
    #    "per_segment": True,
    #    "recorded_sections": record_method,
    #},
    "agg_ik": {
        "type": "ionic", 
        "mech": None, 
        "ref": "_ref_ik",
        "per_segment": True, 
        "recorded_sections": record_method,
        },
    "agg_ica": {
        "type": "ionic",
        "mech": None,
        "ref": "_ref_ica",
        "per_segment": True, 
        "recorded_sections": record_method,
        },
    
    # sodium (summed over all the mechanisms in the segment)
    "ina_NaTa_t_hay2011": {
        "type": "ionic",
        "mech": "NaTa_t_hay2011",
        "ref": "_ref_ina",
        "per_segment": True, 
        "recorded_sections": record_method,
        },
    "ina_NaTa_t_32d_hay2011": {
        "type": "ionic",
        "mech": "NaTa_t_32d_hay2011",
        "ref": "_ref_ina",
        "per_segment": True,
        "recorded_sections": record_method,
        },
    "ina_NaTs2_t_hay2011": {
        "type": "ionic",
        "mech": "NaTs2_t_hay2011",
        "ref": "_ref_ina",
        "per_segment": True,
        "recorded_sections": record_method,
        },
    "ina_NaTs2_t_32d_hay2011": {
        "type": "ionic",
        "mech": "NaTs2_t_32d_hay2011",
        "ref": "_ref_ina",
        "per_segment": True, 
        "recorded_sections": record_method,
        },
    "ina_Nap_Et2_hay2011": {
        "type": "ionic",
        "mech": "Nap_Et2_hay2011",
        "ref": "_ref_ina",
        "per_segment": True, 
        "recorded_sections": record_method,
        },
    "ina_nas_golomb2007": {
        "type": "ionic",
        "mech": "nas_golomb2007",
        "ref": "_ref_ina",
        "per_segment": True,
        "recorded_sections": record_method,
    },

    # potassium:
    "ik_SKv3_1_hay2011": {
        "type": "ionic",
        "mech": "SKv3_1_hay2011",
        "ref": "_ref_ik",
        "per_segment": True,
        "recorded_sections": record_method,
    },
    "ik_SK_E2_hay2011": {
        "type": "ionic",
        "mech": "SK_E2_hay2011",
        "ref": "_ref_ik",
        "per_segment": True,
        "recorded_sections": record_method,
    },
    "ik_Im_hay2011": {
        "type": "ionic",
        "mech": "Im_hay2011",
        "ref": "_ref_ik",
        "per_segment": True,
        "recorded_sections": record_method,
    },
    "ik_K_Pst_hay2011": {
        "type": "ionic",
        "mech": "K_Pst_hay2011",
        "ref": "_ref_ik",
        "per_segment": True,
        "recorded_sections": record_method,
    },
    "ik_K_Tst_hay2011": {
        "type": "ionic",
        "mech": "K_Tst_hay2011",
        "ref": "_ref_ik",
        "per_segment": True,
        "recorded_sections": record_method,
    },
    "ik_kdr_golomb2007": {
        "type": "ionic",
        "mech": "kdr_golomb2007",
        "ref": "_ref_ik",
        "per_segment": True,
        "recorded_sections": record_method,
    },
    "ik_kd_golomb2007": {
        "type": "ionic",
        "mech": "kd_golomb2007",
        "ref": "_ref_ik",
        "per_segment": True,
        "recorded_sections": record_method,
    },

    # calcium:
    "ica_Ca_HVA_hay2011": {
        "type": "ionic",
        "mech": "Ca_HVA_hay2011",
        "ref": "_ref_ica",
        "per_segment": True,
        "recorded_sections": record_method,
    },
    "ica_Ca_LVAst_hay2011": {
        "type": "ionic",
        "mech": "Ca_LVAst_hay2011",
        "ref": "_ref_ica",
        "per_segment": True,
        "recorded_sections": record_method,
    },

    # non-specific:
    "ihcn_Ih_hay2011": {
        "type": "ionic",
        "mech": "Ih_hay2011",
        "ref": "_ref_ihcn",
        "per_segment": True,
        "recorded_sections": record_method,
    },
    "i_pas": {
        "type": "ionic",
        "mech": "pas",
        "ref": "_ref_i",
        "per_segment": True,
        "recorded_sections": record_method,
    },
}



n_trials = 1

adaptive = True
dt = 0.025 # ms

if "dpls" not in locals():
    with MPIBackend(n_procs=1):
        dpls = simulate_dipole(
            net,
            tstop=70.0,#170.0,#170.0,
            dt=dt,#0.00625,#0.0125, # 0.025
            n_trials=n_trials,
            tm_currents=tm_currents,
            record_isec="all",   # synaptic currents
            record_vsec="soma",   # voltages
            #record_ca="all",     # calcium concentration
            bsl_cor="duecker"
        )

'''
if "dpls" not in locals():
    with JoblibBackend(1):
        dpls = simulate_dipole(
            net,
            tstop=170.0,
            dt=0.025,#0.00025,#0.00625,#0.0125, # 0.025
            n_trials=n_trials,
            record_vsec='soma'
            #record_agg_i_mem="all",   # aggregated total transmembrane current
            ## record_agg_ina="all",
            ## record_agg_ik="all",
            #record_agg_i_cap="all",   # aggregated capacitive current
            #record_ina_hh2="all",
            #record_ik_hh2="all",
            #record_ik_kca="all",
            #record_ik_km="all",
            #record_ica_ca="all",
            #record_ica_cat="all",
            #record_il_hh2="all",      # aggregated leak current
            #record_i_ar="all",
            #record_isec="all",
        )
'''

scaling_factor = 3000
for dpl in dpls:
    dpl.scale(scaling_factor)

def _to_numpy_nested(d):
    """Recursively convert every leaf list in a nested dict to a numpy array, in place."""
    for key, value in d.items():
        if isinstance(value, dict):
            _to_numpy_nested(value)
        elif value is not None:
            d[key] = np.asarray(value, dtype=float)


def convert_currents_to_numpy(net, channels=None, include_isec=True,
                              include_vsec=True, include_ca=True):
    cr = net.cell_response
    cr._times = np.asarray(cr.times, dtype=float)

    channels = channels or list(cr.tm_currents.keys())
    for channel in channels:
        for trial_data in cr.tm_currents[channel]["data"]:
            _to_numpy_nested(trial_data)

    for flag, attr in [(include_isec, "_isec"), (include_vsec, "_vsec"), (include_ca, "_ca")]:
        if flag:
            for trial_data in getattr(cr, attr):
                _to_numpy_nested(trial_data)


# match the record_* settings used in simulate_dipole above
convert_currents_to_numpy(net, include_isec=True, include_vsec=True, include_ca=False)

dpl = dpls[0]
times = net.cell_response.times
vsec = net.cell_response.vsec[0]

import os
import pickle

#os.makedirs('runscripts/data', exist_ok=True)

#with open('runscripts/data/dipole_adaptive_timestep.pkl', 'wb') as f:
#with open('runscripts/data/dipole_fixed_timestep.pkl', 'wb') as f:
#    pickle.dump(dpls, f)


results = {
    'net': net,
    'dipole': dpl,
    'dt': dt,
    'depths': depths,
    'cell_response': net.cell_response,
    'rec_arrays': net.rec_arrays,
    'vsec': vsec,
    'times': times
}
dt_tag = str(dt).replace('.', '')          # 0.025 -> '0025', 0.00625 -> '000625'
step_tag = 'adaptive_timestep' if adaptive else f'fixed_timestep_{dt_tag}'
fname = f'runscripts/data/TM_duecker_{step_tag}_voltage.pkl'
print('Saving to', fname)

with open(fname, 'wb') as f:
    pickle.dump(results, f)

'''
#with open('runscripts/data/Fixed_timestep.pkl', 'wb') as f:
#with open('runscripts/data/Fixed_timestep_0025.pkl', 'wb') as f:
with open('runscripts/data/duecker_fixed_timestep_0025_voltage.pkl', 'wb') as f:
#with open('runscripts/data/Adaptive_timestep.pkl', 'wb') as f:
    pickle.dump(results, f)
'''
# UP to here to save new simulation results.


# -------

# Start from here to load the results and plot them.

import os
from hnn_core import Dipole

dts_fixed = [0.025, 0.00625, 0.0025, 0.00025]   # ms, fixed-step runs to compare
include_adaptive = True     # also load duecker_adaptive_timestep_voltage.pkl if it exists
data_dir = 'runscripts/data'


def _fname(dt=None):
    """File name used when saving: same rule as in the first part of the script."""
    if dt is None:
        return os.path.join(data_dir, 'duecker_adaptive_timestep_voltage.pkl')
    dt_tag = str(dt).replace('.', '')          # 0.025 -> '0025', 0.00625 -> '000625'
    return os.path.join(data_dir, f'duecker_fixed_timestep_{dt_tag}_voltage.pkl')


# Load every run into one dict: label -> results dict
runs = {}
for dt_i in dts_fixed:
    fname = _fname(dt_i)
    if not os.path.exists(fname):
        print(f'Missing, skipped: {fname}')
        continue
    with open(fname, 'rb') as f:
        runs[f'Fixed dt={dt_i}'] = pickle.load(f)

if include_adaptive and os.path.exists(_fname(None)):
    with open(_fname(None), 'rb') as f:
        runs['Adaptive'] = pickle.load(f)

print('Loaded runs:', list(runs))


# unsmoothed dipole
plt.figure(figsize=(10, 4))
for label, res in runs.items():
    plt.plot(res['dipole'].times, res['dipole'].data['agg'], label=label)
plt.xlabel('Time (ms)')
plt.ylabel('Dipole (nAm)')
plt.title('Duecker model: dipole at different time steps')
plt.legend()
plt.show()


# smoothed dipole
window_len = 30  # ms
dt_uniform = 0.00025  # ms, grid used to resample the adaptive run before smoothing

smoothed = {}
for label, res in runs.items():
    dpl_i = res['dipole']
    if label == 'Adaptive':
        # 1. handle duplicate timestamps
        t_adapt, idx = np.unique(dpl_i.times, return_index=True)
        data_adapt = dpl_i.data['agg'][idx]
        # 2. interpolate onto a uniform grid
        t_uniform = np.arange(t_adapt[0], t_adapt[-1], dt_uniform)
        data_uniform = np.interp(t_uniform, t_adapt, data_adapt)
        dpl_i = Dipole(t_uniform, data_uniform)  # single column -> treated as 'agg'
    # 3. smoothing
    smoothed[label] = dpl_i.copy().smooth(window_len)

plt.figure(figsize=(10, 4))
for label, dpl_s in smoothed.items():
    plt.plot(dpl_s.times, dpl_s.data['agg'], label=label)
plt.xlabel('Time (ms)')
plt.ylabel('Smoothed dipole (nAm)')
plt.title(f'Duecker model: smoothed dipole ({window_len} ms window)')
plt.legend()
plt.show()


# raster plots
# Duecker gid ranges: L2_inhibitory 0-34, L2_pyramidal 35-134,
#                     L5_inhibitory 135-169, L5_pyramidal 170-269
plt.figure(figsize=(10, 6))
for label, res in runs.items():
    cr = res['cell_response']
    gids = np.array(cr.spike_gids[0])
    spike_times = np.array(cr.spike_times[0])
    mask = gids < 270          # exclude drive (artificial) cells
    plt.scatter(spike_times[mask], gids[mask], s=2, label=label)

# divide cell types
plt.axhline(34.5, linestyle='--', linewidth=1, color='gray')
plt.axhline(134.5, linestyle='--', linewidth=1, color='gray')
plt.axhline(169.5, linestyle='--', linewidth=1, color='gray')
plt.ylim(270, 0)
plt.xlabel('Time (ms)')
plt.ylabel('GID')
plt.legend()
plt.show()


# compare the somatic membrane potentials
gid = 199   # 135-169: L5_inhibitory; use 170-269 for an L5_pyramidal cell
plt.figure(figsize=(10, 6))
for label, res in runs.items():
    plt.plot(res['times'], res['vsec'][gid]['soma'], label=label)
plt.xlabel('Time (ms)')
plt.ylabel('Vm soma (mV)')
plt.title(f'Soma membrane potential, gid {gid}')
plt.legend()
plt.show()



'''
plt.figure(figsize=(10, 6))

plt.scatter(cell_response_adaptive.spike_times[0],
            cell_response_adaptive.spike_gids[0],
            s=2,
            label='Adaptive')

plt.scatter(cell_response_fixed_0025.spike_times[0],
            cell_response_fixed_0025.spike_gids[0],
            s=2,
            label='Fixed')

plt.xlabel('Time (ms)')
plt.ylabel('GID')
plt.legend()
plt.show()
'''


