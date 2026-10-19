#dpl_gabaa = tme.reconstruct_dipole_from_sources(net, sources_syn_GABAA, I_syn_GABAA)import matplotlib.pyplot as plt
import numpy as np
#from IPython.core.getipython import get_ipython
#from matplotlib.lines import Line2D
import pickle
#import postproc_tm_currents_dipole_lfp_csd as tme
#import tm_currents_utils_forLFP as tme_lfp
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
from hnn_core.network_models import add_erp_drives_to_jones_model

net = jones_2009_model()
add_erp_drives_to_jones_model(net)
net.set_cell_positions(inplane_distance=30.)
#net.connectivity.clear() #<- clears everything, including drives
#net.clear_connectivity()  # this keeps drives, removes recurrent


# Laminar probe
depths = np.arange(-625, 2150, 100)
electrode_pos = [(135, 135, z) for z in depths]
net.add_electrode_array('probe1', electrode_pos)

n_trials = 1

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
l5_component_channels = [
    "agg_i_cap",
    "ina_hh2",
    "ik_hh2",
    "ik_kca",
    "ik_km",
    "ica_ca",
    "ica_cat",
    "il_hh2",
    "i_ar",
]
'''

scaling_factor = 3000
for dpl in dpls:
    dpl.scale(scaling_factor)

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
    'dipole': dpl,
    'cell_response': net.cell_response,
    'rec_arrays': net.rec_arrays,
    'vsec': vsec,
    'times': times
}

#with open('runscripts/data/Fixed_timestep.pkl', 'wb') as f:
#with open('runscripts/data/Fixed_timestep_0025.pkl', 'wb') as f:
with open('runscripts/data/Fixed_timestep_0025_voltage.pkl', 'wb') as f:
#with open('runscripts/data/Adaptive_timestep.pkl', 'wb') as f:
    pickle.dump(results, f)

# UP to here to save new simulation results.


# ------- 

# Start from here to load the results and plot them.

# to plot the results:
# Load fixed timestep dipole (dt=0.025)
with open('runscripts/data/Fixed_timestep_0025_voltage.pkl', 'rb') as f:
    results = pickle.load(f)

dpl_fixed_0025 = results['dipole']
cell_response_fixed_0025 = results['cell_response']
rec_arrays_fixed_0025 = results['rec_arrays']
vsec_fixed_0025 = results['vsec']
times_0025 = results['times']

#v = vsec_fixed_0025[265]['soma']                      # membrane potential trace for that cell
#plt.plot(times_0025, vsec_fixed_0025[265]['soma'] )

with open('runscripts/data/Fixed_timestep_00025_voltage.pkl', 'rb') as f:
    results = pickle.load(f)

dpl_fixed_00025 = results['dipole']
cell_response_fixed_00025 = results['cell_response']
rec_arrays_fixed_00025 = results['rec_arrays']
vsec_fixed_00025 = results['vsec']
times_00025 = results['times']

with open('runscripts/data/Fixed_timestep_000025_voltage.pkl', 'rb') as f:
    results = pickle.load(f)

dpl_fixed_000025 = results['dipole']
cell_response_fixed_000025 = results['cell_response']
rec_arrays_fixed_000025 = results['rec_arrays']
vsec_fixed_000025 = results['vsec']
times_000025 = results['times']

# Load adaptive timestep dipole
with open('runscripts/data/Adaptive_timestep.pkl', 'rb') as f:
    results = pickle.load(f)

dpl_adaptive = results['dipole']
cell_response_adaptive = results['cell_response']
rec_arrays_adaptive = results['rec_arrays']
vsec_adaptive = results['vsec']
times_adaptive = results['times']


# unsmoothed dipole
plt.figure(figsize=(10, 4))

plt.plot(
    dpl_fixed_0025.times,
    dpl_fixed_0025.data['agg'],
    label='Fixed timestep (dt=0.025)'
)

plt.plot(
    dpl_fixed_00025.times,
    dpl_fixed_00025.data['agg'],
    label='Fixed timestep (dt=0.0025)'
)

plt.plot(
    dpl_fixed_000025.times,
    dpl_fixed_000025.data['agg'],
    label='Fixed timestep (dt=0.00025)'
)

plt.plot(
    dpl_adaptive.times,
    dpl_adaptive.data['agg'],
    label='Adaptive timestep'
)

plt.xlabel('Time (ms)')
plt.ylabel('Dipole (nAm)')
plt.legend()
plt.show()




window_len = 30  # ms

dpl_fixed_0025_smooth = dpl_fixed_0025.copy().smooth(window_len)
dpl_fixed_00025_smooth = dpl_fixed_00025.copy().smooth(window_len)
dpl_fixed_000025_smooth = dpl_fixed_000025.copy().smooth(window_len)

#To smooth the dipole from adaptive time step
# 1. handle duplicate timestamps
t_adapt, idx = np.unique(dpl_adaptive.times, return_index=True)
data_adapt = dpl_adaptive.data['agg'][idx]

# 2. interpolate onto a uniform grid
dt_uniform = 0.00025  # 
t_uniform = np.arange(t_adapt[0], t_adapt[-1], dt_uniform)
data_uniform = np.interp(t_uniform, t_adapt, data_adapt)
plt.figure(figsize=(10, 4))
plt.plot(
    t_uniform,
    data_uniform,
    label='Interpolated adaptive time step'
)


# 3. smoothing
from hnn_core import Dipole

dpl_adaptive_uniform = Dipole(t_uniform, data_uniform)  # single column -> treated as 'agg'
dpl_adaptive_smooth = dpl_adaptive_uniform.copy().smooth(window_len)

plt.plot(dpl_adaptive_smooth.times, dpl_adaptive_smooth.data['agg'],
          label='Adaptive (interpolated + smoothed)')




# smoothed dipole
plt.figure(figsize=(10, 4))

plt.plot(
    dpl_fixed_0025.times,
    dpl_fixed_0025_smooth.data['agg'],
    label='Fixed timestep (dt=0.025)'
)

plt.plot(
    dpl_fixed_00025.times,
    dpl_fixed_00025_smooth.data['agg'],
    label='Fixed timestep (dt=0.0025)'
)

plt.plot(
    dpl_fixed_000025.times,
    dpl_fixed_000025_smooth.data['agg'],
    label='Fixed timestep (dt=0.00025)'
)

plt.plot(
    dpl_adaptive_smooth.times,
    dpl_adaptive_smooth.data['agg'],
    label='Adaptive timestep (interpolated + smoothed)'
)

plt.xlabel('Time (ms)')
plt.ylabel('Smoothed dipole (nAm)')
plt.legend()
plt.show()



# plot raster plots
plt.figure(figsize=(10, 6))

gids_0025 = np.array(cell_response_fixed_0025.spike_gids[0])
spike_times_0025 = np.array(cell_response_fixed_0025.spike_times[0])
mask_0025 = gids_0025 < 270

plt.scatter(spike_times_0025[mask_0025],
            gids_0025[mask_0025],
            s=2,
            label='Fixed 0.025')

gids_00025 = np.array(cell_response_fixed_00025.spike_gids[0])
spike_times_00025 = np.array(cell_response_fixed_00025.spike_times[0])
mask_00025 = gids_00025 < 270

plt.scatter(spike_times_00025[mask_00025],
            gids_00025[mask_00025],
            s=2,
            label='Fixed 0.0025')

gids_000025 = np.array(cell_response_fixed_000025.spike_gids[0])
spike_times_000025 = np.array(cell_response_fixed_000025.spike_times[0])
mask_000025 = gids_000025 < 270

plt.scatter(spike_times_000025[mask_000025],
            gids_000025[mask_000025],
            s=2,
            label='Fixed 0.00025')

gids_adaptive = np.array(cell_response_adaptive.spike_gids[0])
spike_times_adaptive = np.array(cell_response_adaptive.spike_times[0])
mask_adaptive = gids_adaptive < 270

plt.scatter(spike_times_adaptive[mask_adaptive],
            gids_adaptive[mask_adaptive],
            s=2,
            label='Adaptive')

# divide cell types
plt.axhline(34.5, linestyle='--', linewidth=1, color='gray')
plt.axhline(134.5, linestyle='--', linewidth=1, color='gray')
plt.axhline(169.5, linestyle='--', linewidth=1, color='gray')
plt.ylim(270, 0)
#plt.gca().invert_yaxis()
plt.xlabel('Time (ms)')
plt.ylabel('GID')
plt.legend()
plt.show()

# to compare the membrane potentials
gid = 146
plt.figure(figsize=(10, 6))
plt.plot(times_0025, vsec_fixed_0025[gid]['soma'], label='Fixed 0.025')
plt.plot(times_00025, vsec_fixed_00025[gid]['soma'], label='Fixed 0.0025')
plt.plot(times_000025, vsec_fixed_000025[gid]['soma'], label='Fixed 0.00025')
plt.plot(times_adaptive, vsec_adaptive[gid]['soma'], label='Adaptive')
plt.xlabel('Time (ms)')
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


