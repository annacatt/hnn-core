#dpl_gabaa = tme.reconstruct_dipole_from_sources(net, sources_syn_GABAA, I_syn_GABAA)import matplotlib.pyplot as plt
import numpy as np
from IPython.core.getipython import get_ipython
from matplotlib.lines import Line2D
import pickle
import postproc_tm_currents_dipole_lfp_csd as tme
#import tm_currents_utils_forLFP as tme_lfp
from hnn_core.extracellular import calculate_csd2d
import Plotting_tools as plott
import matplotlib.pyplot as plt

from hnn_core import (
    JoblibBackend,
    jones_2009_model,
    simulate_dipole,
)
from hnn_core.cells_default import pyramidal
from hnn_core.network_builder import load_custom_mechanisms
from hnn_core.network_models import add_erp_drives_to_jones_model

#net = jones_2009_model()
net = jones_2009_model(use_dataframe=True)
#add_erp_drives_to_jones_model(net)
#net_a = neymotin_2020_model(use_data_frame=False)

''' Satvik's drive
net.add_evoked_drive(
    "evprox",
    mu=40,
    sigma=5,
    numspikes=1,
    location="proximal",
    weights_ampa={
        "L2_pyramidal": 0.01,
        "L5_pyramidal": 0.01,
    },
    synaptic_delays={
        "L2_pyramidal": 0.1,
        "L5_pyramidal": 0.1,
    },
)
'''

tstart = 0.0  

# --- Distal drive ---

net.add_evoked_drive(
    "evdist1",
    mu=63.53 + tstart,
    sigma=3.85,
    numspikes=1,
    weights_ampa={
        "L2_basket": 0.006562,
        "L2_pyramidal": 7e-6,
        "L5_pyramidal": 0.142300,
    },
    weights_nmda={
        "L2_basket": 0.019482,
        "L2_pyramidal": 0.004317,
        "L5_pyramidal": 0.080074,
    },
    location="distal",
    synaptic_delays={"L2_basket": 0.1, "L2_pyramidal": 0.1, "L5_pyramidal": 0.1},
    event_seed=274,
)


# --- Proximal drive 1 ---

net.add_evoked_drive(
    "evprox1",
    mu=26.61 + tstart,
    sigma=2.47,
    numspikes=1,
    weights_ampa={
        "L2_basket": 0.08831,
        "L2_pyramidal": 0.01525,
        "L5_basket": 0.19934,
        "L5_pyramidal": 0.00865,
    },
    weights_nmda=None,
    location="proximal",
    synaptic_delays={
        "L2_basket": 0.1,
        "L2_pyramidal": 0.1,
        "L5_basket": 1.0,
        "L5_pyramidal": 1.0,
    },
    event_seed=544,
)
# --- Proximal drive 2 ---
net.add_evoked_drive(
    "evprox2",
    mu=137.12 + tstart,
    sigma=8.33,
    numspikes=1,
    weights_ampa={
        "L2_basket": 0.000003,
        "L2_pyramidal": 1.438840,
        "L5_basket": 0.008958,
        "L5_pyramidal": 0.684013,
    },
    location="proximal",
    synaptic_delays={
        "L2_basket": 0.1,
        "L2_pyramidal": 0.1,
        "L5_basket": 1.0,
        "L5_pyramidal": 1.0,
    },
    event_seed=814,
)


net.set_cell_positions(inplane_distance=30.)
#net.connectivity.clear() #<- clears everything, including drives
#net.clear_connectivity()  # this keeps drives, removes recurrent

'''
# Scale the maximal conductance of all external drives.
scale = 0.003

for conn in net.connectivity:
    if conn['src_type'] in ('evdist1', 'evprox1', 'evprox2') and \
            conn['receptor'] in ('ampa', 'nmda'):
        conn['nc_dict']['A_weight'] *= scale
'''

# Laminar probe
#depths = np.arange(-625, 2150, 100) what I usually use
#depths = np.arange(-650, 2150, 100) #v1
depths = np.arange(-550, 2150, 100) #v2
#depths = np.arange(-650.5, 2100, 100)
electrode_pos = [(135, 135, z) for z in depths]
net.add_electrode_array('probe1', electrode_pos)

## Add a second probe close to cell gid 226
#soma_xy = np.array([150., 180.])
#new_xy = soma_xy + np.array([15., 15.])   # [165., 195.]

#electrode_pos2 = [(new_xy[0], new_xy[1], z) for z in depths]
#net.add_electrode_array('probe2', electrode_pos2)

#depths3 = np.arange(-625, 2150, 100) #v2
#electrode_pos3 = [(new_xy[0], new_xy[1], z) for z in depths3]
#net.add_electrode_array('probe3', electrode_pos3)


n_trials = 1


'''
for cell_type in ['L2_pyramidal', 'L5_pyramidal']:
    template = net.cell_types[cell_type]['cell_object']
    print(f"\n=== {cell_type} ===")
    for sec_name, sec in template.sections.items():
        for mech_name, params in sec.mechs.items():
            for param_name, value in params.items():
                if param_name == 'gl_hh2':
                    print(f"  {sec_name} / {mech_name} / {param_name}: {value}  <- KEEP (leak)")
                elif param_name.startswith('g'):
                    print(f"  {sec_name} / {mech_name} / {param_name}: {value}  <- ZERO")
'''
'''
def zero_gbar(value):
    """Zero out a conductance value, whether it's a plain scalar or the
    [[positions], [values]] format hnn-core uses for spatially-varying gbar_ar."""
    if isinstance(value, list):
        positions, values = value
        return [positions, [0.0 for _ in values]]
    return 0.0

for cell_type in ['L2_pyramidal', 'L5_pyramidal']:
    template = net.cell_types[cell_type]['cell_object']
    for sec_name, sec in template.sections.items():
        #if sec_name == 'soma':
        #    continue  # keep soma fully active
        for mech_name, params in sec.mechs.items():
            for param_name in list(params.keys()):
                if param_name == 'gl_hh2':
                    continue  # this is the leak conductance -- keep it
                if param_name.startswith('g'):
                    params[param_name] = zero_gbar(params[param_name])
'''


if "dpls" not in locals():
    with JoblibBackend(1):
        dpls = simulate_dipole(
            net,
            tstop=170.0,
            dt=0.00625,#0.0125, # 0.025
            n_trials=n_trials,
            record_agg_i_mem="all",   # aggregated total transmembrane current
            # record_agg_ina="all",
            # record_agg_ik="all",
            record_agg_i_cap="all",   # aggregated capacitive current
            record_ina_hh2="all",
            record_ik_hh2="all",
            record_ik_kca="all",
            record_ik_km="all",
            record_ica_ca="all",
            record_ica_cat="all",
            record_il_hh2="all",      # aggregated leak current
            record_i_ar="all",
            record_isec="all",
            record_vsec="all",
            record_ca="all"
        )

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

scaling_factor = 3000
for dpl in dpls:
    dpl.scale(scaling_factor)

dpl = dpls[0]
times = net.cell_response.times


def _to_numpy_nested(d):
    """Recursively convert every leaf list in a nested dict to a numpy array, in place."""
    for key, value in d.items():
        if isinstance(value, dict):
            _to_numpy_nested(value)
        else:
            d[key] = np.asarray(value, dtype=float)


def convert_currents_to_numpy(net, channels=None, include_isec=True,
                               include_vsec=True, include_ca=True):
    cr = net.cell_response
    cr._times = np.asarray(cr.times, dtype=float)

    channels = channels or list(cr.transmembrane_currents.keys())
    for channel in channels:
        for trial_data in getattr(cr, f"_{channel}"):
            _to_numpy_nested(trial_data)

    for flag, attr in [(include_isec, "_isec"), (include_vsec, "_vsec"), (include_ca, "_ca")]:
        if flag:
            for trial_data in getattr(cr, attr):
                _to_numpy_nested(trial_data)

convert_currents_to_numpy(net, include_isec=True)

results = {
    "net": net,
    "dpl": dpl,
    "dt": net._dt,
    "scaling_factor": scaling_factor,
    "depths": depths,
    "times": times,
}
#with open("runscripts/data/sim_results_dt0025.pkl", "wb") as f:
#with open("runscripts/data/sim_results_dt000625.pkl", "wb") as f:
#with open("runscripts/data/sim_results_dt000625_withmembranepot_newelect_calc_norecconn.pkl", "wb") as f:
#with open("runscripts/data/sim_results_dt000625_withmembranepot_newelect_calc_norecconn_calcium_blocked.pkl", "wb") as f:
#with open("runscripts/data/sim_results_dt000625_withmembranepot_newelect_calc_noreccon_allpassive.pkl", "wb") as f:
#with open("runscripts/data/sim_results_dt000625_withmembranepot_newelect_calc_noreccon_allpassive_excsoma.pkl", "wb") as f:
#with open("runscripts/data/sim_results_dt000625_withmembranepot_newelect_calc_noreccon_allpassive_newprobe2.pkl", "wb") as f:
#with open("runscripts/data/sim_results_dt000625_withmembranepot_newelect_calc_noreccon_allactive_subthreshold.pkl", "wb") as f:
#    pickle.dump(results, f)

with open("runscripts/data/sim_results_dt000625_conn_dataframe_isec.pkl", "wb") as f:
    pickle.dump(results, f)
#with open("runscripts/data/sim_results_dt000625_conn_legacy.pkl", "wb") as f:
#    pickle.dump(results, f)


# ------------

# check if dipole with and without dataframe are the same:
import pickle
import numpy as np
import matplotlib.pyplot as plt

path_with_dataframe = "runscripts/data/sim_results_dt000625_conn_dataframe.pkl"
path_without_dataframe = "runscripts/data/sim_results_dt000625_conn_legacy.pkl"


def load_results(path):
    with open(path, "rb") as f:
        return pickle.load(f)


results_df = load_results(path_with_dataframe)
results_nodf = load_results(path_without_dataframe)

dpl_df = results_df["dpl"]
dpl_nodf = results_nodf["dpl"]

times_df = results_df["times"]
times_nodf = results_nodf["times"]

scaling_df = results_df["scaling_factor"]
scaling_nodf = results_nodf["scaling_factor"]

if scaling_df != scaling_nodf:
    print(f"Warning: different scaling factors ({scaling_df} vs {scaling_nodf}) "
          "-- traces below are NOT on the same scale.")

agg_df = np.asarray(dpl_df.data['agg'])
agg_nodf = np.asarray(dpl_nodf.data['agg'])

# --- Overlay + difference figure ---
fig, (ax_overlay, ax_diff) = plt.subplots(
    2, 1, sharex=True, figsize=(9, 7),
    gridspec_kw={'height_ratios': [3, 1]},
    constrained_layout=True,
)

ax_overlay.plot(times_df, agg_df, label='use_dataframe=True', color='tab:blue', lw=1.2)
ax_overlay.plot(times_nodf, agg_nodf, label='use_dataframe=False', color='tab:red', lw=1.2, alpha=0.8)
ax_overlay.set_ylabel('Dipole moment\n(nAm ' + r'$\times$' + f' {scaling_df:.0f})')
ax_overlay.set_title('Aggregate dipole: use_dataframe True vs. False')
ax_overlay.legend()

# difference panel (only meaningful if both runs share the same time vector)
if np.array_equal(times_df, times_nodf):
    diff = agg_df - agg_nodf
    ax_diff.plot(times_df, diff, color='k', lw=1.0)
    ax_diff.axhline(0, color='gray', ls=':', lw=0.8)
    ax_diff.set_ylabel('Difference')
    print(f"Max abs diff:  {np.max(np.abs(diff)):.4g}")
    print(f"Mean abs diff: {np.mean(np.abs(diff)):.4g}")
    print(f"Allclose:      {np.allclose(agg_df, agg_nodf)}")
else:
    ax_diff.text(0.5, 0.5, "time vectors differ -- can't compute pointwise difference",
                 ha='center', va='center', transform=ax_diff.transAxes)
    print("Time vectors differ in length or values -- skipping difference computation.")

ax_diff.set_xlabel('Time (ms)')

plt.show()

'''
results = {
    "net": net,
    "dpl": dpl,
    "dt": net._dt,
    "scaling_factor": scaling_factor,
    "depths": depths,
    'times': times
}

with open("runscripts/data/sim_results_dt0025.pkl", "wb") as f:
    pickle.dump(results, f)
'''


