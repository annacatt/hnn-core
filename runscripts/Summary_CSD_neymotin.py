"""Summary CSD figure for the Neymotin (jones_2009_model) network, straight from the data.

Makes only the summary figures, saved to runscripts/Figures/<data file name>/:
  1) morphology + Total / Capacitive / Ionic / Synaptic / residual / GABA-B / GABA-A / AMPA / NMDA
     (plott.make_csd_contribution_summary_figure), without and with the electrode-bin
     boundaries on the morphology panel:
       neymotin_summary_csd_contributions.png
       neymotin_summary_csd_contributions_with_slabs.png
  2) the 2x3 "synaptic breakdown" figure
       Total (agg_i_mem) | Synaptic (all) | Inhibitory
       Excitatory        | Recurrent      | External drive
     with its own colour scale per panel and with a fixed scale:
       neymotin_summary_synaptic_breakdown_own_scale.png
       neymotin_summary_synaptic_breakdown_fixed_scale.png
  3) LFP / morphology / CSD of single ionic currents (Na, K, KM, Ca), same plot as the
     per-channel loop of the main script:
       neymotin_csd_ina_hh2.png, neymotin_csd_ik_hh2.png, neymotin_csd_ik_km.png, neymotin_csd_ica_ca.png
  4) cumulative sum of the currents, one panel per step (Na, +K, +KM, +Ca, +other ionic,
     +synaptic, +capacitive = total) next to the total agg_i_mem CSD, on one colour scale:
       neymotin_csd_cumulative_sum.png
  5) CSD (agg_i_mem) of L5 pyramidal cells as cells are progressively added (N = 1, 2, 5, ...),
     each panel scaled to its own 99th percentile (same figure as in single_cell_CSD_analysis.py):
       neymotin_csd_progressive_ncells.png
Same computations as Understand_CSD_contributions_full_network_dataframe_neymotin.py,
without all the other figures. Run from the hnn-core folder (data paths are relative to it).
"""
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

import postproc_tm_currents_dipole_lfp_csd as tme
import Plotting_tools as plott

# ---------------------------------------------------------------- settings
# which simulation to use (control or no-inhibition)
pkl_path = 'runscripts/data/sim_results_dt000625_conn_dataframe_isec_v3_neymotin.pkl'
#pkl_path = 'runscripts/data/sim_results_dt000625_conn_dataframe_isec_v3_neymotin_noinhib.pkl'
#pkl_path = 'runscripts/data/sim_results_dt000625_conn_dataframe_isec_v3_neymotin_norecconn.pkl'

# True: save every figure as PNG in Figures/<data file name>/; False: only show them
save_figures = True

# which figures to make (False = skip that section entirely: no plotting, no saving)
make_figures = {
    "contributions":      False,   # 1) morphology + total/cap/ionic/syn/... (+ version with slabs)
    "synaptic_breakdown": False,   # 2) Total/Synaptic/Inhibitory/Excitatory/Recurrent/External (own + fixed scale)
    "single_channels":    True,   # 3) Na, K, KM, Ca: LFP | morphology | CSD
    "cumulative_sum":     False,   # 4) Na, +K, +KM, +Ca, ... = total
    "progressive_ncells": False,   # 5) L5 CSD as cells are added
    "vsec_sections":      False,   # 6) membrane potential of each section, one panel per section
    "lfp_currents_depth": False,   # 7) morphology | LFP per electrode | main ionic currents per electrode bin
    "vsec_four_ionic":    False,   # 8) morphology | Vm per section | Na, K, KM, Ca of every segment
    "each_ionic_csd":     False,   # 9) CSD of every ionic current, each with its own colour scale
    "whole_cell":         False,   # 10) figures 6 and 8 again, showing the whole neuron (no zoom)
}
# True: open the figure windows at the end; False: close them (useful when only saving)
show_figures = True

contribution_figsize = (22, 11)   # size (inches) of the contribution summary figure; the default is (28, 8)

step = 4                       # downsampling factor (same as the main script)
cell_types = ["L2_pyramidal", "L5_pyramidal"]
drive_names = ["evprox1", "evdist1", "evprox2"]
recurrent_exc = ["L2_pyramidal", "L5_pyramidal"]

# ionic channels of the Neymotin pyramidal cells (everything except capacitive)
ionic_channels = ["ina_hh2", "ik_hh2", "ik_kca", "ik_km",
                  "ica_ca", "ica_cat", "il_hh2", "i_ar"]

# single ionic currents plotted on their own (channel name -> panel title)
single_channels = {
    "ina_hh2": "Na (ina_hh2)",
    "ik_hh2":  "K (ik_hh2)",
    "ik_kca":  "KCa (ik_kca)",
    "ik_km":   "KM (ik_km)",
    "ica_ca":  "Ca (ica_ca)",
    "ica_cat": "CaT (ica_cat)",
    "il_hh2":  "Leak (il_hh2)",
    "i_ar":    "Ih (i_ar)",
}
# spikes overlaid on the CSD of the single-channel figures:
# "hist" = spike-count histograms (as in the main script now), "raster" = spike dots
spike_overlay = "hist"
csd_hist_cell_types = ['L2_pyramidal', 'L5_pyramidal', 'L2_basket', 'L5_basket']
csd_hist_colors = {'L2_pyramidal': 'k', 'L5_pyramidal': 'k',
                   'L2_basket': 'magenta', 'L5_basket': 'magenta'}
csd_hist_scale = "shared"
csd_hist_baseline = {'L2_basket': 'L2_pyramidal', 'L5_basket': 'L5_pyramidal'}

# cumulative sum: components added one after the other, in this order (label, component key).
# keys: the single_channels above, "other_ionic" (= all ionic minus the single channels),
# "synaptic", "capacitive". After the last step the sum is the total CSD (up to the residual).
cumulative_steps = [
    ("Na (ina_hh2)",           "ina_hh2"),
    ("+ K (ik_hh2)",           "ik_hh2"),
    ("+ KM (ik_km)",           "ik_km"),
    ("+ Ca (ica_ca)",          "ica_ca"),
    ("+ other ionic\n(KCa, CaT, leak, Ih)", "other_ionic"),
    ("+ synaptic",             "synaptic"),
    ("+ capacitive\n(= cap + ionic + syn)", "capacitive"),
]
# colour limit of the cumulative figure (µA/mm³); None = largest |CSD| over all its panels
vmax_cumulative = None

# progressive cell-addition figure: cells of this type are added one after the other,
# starting from first_gid, then in gid order; one panel per number of cells in n_cells
progressive_cell_type = "L5_pyramidal"
first_gid = 226
n_cells = [1, 2, 5, 10, 20, 40, 70, None]   # None = all cells of that type

# colour limits of the contribution summary figure (µA/mm³): row 0 = total/cap/ionic/syn, row 1 = synapse types
vmax_row0, vmax_row1 = 60, 10

# colour limits of the fixed-scale synaptic breakdown (µA/mm³)
vmax_fixed = {"Total (agg_i_mem)": 60, "Synaptic (all)": 60, "Inhibitory": 10,
              "Excitatory": 10, "Recurrent": 10, "External drive": 10}

# ---------------------------------------------------------------- figure folder
run_tag = os.path.splitext(os.path.basename(pkl_path))[0]
FIGURES_DIR = os.path.join("/Users/annacattani/Documents/HNN/hnn-core/runscripts/Figures", run_tag)
if save_figures:
    os.makedirs(FIGURES_DIR, exist_ok=True)

def save_fig(name, fig=None, dpi=150):
    """Save a figure as Figures/<run_tag>/<name>.png (current figure if fig is None).
    Does nothing when save_figures is False."""
    if not save_figures:
        return
    fig = fig if fig is not None else plt.gcf()
    path = os.path.join(FIGURES_DIR, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print("Saved", path)

# ---------------------------------------------------------------- load + downsample
print("Loading", pkl_path)
with open(pkl_path, 'rb') as f:
    results = pickle.load(f)

net = results['net']
times = results['times']
depths = results['depths']
contact_labels = np.asarray(depths, dtype=int)

tme.downsample_currents(net, step=step, include_isec=True, include_vsec=True, include_ca=True)
times = times[::step]
times_ = times[1:]

# ---------------------------------------------------------------- CSD from sources
def compute_lfp_csd_from_sources(net, sources, I, array_name="probe1"):
    B, V_bin, _ = tme.build_binning_matrix_for_sources(net, sources, array_name=array_name)
    csd = tme.compute_csd_from_sources(B, I, V_bin)
    T = tme.build_transfer_resistance_matrix_for_sources(net, sources, array_name=array_name)
    lfp = tme.reconstruct_lfp_from_sources(T, I)
    return lfp[:, 1:], csd[:, 1:]

# total membrane current
sources_agg, I_agg = tme.collect_intrinsic_sources(
    net, trial_idx=0, cell_types=cell_types, channels=["agg_i_mem"])
_, csd_from_sources_ = compute_lfp_csd_from_sources(net, sources_agg, I_agg)

# capacitive and ionic currents
sources_cap, I_cap = tme.collect_intrinsic_sources(
    net, trial_idx=0, cell_types=cell_types, channels=["agg_i_cap"])
_, csd_cap_ = compute_lfp_csd_from_sources(net, sources_cap, I_cap)

sources_ionic, I_ionic = tme.collect_intrinsic_sources(
    net, trial_idx=0, cell_types=cell_types, channels=ionic_channels)
_, csd_ionic_ = compute_lfp_csd_from_sources(net, sources_ionic, I_ionic)

# single ionic currents (taken from the ionic sources collected above)
if make_figures["single_channels"] or make_figures["cumulative_sum"]:
    lfp_csd_single = {}
    for ch in single_channels:
        sources_ch, I_ch = tme.filter_sources(sources_ionic, I_ionic, labels=[ch])
        lfp_csd_single[ch] = compute_lfp_csd_from_sources(net, sources_ch, I_ch)   # (lfp_, csd_)

# all synaptic currents
sources_syn, I_syn = tme.collect_synaptic_sources(net, trial_idx=0, cell_types=cell_types)
_, csd_syn_ = compute_lfp_csd_from_sources(net, sources_syn, I_syn)

def syn_csd(syn_names, presyn_types=None, label=""):
    """CSD of a subset of the synaptic sources; zeros if the subset is empty
    (e.g. GABA synapses in a simulation with inhibition removed)."""
    kwargs = dict(syn_names=syn_names)
    if presyn_types is not None:
        kwargs["presyn_types"] = presyn_types
    sources, I = tme.filter_sources(sources_syn, I_syn, **kwargs)
    if len(sources) == 0:
        print(f"  no sources for {label or syn_names} -> CSD set to zero")
        return np.zeros_like(csd_syn_)
    return compute_lfp_csd_from_sources(net, sources, I)[1]

csd_syn_gabaa_ = syn_csd(["gabaa"], label="GABA-A")
csd_syn_gabab_ = syn_csd(["gabab"], label="GABA-B")
csd_syn_ampa_ = syn_csd(["ampa"], label="AMPA")
csd_syn_nmda_ = syn_csd(["nmda"], label="NMDA")
csd_syn_ampa_drives_ = syn_csd(["ampa"], drive_names, "AMPA from drives")
csd_syn_nmda_drives_ = syn_csd(["nmda"], drive_names, "NMDA from drives")
csd_syn_ampa_recurrent_ = syn_csd(["ampa"], recurrent_exc, "AMPA recurrent")
csd_syn_nmda_recurrent_ = syn_csd(["nmda"], recurrent_exc, "NMDA recurrent")

csd_syn_exc_ = csd_syn_ampa_ + csd_syn_nmda_          # excitatory synaptic CSD
csd_syn_inh_ = csd_syn_gabaa_ + csd_syn_gabab_        # inhibitory synaptic CSD (all recurrent)
# GABA synapses only come from L2_basket/L5_basket: no external-drive part for inhibition,
# so "Recurrent" is recurrent EXCITATION only (as in the main script)
csd_recurrent_ = csd_syn_ampa_recurrent_ + csd_syn_nmda_recurrent_
csd_external_ = csd_syn_ampa_drives_ + csd_syn_nmda_drives_

# sanity checks (should be ~0, float noise only)
print("max abs diff (recurrent+external vs excitatory):",
      np.max(np.abs((csd_recurrent_ + csd_external_) - csd_syn_exc_)))
print("max abs diff (exc+inh vs all synaptic):",
      np.max(np.abs((csd_syn_exc_ + csd_syn_inh_) - csd_syn_)))
print("max abs residual Total-(cap+ionic+syn):",
      np.max(np.abs(csd_from_sources_ - (csd_cap_ + csd_ionic_ + csd_syn_))))

panels_summary = [
    ("Total (agg_i_mem)", csd_from_sources_),
    ("Synaptic (all)",    csd_syn_),
    ("Inhibitory",        csd_syn_inh_),
    ("Excitatory",        csd_syn_exc_),
    ("Recurrent",         csd_recurrent_),
    ("External drive",    csd_external_),
]

# ---------------------------------------------------------------- figures
# 1) contribution summary: morphology + total / cap / ionic / syn / residual / synapse types
if make_figures["contributions"]:
    def contribution_figure():
        fig = plott.make_csd_contribution_summary_figure(
            net, contact_labels, times_,
            csd_from_sources_, csd_cap_, csd_ionic_, csd_syn_,
            csd_syn_gabab_, csd_syn_gabaa_, csd_syn_ampa_, csd_syn_nmda_,
            vmax_row0=vmax_row0, vmax_row1=vmax_row1,
            suptitle="Neymotin - Whole network (all pyramidal cells) - from sources",
        )
        fig.set_size_inches(*contribution_figsize)   # the function creates it at (28, 8): very wide and slim
        return fig

    fig_all = contribution_figure()
    save_fig("neymotin_summary_csd_contributions", fig_all)

    # same, with the slab (electrode-bin) boundaries on the morphology panel
    z_edges = tme._z_edges_from_array(net, "probe1")
    fig_all_slabs = contribution_figure()
    ax_morph = next(ax for ax in fig_all_slabs.axes if ax.get_title() == "Morphology")
    for z in z_edges:
        ax_morph.axhline(z, color='red', alpha=0.4, lw=1)
    save_fig("neymotin_summary_csd_contributions_with_slabs", fig_all_slabs)

# 2) synaptic breakdown
if make_figures["synaptic_breakdown"]:
    def summary_figure(own_scale, figsize):
        fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
        for ax, (title, data) in zip(axes.flat, panels_summary):
            if own_scale:
                vmax_panel = max(np.max(np.abs(data)), 1e-12)   # untruncated, symmetric -> green = 0
            else:
                vmax_panel = vmax_fixed[title]
            plott.plot_laminar_csd_AC(
                times_, data, contact_labels,
                ax=ax, vmin=-vmax_panel, vmax=vmax_panel,
                overlay_csd_traces=True,
                unit_csd="µA/mm³",
                sink="red",
                colorbar=False,
                show=False,
            )
            ax.set_title(title)
            fig.colorbar(ax.collections[-1], ax=ax, pad=0.02).set_label("CSD (µA/mm³)")
        fig.suptitle("Whole network (all pyramidal cells) - from sources")
        return fig

    fig_own = summary_figure(own_scale=True, figsize=(18, 10))
    save_fig("neymotin_summary_synaptic_breakdown_own_scale", fig_own)

    fig_fixed = summary_figure(own_scale=False, figsize=(15, 8.3))
    save_fig("neymotin_summary_synaptic_breakdown_fixed_scale", fig_fixed)

# 3) single ionic currents: LFP | morphology | CSD (+ spikes), own symmetric colour scale
if make_figures["single_channels"]:
    for ch, title in single_channels.items():
        lfp_, csd_ = lfp_csd_single[ch]
        vmax = np.max(np.abs(csd_))
        # LFP scale bar sized to this channel's LFP (a fixed 200 µV bar becomes huge for small currents)
        lfp_ptp = np.max(np.ptp(lfp_, axis=1))
        lfp_bar = None
        if lfp_ptp > 0:
            k = 10 ** np.floor(np.log10(lfp_ptp / 2))
            lfp_bar = max(m * k for m in (1, 2, 5) if m * k <= lfp_ptp / 2)   # round value ~half the largest deflection
            if lfp_bar < 1:
                lfp_bar = None   # the LFP panel labels the bar in whole µV (it would read "0 µV"): no bar below 1 µV
        overlay_kwargs = dict(overlay_raster_on_csd=True, spike_overlay=spike_overlay)
        if spike_overlay == "hist":
            overlay_kwargs.update(
                hist_bin_ms=1.0,           # bin width (ms)
                hist_height_um=1000.0,     # height of the tallest bar on the depth axis (µm)
                raster_cell_types=csd_hist_cell_types,
                hist_color=csd_hist_colors,
                hist_scale=csd_hist_scale,
                hist_baseline=csd_hist_baseline,
            )
        fig = plott.plot_lfp_morph_csd(
            times_, lfp_, csd_, contact_labels, net,
            ext_inputs=net.cell_response,
            spike_types={'Distal': ['evdist'], 'Proximal': ['evprox']},
            scale_lfp=3.0,
            voltage_scalebar=lfp_bar,
            vmin=-vmax,
            vmax=vmax,
            figsize=(18, 6),
            overlay_csd_traces=True,
            unit_csd="µA/mm³",
            sink="r",
            show=False,
            **overlay_kwargs,
        )
        fig.suptitle(f"LFP/CSD from {title}")
        save_fig(f"neymotin_csd_{ch}", fig)

# 4) cumulative sum of the currents (CSD is linear in the currents, so CSDs simply add up)
if make_figures["cumulative_sum"]:
    components = {ch: lfp_csd_single[ch][1] for ch in single_channels}
    components["other_ionic"] = csd_ionic_ - sum(components[ch] for ch in single_channels)
    components["synaptic"] = csd_syn_
    components["capacitive"] = csd_cap_

    cumulative_panels = []
    running = np.zeros_like(csd_from_sources_)
    for label, key in cumulative_steps:
        running = running + components[key]
        cumulative_panels.append((label, running.copy()))
    cumulative_panels.append(("Total (agg_i_mem)", csd_from_sources_))   # reference

    vmax_cum = vmax_cumulative
    if vmax_cum is None:
        vmax_cum = max(np.max(np.abs(data)) for _, data in cumulative_panels)

    n_panels = len(cumulative_panels)
    n_cols = 4
    n_rows = int(np.ceil(n_panels / n_cols))
    fig_cum, axes_cum = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows),
                                     constrained_layout=True, squeeze=False)
    for ax, (label, data) in zip(axes_cum.flat, cumulative_panels):
        plott.plot_laminar_csd_AC(
            times_, data, contact_labels,
            ax=ax, vmin=-vmax_cum, vmax=vmax_cum,
            overlay_csd_traces=True,
            unit_csd="µA/mm³",
            sink="red",
            colorbar=False,
            show=False,
        )
        ax.set_title(label)
    for ax in list(axes_cum.flat)[n_panels:]:    # unused grid cells
        ax.set_visible(False)
    fig_cum.colorbar(axes_cum.flat[0].collections[-1], ax=axes_cum, pad=0.01,
                     shrink=0.8).set_label("CSD (µA/mm³)")
    fig_cum.suptitle("Neymotin - cumulative sum of the CSD contributions (all pyramidal cells)")
    save_fig("neymotin_csd_cumulative_sum", fig_cum)

# 5) progressive cell addition: total membrane current (agg_i_mem) of the first N cells
if make_figures["progressive_ncells"]:
    gids_all = sorted(net.gid_ranges[progressive_cell_type])
    assert first_gid in gids_all, f"gid {first_gid} is not a {progressive_cell_type}"
    ordered_gids = [first_gid] + [g for g in gids_all if g != first_gid]
    n_list = [len(ordered_gids) if n is None else min(n, len(ordered_gids)) for n in n_cells]

    sources_agg_type, I_agg_type = tme.filter_sources(
        sources_agg, I_agg, cell_types=[progressive_cell_type])

    def csd_for_gids(gids):
        sources_g, I_g = tme.filter_sources(sources_agg_type, I_agg_type, gid_subset=gids)
        B, V_bin, _ = tme.build_binning_matrix_for_sources(net, sources_g, array_name="probe1")
        csd = tme.compute_csd_from_sources(B, I_g, V_bin)
        return csd[:, 1:]  # drop first sample, matches times_

    n_cols_prog = 4
    n_rows_prog = int(np.ceil(len(n_list) / n_cols_prog))
    fig_prog, axes_prog = plt.subplots(n_rows_prog, n_cols_prog, figsize=(22, 4.5 * n_rows_prog),
                                       constrained_layout=True, sharex=True, sharey=True,
                                       squeeze=False)
    for ax, n in zip(axes_prog.flat, n_list):
        csd_n = csd_for_gids(ordered_gids[:n])
        vmax_n = max(np.percentile(np.abs(csd_n), 99), 1e-12)  # per-panel scaling -- shows shape, not magnitude
        plott.plot_laminar_csd_AC(
            times_, csd_n, contact_labels,
            ax=ax, vmin=-vmax_n, vmax=vmax_n,
            overlay_csd_traces=False, unit_csd="µA/mm³", sink="red", show=False,
            colorbar=True,
        )
        ax.set_title(f"N = {n} cell{'s' if n > 1 else ''}")
    for ax in list(axes_prog.flat)[len(n_list):]:
        ax.set_visible(False)
    fig_prog.suptitle(f"{progressive_cell_type}: CSD as cells are progressively added "
                      f"(each panel independently scaled) - {run_tag}", fontsize=13)
    save_fig("neymotin_csd_progressive_ncells", fig_prog)

# here
# ==================== settings shared by figures 6, 7 and 8 ====================
# each figure is made once per group: a single cell, or a population whose currents are summed
# (all cells of a group must be of the same type, so they share morphology and depths)
cell_groups = {
    "gid226":   [226],                                     # one L5 pyramidal cell
    "allL5pyr": list(net.gid_ranges["L5_pyramidal"]),      # all 100 L5 pyramidal cells
}
zoom_depth = (1800, 2400)    # depth range to show (µm from the top contact), or None for the whole probe
vm_scale_um_per_mV = 0.8     # 6, 8) vertical scale of the Vm traces: µm of depth per mV (100 mV -> 80 µm)
vm_scalebar_mV = 100         # 6, 8) size of the voltage scale bar (mV)
vm_baseline_ms = 5.0         # 6, 8) each Vm trace is drawn relative to its mean over the first ms
vm_show_individual = True    # 6, 8) groups of several cells: also draw every cell's Vm (thin grey) under the mean
lfp_gain = 0.8               # 7) 1 = largest LFP deflection (in view) spans 0.8 x contact spacing
# 6, 7) net ionic current of every segment (sum of all ionic channels, nA), solid line
net_ionic_color = "tab:red"
# 6, 7) synaptic currents (nA), dashed lines
show_syn_currents = True
syn_per_receptor = True      # True: one line per receptor; False: one net synaptic line (sum of all receptors)
syn_receptors = {"ampa": "AMPA", "nmda": "NMDA", "gabaa": "GABA-A", "gabab": "GABA-B"}
syn_colors = {"ampa": "tab:purple", "nmda": "tab:brown", "gabaa": "tab:cyan", "gabab": "tab:olive"}
net_syn_color = "tab:blue"   # colour of the net synaptic line (syn_per_receptor = False)
# 8) the four main ionic currents of every segment (nA), solid lines, no synaptic currents
four_ionic = {"ina_hh2": "Na", "ik_hh2": "K", "ik_km": "KM", "ica_ca": "Ca"}
four_ionic_colors = {"ina_hh2": "tab:red", "ik_hh2": "tab:blue",
                     "ik_km": "tab:green", "ica_ca": "tab:orange"}
# 8) also draw the net ionic current (sum of ALL ionic channels) on top of the four currents:
#    "population" = only for groups of several cells, "all" = also for single cells, None = never
four_ionic_add_net = "population"
four_ionic_net_color = "k"
current_scale_um_per_nA = None   # None: automatic (see current_gain); or a fixed value in µm per nA
current_gain = 0.8               # 1 = largest current in view spans 0.8 x contact spacing (per figure)

# ==================== helpers ====================
spacing = np.median(np.diff(np.sort(contact_labels)))
z_surface = np.max(contact_labels)                       # depth 0 = top contact, as on the axis labels
if zoom_depth is not None:
    z_lo, z_hi = z_surface - max(zoom_depth), z_surface - min(zoom_depth)   # HNN z of the zoom window
else:
    z_lo, z_hi = -np.inf, np.inf

def nice_bar(x):
    """Round number (1, 2, 5 x 10^k) not larger than x, for scale bars."""
    k = 10 ** np.floor(np.log10(x))
    return max(m * k for m in (1, 2, 5) if m * k <= x)

def group_geometry(gids):
    """Cell type of the group and a function giving (x, z) of a point along a section, in the
    morphology-plot coordinates of the first cell (x relative to its soma, z = HNN z), as in
    plott._draw_cell_morphology. All cells of the group must share type (and thus depths)."""
    types = {net.gid_to_type(g) for g in gids}
    assert len(types) == 1, f"a group must contain one cell type, got {types}"
    cell_type = types.pop()
    template_sections = net.cell_types[cell_type]["cell_object"].sections
    soma_pos = np.asarray(net.pos_dict[cell_type][gids[0] - net.gid_ranges[cell_type][0]], dtype=float)
    def section_point(sec_name, xn):
        p0, p1 = np.asarray(template_sections[sec_name]._end_pts, dtype=float)
        p = p0 + xn * (p1 - p0)
        return p[0], p[2] + soma_pos[2]
    return cell_type, section_point

def segment_currents(gids, section_point, mode="net", add_net=False):
    """Currents of a group of cells, summed over the cells, one entry per (segment, current):
    (section, segment x, kind, name, trace nA)
    mode = "net":  kind "ionic", name "net"  = sum of all ionic channels of the segment,
                   plus kind "syn" (name = receptor, or "net" if syn_per_receptor is False)
    mode = "four": kind "ionic", name = channel, for the channels in four_ionic (no synaptic);
                   with add_net=True also kind "ionic", name "net" (sum of ALL ionic channels)
    Also returns the (x, z) of every segment centre and the current scale (µm per nA)."""
    sums = {}                                            # (section, x, kind, name) -> summed trace
    def add(key, trace):
        sums[key] = sums.get(key, 0) + trace
    src_c, I_c = tme.filter_sources(sources_ionic, I_ionic, gid_subset=gids)
    for k, src in enumerate(src_c):
        if mode == "four" and src.label in four_ionic:
            add((src.section, src.segment_x, "ionic", src.label), I_c[k])
        if mode == "net" or add_net:
            add((src.section, src.segment_x, "ionic", "net"), I_c[k])
    if mode == "net" and show_syn_currents:
        src_s, I_s = tme.filter_sources(sources_syn, I_syn, gid_subset=gids)
        for k, src in enumerate(src_s):
            # syn_name = "{source}_{section}_{receptor}_{segX}", e.g. "L5_basket_soma_gabaa_0.5"
            receptor = next((tok for tok in str(src.syn_name).split("_") if tok in syn_receptors), None)
            if receptor is not None:
                add((src.section, src.segment_x, "syn", receptor if syn_per_receptor else "net"), I_s[k])
    I_seg = [(sec, xs, kind, name, tr) for (sec, xs, kind, name), tr in sums.items()]

    seg_xz = {(sec, xs): section_point(sec, xs) for sec, xs, *_ in I_seg}
    in_view = [tr for sec, xs, _, _, tr in I_seg if z_lo <= seg_xz[(sec, xs)][1] <= z_hi]
    i_max = max(np.max(np.abs(tr)) for tr in (in_view or [tr for *_, tr in I_seg]))
    i_scale = current_scale_um_per_nA or (current_gain * 0.8 * spacing / i_max)
    return I_seg, seg_xz, i_scale

def draw_morphology(ax_m, gids, cell_type, seg_xz):
    """Morphology of the first cell of the group (sets the shared depth axis), segment dots, zoom."""
    plott.plot_cell_morphology_for_lfp_csd(
        net, contact_labels, cell_types=(cell_type,), gid=gids[0], ax=ax_m, show=False)
    if len(gids) > 1:
        ax_m.set_title(f"{cell_type}\n(all {len(gids)} cells; drawn: gid {gids[0]})", fontsize=10)
    for x_s, z_s in seg_xz.values():
        ax_m.plot(x_s, z_s, "o", ms=2.5, color="k", zorder=5)
    if zoom_depth is not None:                           # after the morphology call, which sets its own ylim
        ax_m.set_ylim(z_lo, z_hi)

def draw_vm(ax_v, gids, section_point):
    """Vm of every section at its midpoint depth: the cell's Vm (one cell) or the mean over the
    group (black) with every cell in thin grey underneath (vm_show_individual). mV scale bar."""
    vsec = [net.cell_response.vsec[0][g] for g in gids]  # one {section: Vm trace} per cell (trial 0)
    for sec in sorted(vsec[0].keys(), key=lambda s: section_point(s, 0.5)[1], reverse=True):
        vms = np.array([np.asarray(v[sec], dtype=float) for v in vsec])   # (n_cells, n_times)
        t = times[:vms.shape[1]]
        early = t <= vm_baseline_ms
        baseline = vms[:, early].mean(axis=1, keepdims=True) if np.any(early) else vms[:, :1]
        _, z_mid = section_point(sec, 0.5)
        y = z_mid + vm_scale_um_per_mV * (vms - baseline)
        if len(gids) > 1 and vm_show_individual:
            ax_v.plot(t, y.T, color="0.75", lw=0.3)
        ax_v.plot(t, y.mean(axis=0), color="k", lw=0.9)
        if z_lo <= z_mid <= z_hi:                        # label only the sections in view
            ax_v.text(t[-1] + 1, z_mid, sec, fontsize=7, va="center")
    y_top = ax_v.get_ylim()[1]
    bar_len = vm_scalebar_mV * vm_scale_um_per_mV
    ax_v.plot([times[0] + 2] * 2, [y_top - 30 - bar_len, y_top - 30], color="k", lw=2)
    ax_v.text(times[0] + 4, y_top - 30 - bar_len / 2, f"{vm_scalebar_mV} mV", va="center", fontsize=8)
    ax_v.set_xlim(times[0], times[-1] + 15)              # room for the section labels
    what = "Vm" if len(gids) == 1 else f"mean Vm of {len(gids)} cells" + (" (grey: each cell)" if vm_show_individual else "")
    ax_v.set_title(f"{what} per section (section midpoint),\nat the section's depth", fontsize=10)

def draw_segment_currents(ax_c, I_seg, seg_xz, i_scale, title, net_color=None):
    """Currents of every segment at its depth: ionic solid (net and/or per channel), synaptic dashed;
    legend and nA scale bar. net_color: colour of the net ionic line (default net_ionic_color);
    the net line is drawn last, on top of the per-channel lines."""
    net_color = net_color or net_ionic_color
    for sec, xs, kind, name, i_tr in sorted(I_seg, key=lambda e: e[2] == "ionic" and e[3] == "net"):
        _, z_s = seg_xz[(sec, xs)]
        if kind == "ionic":
            is_net = name == "net"
            ax_c.plot(times[:len(i_tr)], z_s + i_scale * i_tr,
                      color=net_color if is_net else four_ionic_colors[name], lw=0.9 if is_net else 0.7)
        else:
            color = net_syn_color if name == "net" else syn_colors[name]
            ax_c.plot(times[:len(i_tr)], z_s + i_scale * i_tr, color=color, lw=0.8, ls="--")
    # zero line of every segment (the depth each current trace is drawn around)
    for _, z_s in seg_xz.values():
        ax_c.axhline(z_s, color="k", lw=0.5, ls=":", zorder=3)
    present = {(kind, name) for _, _, kind, name, _ in I_seg}
    handles = []
    if ("ionic", "net") in present:
        handles.append(plt.Line2D([], [], color=net_color, label="net ionic"))
    handles += [plt.Line2D([], [], color=four_ionic_colors[ch], label=lab)
                for ch, lab in four_ionic.items() if ("ionic", ch) in present]
    if ("syn", "net") in present:
        handles.append(plt.Line2D([], [], color=net_syn_color, ls="--", label="net synaptic"))
    handles += [plt.Line2D([], [], color=syn_colors[r], ls="--", label=lab)
                for r, lab in syn_receptors.items() if ("syn", r) in present]
    ax_c.legend(handles=handles, loc="upper right", fontsize=8, ncol=len(handles), frameon=False)
    y_top = ax_c.get_ylim()[1]
    i_bar = nice_bar(0.5 * spacing / i_scale)            # round scale bar ~half a contact spacing
    ax_c.plot([times[0] + 2] * 2, [y_top - 30 - i_bar * i_scale, y_top - 30], color="k", lw=2)
    ax_c.text(times[0] + 4, y_top - 30 - i_bar * i_scale / 2, f"{i_bar:g} nA", va="center", fontsize=8)
    ax_c.set_xlim(times[0], times[-1])
    ax_c.set_title(title, fontsize=10)

def finish_trace_axes(axes):
    for ax in axes:
        for z in contact_labels:
            ax.axhline(z, color="lightgray", lw=0.3, zorder=0)
        ax.set_xlabel("time (ms)")
        ax.tick_params(axis="y", labelleft=False)
        ax.spines[["top", "right"]].set_visible(False)

def three_panel_figure():
    return plt.subplots(1, 3, figsize=(20, 8), sharey=True, constrained_layout=True,
                        gridspec_kw={"width_ratios": [1, 2.5, 2.5]})

def group_text(gids, cell_type):
    """'gid=226' for one cell, 'all 100 L5_pyramidal cells (summed)' for a group."""
    return f"gid={gids[0]}" if len(gids) == 1 else f"all {len(gids)} {cell_type} cells (summed)"

zoom_tag = "" if zoom_depth is None else f"_depth{min(zoom_depth)}-{max(zoom_depth)}"
syn_txt = " + synaptic (dashed)" if show_syn_currents else ""

# ==================== 6) morphology | Vm per section | net ionic (+ synaptic) currents of every segment ====================
if make_figures["vsec_sections"]:
    for tag, gids in cell_groups.items():
        cell_type, section_point = group_geometry(gids)
        I_seg, seg_xz, i_scale = segment_currents(gids, section_point, mode="net")
        fig_v, (ax_m, ax_v, ax_i) = three_panel_figure()
        draw_morphology(ax_m, gids, cell_type, seg_xz)
        draw_vm(ax_v, gids, section_point)
        draw_segment_currents(ax_i, I_seg, seg_xz, i_scale,
                              f"net ionic{syn_txt} currents of every segment,\n"
                              f"{group_text(gids, cell_type)}, at the segment's depth (nA; inward < 0)")
        finish_trace_axes((ax_v, ax_i))
        fig_v.suptitle(f"{cell_type} {group_text(gids, cell_type)}: membrane potential per section "
                       f"and net ionic / synaptic currents of every segment")
        save_fig(f"neymotin_vsec_morph_{tag}{zoom_tag}", fig_v)

# ==================== 7) morphology | LFP of the group per electrode | net ionic (+ synaptic) currents of every segment ====================
if make_figures["lfp_currents_depth"]:
    contact_z = np.asarray(net.rec_arrays["probe1"].positions, dtype=float)[:, 2]
    for tag, gids in cell_groups.items():
        cell_type, section_point = group_geometry(gids)
        I_seg, seg_xz, i_scale = segment_currents(gids, section_point, mode="net")

        # extracellular potential of these cells only, at each contact (µV), from their agg_i_mem
        src_grp_agg, I_grp_agg = tme.filter_sources(sources_agg, I_agg, gid_subset=gids)
        lfp_grp, _ = compute_lfp_csd_from_sources(net, src_grp_agg, I_grp_agg)   # (n_contacts, len(times_))
        lfp_grp = lfp_grp - lfp_grp[:, :1]                        # each trace relative to its first sample
        lfp_in_view = lfp_grp[(contact_z >= z_lo) & (contact_z <= z_hi)]
        lfp_max = np.max(np.abs(lfp_in_view if lfp_in_view.size else lfp_grp))
        lfp_scale = lfp_gain * 0.8 * spacing / lfp_max            # µm per µV

        fig_l, (ax_m, ax_l, ax_c) = three_panel_figure()
        draw_morphology(ax_m, gids, cell_type, seg_xz)

        # middle: LFP of the group at each contact depth (black)
        for k, zc in enumerate(contact_z):
            ax_l.plot(times_, zc + lfp_scale * lfp_grp[k], color="k", lw=0.8)
        y_top = ax_l.get_ylim()[1]
        lfp_bar = nice_bar(0.5 * spacing / lfp_scale)             # bar about half a spacing tall
        ax_l.plot([times_[0] + 2] * 2, [y_top - 30 - lfp_bar * lfp_scale, y_top - 30], color="k", lw=2)
        ax_l.text(times_[0] + 4, y_top - 30 - lfp_bar * lfp_scale / 2, f"{lfp_bar:g} µV",
                  va="center", fontsize=8)
        ax_l.set_xlim(times[0], times[-1])
        ax_l.set_title(f"extracellular potential of {group_text(gids, cell_type)} only,\n"
                       f"from their agg_i_mem", fontsize=10)

        draw_segment_currents(ax_c, I_seg, seg_xz, i_scale,
                              f"net ionic{syn_txt} currents of every segment,\n"
                              f"{group_text(gids, cell_type)}, at the segment's depth (nA, inward < 0)")
        finish_trace_axes((ax_l, ax_c))
        fig_l.suptitle(f"{cell_type} {group_text(gids, cell_type)}: extracellular potential at each electrode "
                       f"and net ionic / synaptic currents of every segment")
        save_fig(f"neymotin_lfp_currents_depth_{tag}{zoom_tag}", fig_l)

# ==================== 8) morphology | Vm per section | four main ionic currents (+ net ionic) of every segment ====================
if make_figures["vsec_four_ionic"]:
    for tag, gids in cell_groups.items():
        cell_type, section_point = group_geometry(gids)
        add_net = four_ionic_add_net == "all" or (four_ionic_add_net == "population" and len(gids) > 1)
        I_seg, seg_xz, i_scale = segment_currents(gids, section_point, mode="four", add_net=add_net)
        fig_4, (ax_m, ax_v, ax_i) = three_panel_figure()
        draw_morphology(ax_m, gids, cell_type, seg_xz)
        draw_vm(ax_v, gids, section_point)
        net_txt = " (black: net ionic, all channels)" if add_net else ""
        draw_segment_currents(ax_i, I_seg, seg_xz, i_scale,
                              f"Na, K, KM and Ca currents of every segment{net_txt},\n"
                              f"{group_text(gids, cell_type)}, at the segment's depth (nA; inward < 0)",
                              net_color=four_ionic_net_color)
        finish_trace_axes((ax_v, ax_i))
        fig_4.suptitle(f"{cell_type} {group_text(gids, cell_type)}: membrane potential per section "
                       f"and main ionic currents of every segment")
        save_fig(f"neymotin_vsec_four_ionic_{tag}{zoom_tag}", fig_4)


# ==================== 9) CSD of every ionic current, each with its own colour scale ====================
if make_figures.get("each_ionic_csd", True):
    ionic_csd_labels = {                     # channel -> panel title
        "ina_hh2": "Na (ina_hh2)",
        "ik_hh2":  "K (ik_hh2)",
        "ik_kca":  "KCa (ik_kca)",
        "ik_km":   "KM (ik_km)",
        "ica_ca":  "Ca (ica_ca)",
        "ica_cat": "CaT (ica_cat)",
        "il_hh2":  "Leak (il_hh2)",
        "i_ar":    "Ih (i_ar)",
    }
    ionic_csd_vmax = "max"                   # "max": ±largest |CSD| of each panel; or a number, e.g. 99 = 99th percentile
    ionic_csd_ncols = 4

    # CSD of each channel (from the ionic sources collected at the top of the script)
    csd_by_channel = {}
    for ch in ionic_csd_labels:
        src_ch, I_ch = tme.filter_sources(sources_ionic, I_ionic, labels=[ch])
        if len(src_ch) == 0:
            print(f"  {ch}: no sources, skipped")
            continue
        csd_by_channel[ch] = compute_lfp_csd_from_sources(net, src_ch, I_ch)[1]

    # print: peak sink / source of each channel (sink = negative CSD), with time and depth
    z_top = np.max(contact_labels)
    print(f"\nIonic CSD per channel ({run_tag}), µA/mm³:")
    print(f"{'channel':16s} {'max |CSD|':>10s} {'min (sink)':>11s} {'t (ms)':>7s} {'depth':>6s}"
          f" {'max (source)':>13s} {'t (ms)':>7s} {'depth':>6s}")
    for ch, csd in csd_by_channel.items():
        i_min = np.unravel_index(np.argmin(csd), csd.shape)
        i_max = np.unravel_index(np.argmax(csd), csd.shape)
        print(f"{ch:16s} {np.max(np.abs(csd)):10.3f} "
              f"{csd[i_min]:11.3f} {times_[i_min[1]]:7.1f} {z_top - contact_labels[i_min[0]]:6.0f} "
              f"{csd[i_max]:13.3f} {times_[i_max[1]]:7.1f} {z_top - contact_labels[i_max[0]]:6.0f}")
    check = np.max(np.abs(sum(csd_by_channel.values()) - csd_ionic_))
    print(f"check: max |sum of channels - total ionic CSD| = {check:.2e} (should be ~0)\n")

    # plot: one panel per channel, each with its own colour scale and colorbar
    n_panels = len(csd_by_channel)
    n_rows = int(np.ceil(n_panels / ionic_csd_ncols))
    fig_ion, axes_ion = plt.subplots(n_rows, ionic_csd_ncols, figsize=(5.5 * ionic_csd_ncols, 5 * n_rows),
                                     constrained_layout=True, squeeze=False)
    for ax, (ch, csd) in zip(axes_ion.flat, csd_by_channel.items()):
        if ionic_csd_vmax == "max":
            vmax_ch = np.max(np.abs(csd))
        else:
            vmax_ch = np.percentile(np.abs(csd), ionic_csd_vmax)
        vmax_ch = max(vmax_ch, 1e-12)                       # symmetric -> green = 0
        plott.plot_laminar_csd_AC(
            times_, csd, contact_labels,
            ax=ax, vmin=-vmax_ch, vmax=vmax_ch,
            overlay_csd_traces=False,
            unit_csd="µA/mm³",
            sink="red",
            colorbar=True,
            show=False,
        )
        ax.set_title(f"{ionic_csd_labels[ch]}\n(±{vmax_ch:.3g} µA/mm³)", fontsize=10)
    for ax in list(axes_ion.flat)[n_panels:]:               # unused grid cells
        ax.set_visible(False)
    fig_ion.suptitle(f"Neymotin - CSD of every ionic current (all pyramidal cells), own colour scale per panel - {run_tag}")
    save_fig("neymotin_csd_each_ionic_channel", fig_ion)

# ==================== 10) figures 6 and 8 again, for the whole neuron (no zoom) ====================
if make_figures.get("whole_cell", True):
    whole_cell_figsize = (20, 14)            # taller: the whole cell spans ~1600 µm instead of 600
    whole_cell_groups = cell_groups          # same groups as figures 6-8 (gid 226 and all L5 cells)

    # switch the zoom off: the helpers read zoom_depth, z_lo and z_hi
    zoom_depth_saved = zoom_depth
    zoom_depth = None
    z_lo, z_hi = -np.inf, np.inf

    def tall_three_panel_figure():
        return plt.subplots(1, 3, figsize=whole_cell_figsize, sharey=True, constrained_layout=True,
                            gridspec_kw={"width_ratios": [1, 2.5, 2.5]})

    for tag, gids in whole_cell_groups.items():
        cell_type, section_point = group_geometry(gids)

        # as figure 6: Vm per section | net ionic + synaptic currents of every segment
        I_seg, seg_xz, i_scale = segment_currents(gids, section_point, mode="net")
        fig_w6, (ax_m, ax_v, ax_i) = tall_three_panel_figure()
        draw_morphology(ax_m, gids, cell_type, seg_xz)
        draw_vm(ax_v, gids, section_point)
        draw_segment_currents(ax_i, I_seg, seg_xz, i_scale,
                              f"net ionic{syn_txt} currents of every segment,\n"
                              f"{group_text(gids, cell_type)}, at the segment's depth (nA; inward < 0)")
        finish_trace_axes((ax_v, ax_i))
        fig_w6.suptitle(f"{cell_type} {group_text(gids, cell_type)} (whole neuron): membrane potential per section "
                        f"and net ionic / synaptic currents of every segment")
        save_fig(f"neymotin_vsec_morph_{tag}_wholecell", fig_w6)

        # as figure 8: Vm per section | Na, K, KM, Ca (+ net ionic) of every segment
        add_net = four_ionic_add_net == "all" or (four_ionic_add_net == "population" and len(gids) > 1)
        I_seg, seg_xz, i_scale = segment_currents(gids, section_point, mode="four", add_net=add_net)
        fig_w8, (ax_m, ax_v, ax_i) = tall_three_panel_figure()
        draw_morphology(ax_m, gids, cell_type, seg_xz)
        draw_vm(ax_v, gids, section_point)
        net_txt = " (black: net ionic, all channels)" if add_net else ""
        draw_segment_currents(ax_i, I_seg, seg_xz, i_scale,
                              f"Na, K, KM and Ca currents of every segment{net_txt},\n"
                              f"{group_text(gids, cell_type)}, at the segment's depth (nA; inward < 0)",
                              net_color=four_ionic_net_color)
        finish_trace_axes((ax_v, ax_i))
        fig_w8.suptitle(f"{cell_type} {group_text(gids, cell_type)} (whole neuron): membrane potential per section "
                        f"and main ionic currents of every segment")
        save_fig(f"neymotin_vsec_four_ionic_{tag}_wholecell", fig_w8)

    # restore the zoom, in case anything after this block uses it
    zoom_depth = zoom_depth_saved
    if zoom_depth is not None:
        z_lo, z_hi = z_surface - max(zoom_depth), z_surface - min(zoom_depth)

if show_figures:
    plt.show()
else:
    plt.close("all")

# here
