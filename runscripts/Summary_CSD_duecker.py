"""Summary CSD figures for the Duecker (duecker_ET_model) network, straight from the data.

Duecker counterpart of Summary_CSD_neymotin.py. Model-specific information (data file, ionic
channels, synapse names, drives, cell types) is taken from
Understand_CSD_contributions_full_network_dataframe_duecker_v2.py.
Since the most contributing ionic currents of this model are not known yet, the figures that
show single ionic currents show ALL the ionic channels of the pyramidal cells (ionic_channels).

Figures, saved to runscripts/Figures/<data file name>/ (when save_figures is True):
  1) morphology + Total / Capacitive / Ionic / Synaptic / residual / GABA-A slow / GABA-A fast /
     AMPA / NMDA (plott.make_csd_contribution_summary_figure), without and with slab borders:
       duecker_summary_csd_contributions.png, duecker_summary_csd_contributions_with_slabs.png
  2) 2x3 synaptic breakdown (Total | Synaptic | Inhibitory / Excitatory | Recurrent | External),
     own colour scale per panel and fixed scale:
       duecker_summary_synaptic_breakdown_own_scale.png, ..._fixed_scale.png
  3) LFP / morphology / CSD of every single ionic current (one figure per channel):
       duecker_csd_<channel>.png
  4) cumulative sum of the currents (each ionic channel, +synaptic, +capacitive) next to the total:
       duecker_csd_cumulative_sum.png
  5) L5 pyramidal CSD (agg_i_mem) as cells are progressively added:
       duecker_csd_progressive_ncells.png
  6) morphology | Vm per section | net ionic + synaptic currents of every segment
       duecker_vsec_morph_<group><zoom>.png
  7) morphology | LFP of the group per electrode | net ionic + synaptic currents of every segment
       duecker_lfp_currents_depth_<group><zoom>.png
  8) morphology | Vm per section | ALL ionic currents (+ net ionic) of every segment
       duecker_vsec_all_ionic_<group><zoom>.png
Run from the hnn-core folder (data paths are relative to it).
"""
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

import postproc_tm_currents_dipole_lfp_csd as tme
import Plotting_tools as plott

# ---------------------------------------------------------------- settings
# which simulation to use
pkl_path = 'runscripts/data/sim_results_dt000625_conn_dataframe_isec_v2_duecker.pkl'

# True: save every figure as PNG in Figures/<data file name>/; False: only show them
save_figures = True

# which figures to make (False = skip that section entirely: no plotting, no saving)
make_figures = {
    "contributions":      False,   # 1) morphology + total/cap/ionic/syn/... (+ version with slabs)
    "synaptic_breakdown": False,   # 2) Total/Synaptic/Inhibitory/Excitatory/Recurrent/External (own + fixed scale)
    "single_channels":    False,   # 3) every ionic channel: LFP | morphology | CSD (one figure per channel)
    "cumulative_sum":     False,   # 4) channel by channel, + synaptic, + capacitive = total
    "progressive_ncells": False,   # 5) L5 CSD as cells are added
    "vsec_sections":      False,   # 6) morphology | Vm per section | net ionic + synaptic of every segment
    "lfp_currents_depth": False,   # 7) morphology | LFP of the group per electrode | net ionic + synaptic
    "vsec_all_ionic":     False,   # 8) morphology | Vm per section | all ionic currents of every segment
    "each_ionic_csd":     True,   # 9) CSD of every ionic current, each with its own colour scale (grid)
}
# True: open the figure windows at the end; False: close them (useful when only saving)
show_figures = True

contribution_figsize = (22, 11)   # size (inches) of the contribution summary figure; the default is (28, 8)

step = 4                       # downsampling factor (same as the main script)
cell_types = ["L2_pyramidal", "L5_pyramidal"]
drive_names = ["prox1", "dist1", "prox2"]          # Duecker drive names
recurrent_exc = ["L2_pyramidal", "L5_pyramidal"]
inhibitory_types = ["L2_inhibitory", "L5_inhibitory"]

# ionic channels of the Duecker pyramidal cells (everything except capacitive); the Golomb
# currents are left out because they only exist in the interneurons
# channel name -> (short label for legends, panel title)
ionic_channel_info = {
    # sodium
    "ina_NaTa_t_hay2011":      ("NaTa_t",      "Na transient, dendritic (NaTa_t, L5)"),
    "ina_NaTa_t_32d_hay2011":  ("NaTa_t_32d",  "Na transient, dendritic (NaTa_t_32d, L2/3)"),
    "ina_NaTs2_t_hay2011":     ("NaTs2_t",     "Na transient, somatic (NaTs2_t, L5)"),
    "ina_NaTs2_t_32d_hay2011": ("NaTs2_t_32d", "Na transient, somatic (NaTs2_t_32d, L2/3)"),
    "ina_Nap_Et2_hay2011":     ("Nap_Et2",     "Na persistent (Nap_Et2)"),
    # potassium
    "ik_SKv3_1_hay2011":       ("SKv3_1",      "K fast (SKv3_1)"),
    "ik_SK_E2_hay2011":        ("SK_E2",       "K Ca-activated (SK_E2)"),
    "ik_Im_hay2011":           ("Im",          "K M-current (Im)"),
    "ik_K_Pst_hay2011":        ("K_Pst",       "K persistent (K_Pst, L5)"),
    "ik_K_Tst_hay2011":        ("K_Tst",       "K transient (K_Tst, L5)"),
    # calcium
    "ica_Ca_HVA_hay2011":      ("Ca_HVA",      "Ca high-voltage (Ca_HVA)"),
    "ica_Ca_LVAst_hay2011":    ("Ca_LVAst",    "Ca low-voltage (Ca_LVAst)"),
    # non-specific
    "ihcn_Ih_hay2011":         ("Ih",          "Ih (Ih)"),
    "i_pas":                   ("leak",        "Leak (pas)"),
}
ionic_channels = list(ionic_channel_info)
# colours: reds = Na, blues/purples = K, oranges = Ca, brown = Ih, grey = leak
channel_colors = {
    "ina_NaTa_t_hay2011": "#d62728", "ina_NaTa_t_32d_hay2011": "#ff9896",
    "ina_NaTs2_t_hay2011": "#8b0000", "ina_NaTs2_t_32d_hay2011": "#e377c2",
    "ina_Nap_Et2_hay2011": "#f7b6d2",
    "ik_SKv3_1_hay2011": "#1f77b4", "ik_SK_E2_hay2011": "#17becf", "ik_Im_hay2011": "#2ca02c",
    "ik_K_Pst_hay2011": "#9467bd", "ik_K_Tst_hay2011": "#aec7e8",
    "ica_Ca_HVA_hay2011": "#ff7f0e", "ica_Ca_LVAst_hay2011": "#bcbd22",
    "ihcn_Ih_hay2011": "#8c564b", "i_pas": "#7f7f7f",
}

# spikes overlaid on the CSD of the single-channel figures:
# "hist" = spike-count histograms (as in the main script), "raster" = spike dots
spike_overlay = "hist"
spike_types = {'Distal': ['dist'], 'Proximal': ['prox']}    # Duecker drives: prox1, dist1, prox2
csd_hist_cell_types = ['L2_pyramidal', 'L5_pyramidal', 'L2_inhibitory', 'L5_inhibitory']
csd_hist_colors = {'L2_pyramidal': 'k', 'L5_pyramidal': 'k',
                   'L2_inhibitory': 'magenta', 'L5_inhibitory': 'magenta'}
csd_hist_scale = "shared"
csd_hist_baseline = {'L2_inhibitory': 'L2_pyramidal', 'L5_inhibitory': 'L5_pyramidal'}

# cumulative sum: every ionic channel in the order of ionic_channel_info, then synaptic and
# capacitive (keys: channel names, "synaptic", "capacitive"); the last step is the total
cumulative_steps = ([(("" if i == 0 else "+ ") + ionic_channel_info[ch][0], ch)
                     for i, ch in enumerate(ionic_channels)]
                    + [("+ synaptic", "synaptic"), ("+ capacitive\n(= cap + ionic + syn)", "capacitive")])
vmax_cumulative = None   # colour limit (µA/mm³); None = largest |CSD| over all panels

# progressive cell-addition figure (first_gid None = the L5 pyramidal cell closest to the probe)
progressive_cell_type = "L5_pyramidal"
first_gid = 226
n_cells = [1, 2, 5, 10, 20, 40, 70, None]   # None = all cells of that type

# colour limits of the contribution summary figure (µA/mm³): row 0 = total/cap/ionic/syn, row 1 = synapse types
vmax_row0, vmax_row1 = 30, 10

# colour limits of the fixed-scale synaptic breakdown (µA/mm³)
vmax_fixed = {"Total (agg_i_mem)": 30, "Synaptic (all)": 30, "Inhibitory": 10,
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

def closest_gid_to_probe(cell_type, array_name="probe1"):
    """gid of the cell of this type whose soma is closest (in x, y) to the laminar probe."""
    probe_xy = np.asarray(net.rec_arrays[array_name].positions, dtype=float)[0, :2]
    pos = np.asarray(net.pos_dict[cell_type], dtype=float)[:, :2]
    k = int(np.argmin(np.linalg.norm(pos - probe_xy, axis=1)))
    return net.gid_ranges[cell_type][k]

example_gid = 226#closest_gid_to_probe("L5_pyramidal")
print("L5 pyramidal cell closest to the probe: gid", example_gid)

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

# every single ionic channel (taken from the ionic sources collected above)
if make_figures["single_channels"] or make_figures["cumulative_sum"]:
    lfp_csd_single = {}
    for ch in ionic_channels:
        sources_ch, I_ch = tme.filter_sources(sources_ionic, I_ionic, labels=[ch])
        if len(sources_ch) == 0:
            print(f"  {ch}: no sources in the pyramidal cells, skipped")
            continue
        lfp_csd_single[ch] = compute_lfp_csd_from_sources(net, sources_ch, I_ch)   # (lfp_, csd_)

# all synaptic currents
sources_syn, I_syn = tme.collect_synaptic_sources(net, trial_idx=0, cell_types=cell_types)
_, csd_syn_ = compute_lfp_csd_from_sources(net, sources_syn, I_syn)

def csd_of(sources, I, label=""):
    """CSD of a set of sources; zeros if the set is empty."""
    if len(sources) == 0:
        print(f"  no sources for {label} -> CSD set to zero")
        return np.zeros_like(csd_syn_)
    return compute_lfp_csd_from_sources(net, sources, I)[1]

def syn_csd(syn_names, presyn_types=None, label=""):
    """CSD of a subset of the synaptic sources (zeros if the subset is empty)."""
    kwargs = dict(syn_names=syn_names)
    if presyn_types is not None:
        kwargs["presyn_types"] = presyn_types
    sources, I = tme.filter_sources(sources_syn, I_syn, **kwargs)
    return csd_of(sources, I, label or str(syn_names))

# GABA-A: the Duecker model has no GABA-B onto pyramidal cells, but a slow GABA-A receptor
# ("gabaa_slow", L2_inhibitory -> L5_pyramidal) next to the regular (fast) one, as in the v2 script
src_gabaa, I_gabaa = tme.filter_sources(sources_syn, I_syn, syn_names=["gabaa"])
is_slow = np.array(["slow" in str(s.syn_name).split("_") for s in src_gabaa], dtype=bool)
print(f"GABA-A sources: {is_slow.sum()} slow, {(~is_slow).sum()} fast")
csd_syn_gabaa_slow_ = csd_of([s for s, k in zip(src_gabaa, is_slow) if k], I_gabaa[is_slow], "GABA-A slow")
csd_syn_gabaa_fast_ = csd_of([s for s, k in zip(src_gabaa, is_slow) if not k], I_gabaa[~is_slow], "GABA-A fast")

csd_syn_ampa_ = syn_csd(["ampa"], label="AMPA")
csd_syn_nmda_ = syn_csd(["nmda"], label="NMDA")
csd_syn_ampa_drives_ = syn_csd(["ampa"], drive_names, "AMPA from drives")
csd_syn_nmda_drives_ = syn_csd(["nmda"], drive_names, "NMDA from drives")
csd_syn_ampa_recurrent_ = syn_csd(["ampa"], recurrent_exc, "AMPA recurrent")
csd_syn_nmda_recurrent_ = syn_csd(["nmda"], recurrent_exc, "NMDA recurrent")

csd_syn_exc_ = csd_syn_ampa_ + csd_syn_nmda_                  # excitatory synaptic CSD
csd_syn_inh_ = csd_syn_gabaa_slow_ + csd_syn_gabaa_fast_      # inhibitory synaptic CSD (all recurrent)
# GABA synapses only come from L2_inhibitory/L5_inhibitory: no external-drive part for inhibition,
# so "Recurrent" is recurrent EXCITATION only (as in the v2 script)
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
            csd_syn_gabaa_slow_, csd_syn_gabaa_fast_, csd_syn_ampa_, csd_syn_nmda_,
            inh_labels=("GABA-A slow", "GABA-A fast"),
            vmax_row0=vmax_row0, vmax_row1=vmax_row1,
            suptitle="Duecker - Whole network (all pyramidal cells) - from sources",
        )
        fig.set_size_inches(*contribution_figsize)   # the function creates it at (28, 8): very wide and slim
        return fig

    fig_all = contribution_figure()
    save_fig("duecker_summary_csd_contributions", fig_all)

    # same, with the slab (electrode-bin) boundaries on the morphology panel
    z_edges = tme._z_edges_from_array(net, "probe1")
    fig_all_slabs = contribution_figure()
    ax_morph = next(ax for ax in fig_all_slabs.axes if ax.get_title() == "Morphology")
    for z in z_edges:
        ax_morph.axhline(z, color='red', alpha=0.4, lw=1)
    save_fig("duecker_summary_csd_contributions_with_slabs", fig_all_slabs)

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
        fig.suptitle("Duecker - Whole network (all pyramidal cells) - from sources")
        return fig

    fig_own = summary_figure(own_scale=True, figsize=(18, 10))
    save_fig("duecker_summary_synaptic_breakdown_own_scale", fig_own)

    fig_fixed = summary_figure(own_scale=False, figsize=(15, 8.3))
    save_fig("duecker_summary_synaptic_breakdown_fixed_scale", fig_fixed)

# 3) every single ionic current: LFP | morphology | CSD (+ spikes), own symmetric colour scale
if make_figures["single_channels"]:
    for ch, (lfp_, csd_) in lfp_csd_single.items():
        vmax = max(np.max(np.abs(csd_)), 1e-12)
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
            spike_types=spike_types,
            scale_lfp=3.0,
            voltage_scalebar=200,
            vmin=-vmax,
            vmax=vmax,
            figsize=(18, 6),
            overlay_csd_traces=True,
            unit_csd="µA/mm³",
            sink="r",
            show=False,
            **overlay_kwargs,
        )
        fig.suptitle(f"Duecker - LFP/CSD from {ionic_channel_info[ch][1]}")
        save_fig(f"duecker_csd_{ch}", fig)

# 4) cumulative sum of the currents (CSD is linear in the currents, so CSDs simply add up)
if make_figures["cumulative_sum"]:
    components = {ch: lfp_csd_single[ch][1] for ch in lfp_csd_single}
    components["synaptic"] = csd_syn_
    components["capacitive"] = csd_cap_

    cumulative_panels = []
    running = np.zeros_like(csd_from_sources_)
    for label, key in cumulative_steps:
        if key not in components:                 # channel absent in the pyramidal cells
            continue
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
    fig_cum.suptitle("Duecker - cumulative sum of the CSD contributions (all pyramidal cells)")
    save_fig("duecker_csd_cumulative_sum", fig_cum)

# 5) progressive cell addition: total membrane current (agg_i_mem) of the first N cells
if make_figures["progressive_ncells"]:
    gids_all = sorted(net.gid_ranges[progressive_cell_type])
    start_gid = first_gid if first_gid is not None else closest_gid_to_probe(progressive_cell_type)
    assert start_gid in gids_all, f"gid {start_gid} is not a {progressive_cell_type}"
    ordered_gids = [start_gid] + [g for g in gids_all if g != start_gid]
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
    fig_prog.suptitle(f"Duecker {progressive_cell_type}: CSD as cells are progressively added, "
                      f"starting from gid {start_gid} (each panel independently scaled) - {run_tag}",
                      fontsize=13)
    save_fig("duecker_csd_progressive_ncells", fig_prog)

# here
# ==================== settings shared by figures 6, 7 and 8 ====================
# each figure is made once per group: a single cell, or a population whose currents are summed
# (all cells of a group must be of the same type, so they share morphology and depths)
cell_groups = {
    f"gid{example_gid}": [example_gid],                    # the L5 pyramidal cell closest to the probe
    "allL5pyr": list(net.gid_ranges["L5_pyramidal"]),      # all L5 pyramidal cells
}
zoom_depth = (1800, 2400)    # depth range to show (µm from the top contact), or None for the whole probe
vm_scale_um_per_mV = 0.8     # 6, 8) vertical scale of the Vm traces: µm of depth per mV (100 mV -> 80 µm)
vm_scalebar_mV = 100         # 6, 8) size of the voltage scale bar (mV)
vm_baseline_ms = 5.0         # 6, 8) each Vm trace is drawn relative to its mean over the first ms
vm_show_individual = True    # 6, 8) groups of several cells: also draw every cell's Vm (thin grey) under the mean
lfp_gain = 0.8               # 7) 1 = largest LFP deflection (in view) spans 0.8 x contact spacing
# 6, 7) net ionic current of every segment (sum of all ionic channels, nA), solid line
net_ionic_color = "tab:red"
# 6, 7) synaptic currents (nA), dashed lines, one line per receptor (GABA-A split into slow and fast)
show_syn_currents = True
syn_per_receptor = True      # True: one line per receptor; False: one net synaptic line (sum of all receptors)
syn_receptors = {"ampa": "AMPA", "nmda": "NMDA",
                 "gabaa_fast": "GABA-A fast", "gabaa_slow": "GABA-A slow", "gabab": "GABA-B"}
syn_colors = {"ampa": "tab:purple", "nmda": "tab:brown",
              "gabaa_fast": "tab:cyan", "gabaa_slow": "tab:blue", "gabab": "tab:olive"}
net_syn_color = "tab:blue"   # colour of the net synaptic line (syn_per_receptor = False)
# 8) ionic currents of every segment drawn one by one (default: all channels; restrict the list
#    once the most contributing ones are known), solid lines, no synaptic currents
fig8_channels = list(ionic_channels)
# 8) also draw the net ionic current (sum of ALL ionic channels) on top:
#    "population" = only for groups of several cells, "all" = also for single cells, None = never
fig8_add_net = "all"
fig8_net_color = "k"
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

def receptor_of(syn_name):
    """Receptor of a synapse from its name "{source}_{section}_{receptor}_{segX}";
    GABA-A is split into "gabaa_slow" and "gabaa_fast" (Duecker's slow GABA-A is named gabaa_slow)."""
    toks = str(syn_name).split("_")
    if "gabaa" in toks:
        return "gabaa_slow" if "slow" in toks else "gabaa_fast"
    return next((t for t in toks if t in ("ampa", "nmda", "gabab")), None)

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

def segment_currents(gids, section_point, mode="net", add_net=False, channels=None):
    """Currents of a group of cells, summed over the cells, one entry per (segment, current):
    (section, segment x, kind, name, trace nA)
    mode = "net":      kind "ionic", name "net" = sum of all ionic channels of the segment,
                       plus kind "syn" (name = receptor, or "net" if syn_per_receptor is False)
    mode = "channels": kind "ionic", name = channel, for the channels in `channels` (no synaptic);
                       with add_net=True also kind "ionic", name "net" (sum of ALL ionic channels)
    Also returns the (x, z) of every segment centre and the current scale (µm per nA)."""
    channels = set(fig8_channels if channels is None else channels)
    sums = {}                                            # (section, x, kind, name) -> summed trace
    def add(key, trace):
        sums[key] = sums.get(key, 0) + trace
    src_c, I_c = tme.filter_sources(sources_ionic, I_ionic, gid_subset=gids)
    for k, src in enumerate(src_c):
        if mode == "channels" and src.label in channels:
            add((src.section, src.segment_x, "ionic", src.label), I_c[k])
        if mode == "net" or add_net:
            add((src.section, src.segment_x, "ionic", "net"), I_c[k])
    if mode == "net" and show_syn_currents:
        src_s, I_s = tme.filter_sources(sources_syn, I_syn, gid_subset=gids)
        for k, src in enumerate(src_s):
            receptor = receptor_of(src.syn_name)
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
    zero line of every segment, legend and nA scale bar. net_color: colour of the net ionic line
    (default net_ionic_color); the net line is drawn last, on top of the per-channel lines."""
    net_color = net_color or net_ionic_color
    for sec, xs, kind, name, i_tr in sorted(I_seg, key=lambda e: e[2] == "ionic" and e[3] == "net"):
        _, z_s = seg_xz[(sec, xs)]
        if kind == "ionic":
            is_net = name == "net"
            ax_c.plot(times[:len(i_tr)], z_s + i_scale * i_tr,
                      color=net_color if is_net else channel_colors.get(name, "0.3"),
                      lw=0.9 if is_net else 0.7)
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
    handles += [plt.Line2D([], [], color=channel_colors.get(ch, "0.3"), label=ionic_channel_info[ch][0])
                for ch in ionic_channels if ("ionic", ch) in present]
    if ("syn", "net") in present:
        handles.append(plt.Line2D([], [], color=net_syn_color, ls="--", label="net synaptic"))
    handles += [plt.Line2D([], [], color=syn_colors[r], ls="--", label=lab)
                for r, lab in syn_receptors.items() if ("syn", r) in present]
    ax_c.legend(handles=handles, loc="upper right", fontsize=7, ncol=min(len(handles), 5), frameon=False)
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
        fig_v.suptitle(f"Duecker {cell_type} {group_text(gids, cell_type)}: membrane potential per section "
                       f"and net ionic / synaptic currents of every segment")
        save_fig(f"duecker_vsec_morph_{tag}{zoom_tag}", fig_v)

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
        fig_l.suptitle(f"Duecker {cell_type} {group_text(gids, cell_type)}: extracellular potential at each electrode "
                       f"and net ionic / synaptic currents of every segment")
        save_fig(f"duecker_lfp_currents_depth_{tag}{zoom_tag}", fig_l)

# ==================== 8) morphology | Vm per section | all ionic currents (+ net ionic) of every segment ====================
if make_figures["vsec_all_ionic"]:
    for tag, gids in cell_groups.items():
        cell_type, section_point = group_geometry(gids)
        add_net = fig8_add_net == "all" or (fig8_add_net == "population" and len(gids) > 1)
        I_seg, seg_xz, i_scale = segment_currents(gids, section_point, mode="channels",
                                                  add_net=add_net, channels=fig8_channels)
        fig_8, (ax_m, ax_v, ax_i) = three_panel_figure()
        draw_morphology(ax_m, gids, cell_type, seg_xz)
        draw_vm(ax_v, gids, section_point)
        net_txt = " (black: net ionic, all channels)" if add_net else ""
        draw_segment_currents(ax_i, I_seg, seg_xz, i_scale,
                              f"ionic currents of every segment{net_txt},\n"
                              f"{group_text(gids, cell_type)}, at the segment's depth (nA; inward < 0)",
                              net_color=fig8_net_color)
        finish_trace_axes((ax_v, ax_i))
        fig_8.suptitle(f"Duecker {cell_type} {group_text(gids, cell_type)}: membrane potential per section "
                       f"and ionic currents of every segment")
        save_fig(f"duecker_vsec_all_ionic_{tag}{zoom_tag}", fig_8)

# ==================== 9) CSD of every ionic current, each with its own colour scale ====================
if make_figures.get("each_ionic_csd", True):
    ionic_csd_vmax = "max"                   # "max": ±largest |CSD| of each panel; or a number, e.g. 99 = 99th percentile
    ionic_csd_ncols = 4

    # CSD of each channel (from the ionic sources collected at the top of the script)
    csd_by_channel = {}
    for ch in ionic_channels:
        src_ch, I_ch = tme.filter_sources(sources_ionic, I_ionic, labels=[ch])
        if len(src_ch) == 0:
            print(f"  {ch}: no sources in the pyramidal cells, skipped")
            continue
        csd_by_channel[ch] = compute_lfp_csd_from_sources(net, src_ch, I_ch)[1]

    # print: peak sink / source of each channel (sink = negative CSD), with time and depth
    z_top = np.max(contact_labels)
    print(f"\nIonic CSD per channel ({run_tag}), µA/mm³:")
    print(f"{'channel':26s} {'max |CSD|':>10s} {'min (sink)':>11s} {'t (ms)':>7s} {'depth':>6s}"
          f" {'max (source)':>13s} {'t (ms)':>7s} {'depth':>6s}")
    for ch, csd in sorted(csd_by_channel.items(), key=lambda kv: -np.max(np.abs(kv[1]))):   # largest first
        i_min = np.unravel_index(np.argmin(csd), csd.shape)
        i_max = np.unravel_index(np.argmax(csd), csd.shape)
        print(f"{ch:26s} {np.max(np.abs(csd)):10.3f} "
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
        ax.set_title(f"{ionic_channel_info[ch][1]}\n(±{vmax_ch:.3g} µA/mm³)", fontsize=10)
    for ax in list(axes_ion.flat)[n_panels:]:               # unused grid cells
        ax.set_visible(False)
    fig_ion.suptitle(f"Duecker - CSD of every ionic current (all pyramidal cells), own colour scale per panel - {run_tag}")
    save_fig("duecker_csd_each_ionic_channel", fig_ion)

if show_figures:
    plt.show()
else:
    plt.close("all")

# here