# %% [markdown] ###########################################################
## Setup
# %% ######################################################################

import matplotlib.pyplot as plt
import numpy as np
from neuron import h

from hnn_core import (
    JoblibBackend,
    jones_2009_model,
    simulate_dipole,
)
from hnn_core.cells_default import pyramidal
from hnn_core.dev_analysis.dipole_processing import (
    dipole_from_components_v1 as postproc_dipole_from_components,
)
from hnn_core.dev_analysis.dipole_processing import (
    postproc_soma_dipole,
)
from hnn_core.network_models import add_erp_drives_to_jones_model

# pyright: reportOptionalMemberAccess=false

# %% [markdown] ----------------------------------------
## Instantiate networks
# %% ---------------------------------------------------


def init_net_base(
    # verbose=False,
    plot=False,
    tstop=70.0,
    record_method="all",
):
    net_base = jones_2009_model()
    add_erp_drives_to_jones_model(net_base)
    tstop = 70.0
    record_method = "all"

    with JoblibBackend(8):
        dpls = simulate_dipole(
            net_base,
            tstop=tstop,
            n_trials=1,
            record_agg_ica=f"{record_method}",
            # record_agg_i_non_specific=f"{record_method}",
            record_agg_i_mem=f"{record_method}",
            record_agg_ina=f"{record_method}",
            record_agg_ik=f"{record_method}",
            record_agg_i_cap=f"{record_method}",
            record_ina_hh2=f"{record_method}",
            record_ik_hh2=f"{record_method}",
            record_ik_kca=f"{record_method}",
            record_ik_km=f"{record_method}",
            record_ica_ca=f"{record_method}",
            record_ica_cat=f"{record_method}",
            record_il_hh2=f"{record_method}",
            record_i_ar=f"{record_method}",
            record_isec=f"{record_method}",
        )

    if plot:
        scaling_factor = 3000
        for dpl in dpls:
            dpl.scale(scaling_factor)

        dpl = dpls[0]
        _ = dpl.plot(
            layer=["L5"],
        )

    return net_base


def init_net_no_local_one_drive(
    verbose=False,
    plot=False,
    tstop=70.0,
    dt=0.025,
    record_method="all",
):
    net_no_local_one_drive = jones_2009_model()

    weights_ampa_p1 = {
        "L2_basket": 0.08831,
        "L2_pyramidal": 0.01525,
        "L5_basket": 0.19934,
        "L5_pyramidal": 0.00865,
    }
    synaptic_delays_prox = {
        "L2_basket": 0.1,
        "L2_pyramidal": 0.1,
        "L5_basket": 1.0,
        "L5_pyramidal": 1.0,
    }

    # all NMDA weights are zero; pass None explicitly
    net_no_local_one_drive.add_evoked_drive(
        "evprox1",
        mu=26.61,
        sigma=2.47,
        numspikes=1,
        weights_ampa=weights_ampa_p1,
        weights_nmda=None,
        location="proximal",
        synaptic_delays=synaptic_delays_prox,
        event_seed=544,
    )

    # print all original connections before removing connections
    if verbose:
        print("--- ORIGINAL CONNECTIVITY ---")
        for conn in net_no_local_one_drive.connectivity:
            print(
                f"Source: {conn['src_type']:<15} -> Target: {conn['target_type']:<15}"
                f" via {conn['receptor']}"
            )

    # keep only connections whose source is NOT a local cell, effectively
    # removing the entire local network
    local_types = ["L2_basket", "L2_pyramidal", "L5_basket", "L5_pyramidal"]

    stripped_connectivity = [
        conn
        for conn in net_no_local_one_drive.connectivity
        if conn["src_type"] not in local_types
    ]

    net_no_local_one_drive.connectivity = stripped_connectivity

    if verbose:
        # print updated connectivity to verify only drives remain
        print("\n--- UPDATED CONNECTIVITY ---")
        for conn in net_no_local_one_drive.connectivity:
            print(
                f"Source: {conn['src_type']:<15} -> Target: {conn['target_type']:<15}"
                f" via {conn['receptor']}"
            )

    # run simulation
    with JoblibBackend(8):
        dpls = simulate_dipole(
            net_no_local_one_drive,
            dt=dt,
            tstop=tstop,
            n_trials=1,
            record_agg_ica=f"{record_method}",
            # record_agg_hh2=f"{record_method}",
            # record_agg_i_non_specific=f"{record_method}",
            record_agg_i_mem=f"{record_method}",
            record_agg_ina=f"{record_method}",
            record_agg_ik=f"{record_method}",
            record_agg_i_cap=f"{record_method}",
            record_ina_hh2=f"{record_method}",
            record_ik_hh2=f"{record_method}",
            record_ik_kca=f"{record_method}",
            record_ik_km=f"{record_method}",
            record_ica_ca=f"{record_method}",
            record_ica_cat=f"{record_method}",
            record_il_hh2=f"{record_method}",
            record_i_ar=f"{record_method}",
            record_isec=f"{record_method}",
            # record_prec_i_cap=f"{record_method}",
        )

    scaling_factor = 3000
    for dpl in dpls:
        dpl.scale(scaling_factor)

    dpl = dpls[0]

    if plot:
        _ = dpl.plot(
            layer=["L5"],
        )

    return net_no_local_one_drive


# %% [markdown] ############################################################
# # [DEV] Exploring dipole discrepancy
# ##########################################################################
#
# overview:
#   - our goal is to recreate the axial dipole computed by hnn-core from the
#     individual (synaptic + ionic + capacitive) transmembrane currents
#   - we can tell NEURON to track the aggregate transmembrane currents
#     ("agg_i_mem") during simultion, and we can use this current recording
#     to recreate the axial dipole shape exactly (there is an "offset" /
#     vertical shift in the plot, but otherwise the shapes match perfectly)
#
# current issue:
#   - there is a discrepancy between the dipole computed from agg_i_mem
#     versus the dipole computed from the constituent currents
#   - since the agg_i_mem dipole matches the axial dipole computed by
#     hnn-core, we need to understand and resolve the discrepancy between
#     agg_i_mem and the constituent currents
#   - of note, the dipoles computed from agg_i_mem versus the constituent
#     currents *does* match perfectly when only looking at the soma, which
#     is composed of a single segment
#   - the fact that the some dipoles match has several possible
#     interpretations:
#       - there is a constituent current missing from the dendrites that
#         is *not*  present in the soma that causes the dipoles for the
#         dendrites to diverge
#       - the simplification of synaptic currents being at the midpoint
#         of the section does not work for multi-segment sections when
#         calculating the dipole, and thus we need segment-specific
#         synaptic currents to correctly reproduce agg_i_mem from the
#         constituent currents
#
# things to check:
#   - [x] check that the soma dipoles match for i_mem versus components
#   - [ ] check that the dendrite dipoles all match for i_mem versus components
#   - [x] check that unit match is correct
#       - notes
#   - [x] check that area math (area_cm2 calculation) is correct
#       - since the aggregates currents (agg_ina, agg_ik) exactly match
#         the sum of the mechanisms (e.g., hh2, kca, km for agg_ik), the
#         area_cm2 calculation is correct
#   - [ ] check that the residuals are zero for Na, K, Ca
#       - [x] Na residuals are 0
#       - [x] K residuals are 0
#       - [ ] Code not presently set up to record agg_ca

# %% --------------------------------------------------
# compare function outputs
# -----------------------------------------------------

# notes:
#   - function for visualizing dipoles computed from different
#     methods (functions) and under different conditions


def compare_imem_to_components(
    net,
    cell_type="L5_pyramidal",
    sections=None,
):
    """ """

    fig, ax = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=(12, 8),
    )

    # generate data
    test_imem = postproc_dipole_from_components(
        net=net,
        from_components=False,
        cell_type=cell_type,
        sections=sections,
    )

    test_comp = postproc_dipole_from_components(
        net=net,
        from_components=True,
        cell_type=cell_type,
        sections=sections,
    )

    # plotting
    times = net.cell_response.times
    ax[0].plot(times[1:], test_imem[1:])
    ax[1].plot(times[1:], test_comp[1:])

    # synchronize scales using data from [1:]
    combined_data = np.concatenate([test_imem[1:], test_comp[1:]])
    y_min, y_max = np.min(combined_data), np.max(combined_data)
    padding = 0.05 * (y_max - y_min)

    for axis in ax:
        axis.set_ylim(y_min - padding, y_max + padding)
        axis.grid(True, alpha=0.3)
        axis.set_ylabel("nAm")

    if sections is None:
        sub_title = "\nSections: all"
    else:
        sub_title = f"\nSections: {sections}"

    ax[0].set_title(f"dipole from imem ({cell_type}){sub_title}")
    ax[1].set_title(f"dipole from components ({cell_type}){sub_title}")
    ax[1].set_xlabel("Time (ms)")

    plt.tight_layout()
    return fig


def dev_postproc_func_test(
    net_all,
    net_soma,
    dpl,
    cell_type="L5_pyramidal",
):
    fig, ax = plt.subplots(
        nrows=4,
        ncols=1,
        sharex=True,
        figsize=(8, 15),
    )

    # net_rec_soma (record="soma") figures
    # ------------------------------

    # from imem
    test_imem_L5 = postproc_soma_dipole(
        net=net_soma,
        from_components=False,
        cell_type=cell_type,
    )

    ax[0].plot(
        dpl.times[1:],
        test_imem_L5[1:],
    )

    # from components
    test_imem_L5 = postproc_dipole_from_components(
        net=net_soma,
        from_components=True,
        cell_type=cell_type,
    )

    ax[1].plot(
        dpl.times,
        test_imem_L5,
    )

    ax[0].set_title('Soma dipole from imem when record="soma"')
    ax[1].set_title('Soma dipole from components when record="soma"')

    ax[0].set_ylim(-10, 80)
    ax[1].set_ylim(-10, 80)

    # net (record="all") figures
    # ------------------------------

    # from imem
    test_imem_L5 = postproc_dipole_from_components(
        net=net_all,
        from_components=False,
        cell_type=cell_type,
    )

    ax[2].plot(
        dpl.times[1:],
        test_imem_L5[1:],
    )

    # from components
    test_imem_L5 = postproc_dipole_from_components(
        net=net_all,
        from_components=True,
        cell_type=cell_type,
    )

    ax[3].plot(
        dpl.times,
        test_imem_L5,
    )

    ax[2].set_title('Agg dipole from imem when record="all"')
    ax[3].set_title('Agg dipole from components when record="all"')

    ax[2].set_ylim(-200, 200)
    ax[3].set_ylim(-200, 200)

    return fig


def get_single_gid_dipole(net, gid, cell_type, sections, from_components):
    from hnn_core.cells_default import pyramidal

    trial = 0
    scaling_factor = 3000
    template_cell = pyramidal(cell_name=cell_type)
    template_cell.build(sec_name_apical="apical_trunk")

    if not from_components:
        all_tm_channels = ["agg_i_mem"]
    else:
        all_tm_channels = [
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

    start_index = net.gid_ranges[cell_type][0]
    soma_pos = np.array(net.pos_dict[cell_type][gid - start_index])

    dipole = None
    cell_channels = {
        ch: net.cell_response.transmembrane_currents[ch][trial][gid]
        for ch in all_tm_channels
    }
    cell_syn_data = net.cell_response.isec[trial][gid]

    for sec_name in sections:
        sec_hoc = template_cell._nrn_sections[sec_name]
        start = np.array([sec_hoc.x3d(0), sec_hoc.y3d(0), sec_hoc.z3d(0)]) + soma_pos
        end = (
            np.array(
                [
                    sec_hoc.x3d(sec_hoc.n3d() - 1),
                    sec_hoc.y3d(sec_hoc.n3d() - 1),
                    sec_hoc.z3d(sec_hoc.n3d() - 1),
                ]
            )
            + soma_pos
        )

        nseg = len(cell_channels[all_tm_channels[0]][sec_name])
        for i, seg_key in enumerate(cell_channels[all_tm_channels[0]][sec_name].keys()):
            pos = (i + 0.5) / nseg
            z_i = (start + pos * (end - start))[2]

            I_t = np.zeros_like(
                np.array(cell_channels[all_tm_channels[0]][sec_name][seg_key])
            )
            for ch in all_tm_channels:
                if seg_key not in cell_channels[ch][sec_name]:
                    continue
                vec = np.array(cell_channels[ch][sec_name][seg_key])
                area = sec_hoc(pos).area() * 1e-8
                I_t += vec if ch == "agg_i_mem" else vec * area

            if from_components and np.isclose(pos, 0.5) and sec_name in cell_syn_data:
                for rec_vec in cell_syn_data[sec_name].values():
                    I_t += np.array(rec_vec) * 1e-6

            contrib = I_t * z_i
            if not from_components:
                contrib = (contrib / 1e6) * scaling_factor
            else:
                contrib = contrib * scaling_factor
            dipole = contrib if dipole is None else dipole + contrib
    return dipole


# %% [markdown] --------------------------------------------------------
# # Diagnostics
# ----------------------------------------------------------------------

# %% --------------------------------------------------
# [DEV] view population correlation, gain, spikes
# -----------------------------------------------------

# notes:
#   - performs a cell-by-cell comparison of dipole reconstruction
#   - calculates correlation, peak gain ratio, and spike count per GID


def cell_specific_dipole_reconstruction(
    net,
    cell_type="L5_pyramidal",
    trial=0,
):
    """
    Analyzes reconstruction accuracy across the whole population.
    """
    import numpy as np
    from scipy.stats import pearsonr

    gids = net.gid_ranges[cell_type]

    # hnn-core stores spikes as a list of arrays (one per trial)
    # We ensure we are looking at the correct trial's array
    trial_spike_gids = np.array(net.cell_response.spike_gids[trial])

    print(f"--- Population Analysis: {cell_type} ---")
    print(f"{'GID':<5} | {'Corr':<10} | {'Gain':<8} | {'Spikes':<5}")
    print("-" * 40)

    results = []

    for gid in gids:
        # 1. get truth and sum for this specific GID (Soma Only)
        d_truth = get_single_gid_dipole(
            net, gid, cell_type, sections=["soma"], from_components=False
        )
        d_sum = get_single_gid_dipole(
            net, gid, cell_type, sections=["soma"], from_components=True
        )

        # 2. calculate correlation (ignoring t=0)
        corr, _ = pearsonr(d_truth[1:], d_sum[1:])

        # 3. calculate gain ratio at the peak of the truth signal
        t_idx = np.argmax(np.abs(d_truth[1:])) + 1
        gain = d_sum[t_idx] / d_truth[t_idx] if d_truth[t_idx] != 0 else 0

        # 4. count spikes for this GID specifically
        n_spikes = np.sum(trial_spike_gids == gid)

        results.append({"gid": gid, "corr": corr, "gain": gain, "spikes": n_spikes})

        # print every 10th cell and any cell with low correlation or high gain
        if gid % 10 == 0 or corr < 0.95 or abs(gain - 1.0) > 0.1:
            print(f"{gid:<5} | {corr:<10.6f} | {gain:<8.4f} | {n_spikes:<5}")

    return results


# ------------------------------
# check RMSE/residuals
# ------------------------------
def check_nrmse_bt_computed_dipoles(
    from_agg,
    from_components,
    verbose=False,
):
    # calculate residual and NRMSE from
    # NRMSE calculated using signal standard deviation (NRMSE = RMSE / signal_std)
    residual = from_agg - from_components
    rmse = np.sqrt(np.mean(residual**2))
    signal_std = np.std(from_agg)
    nrmse_pct = (rmse / signal_std) * 100 if signal_std != 0 else 0

    if verbose:
        print(
            "\nComparing dipole computed when `from_components=True` versus"
            " `from_components=False`"
        )
        print(f"   NRMSE (RMSE / signal_std): {nrmse_pct:.2f}%")

    return rmse, nrmse_pct


def check_rmse_and_residuals(
    net,
    trial=0,
    cell_type="L5_pyramidal",
):
    times = net.cell_response.times

    # get dipole from i_mem
    dpl_imem = postproc_soma_dipole(
        net,
        trial=trial,
        cell_type=cell_type,
        from_components=False,
    )
    # get the dipole reconstructed from the constituent components
    dpl_comp = postproc_soma_dipole(
        net,
        trial=trial,
        cell_type=cell_type,
        from_components=True,
    )

    if dpl_imem is None or dpl_comp is None:
        raise ValueError(
            "Dipole data could not be computed",
        )

    # calculate residual and rmse
    residual = dpl_imem - dpl_comp
    rmse = np.sqrt(np.mean(residual**2))

    # normalize by the peak-to-peak range of agg_i_mem (our "ground truth")
    # dpl_range = np.max(dpl_imem) - np.min(dpl_imem)
    # nrmse_pct = (rmse / dpl_range) * 100 if dpl_range != 0 else 0

    # version 2: signal standard deviation (NRMSE = RMSE / signal_std)
    #   - read as: "error relative to natural variability of the aggregate signal"
    signal_std = np.std(dpl_imem)
    nrmse_pct = (rmse / signal_std) * 100 if signal_std != 0 else 0

    fig, ax = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        sharex=True,
    )

    # overlay plot
    ax[0].plot(
        times[1:],
        dpl_imem[1:],
        label="From agg_i_mem",
        alpha=0.8,
    )
    ax[0].plot(
        times[1:],
        dpl_comp[1:],
        label="From components",
        linestyle="--",
        alpha=0.8,
    )

    # text box for rmse
    error_text = f" RMSE: {rmse:.2f}\nNRMSE:  {nrmse_pct:.2f}%"
    ax[0].text(
        x=0.05,
        y=0.95,
        s=error_text,
        transform=ax[0].transAxes,
        verticalalignment="top",
        fontsize=12,
        fontfamily="Menlo",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.2,
        ),
    )

    ax[0].set_title(
        f"Computed Dipole Comparison for {cell_type.replace('_', ' ').title()} Soma",
    )
    ax[0].set_ylabel("nAm")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(loc="upper right")

    ymin, ymax = ax[0].get_ylim()
    ylim = (max(abs(ymin), abs(ymax))) * 1.05
    ax[0].set_ylim(-ylim, ylim)

    # residual dipole plot
    ax[1].plot(
        times[1:],
        residual[1:],
        color="red",
        label="(i_mem - reconstructed), scaled",
    )
    ax[1].set_title(
        "Residual (Error)",
    )
    ax[1].set_ylabel("nAm")
    ax[1].set_ylim(-ylim, ylim)
    ax[1].grid(True, alpha=0.3)
    # ax[1].legend()

    rightax = ax[1].twinx()
    rightax.plot(
        times[1:],
        residual[1:],
        alpha=0.2,
        label="(i_mem - reconstructed), zoomed",
    )
    ymin, ymax = rightax.get_ylim()
    ylim = max(abs(ymin), abs(ymax))
    rightax.set_ylim(-ylim, ylim)

    lines1, labels1 = ax[1].get_legend_handles_labels()
    lines2, labels2 = rightax.get_legend_handles_labels()

    ax[1].legend(lines1 + lines2, labels1 + labels2)

    plt.tight_layout()
    plt.show()


# ------------------------------
# validate correct area
# ------------------------------

# %% --------------------------------------------------
# [DEV] Area scaling check
# -----------------------------------------------------


def check_section_geometry_consistency(
    net,
    cell_type="L5_pyramidal",
    sections=None,
):
    cell_meta = net.cell_types[cell_type]["cell_object"]
    template_cell = pyramidal(cell_name=cell_type)
    template_cell.build(sec_name_apical="apical_trunk")

    if sections is None:
        sections = list(cell_meta.sections.keys())

    print(
        f"{'Section':<15} | {'Sim Area (um2)':<15} |"
        f" {'Template Area (um2)':<15} | {'Ratio'}"
    )
    print("-" * 65)

    for sec in sections:
        # Simulation Truth (Metadata)
        L_sim, d_sim = cell_meta.sections[sec].L, cell_meta.sections[sec].diam
        area_sim = np.pi * L_sim * d_sim

        # Template (Used in postproc function)
        area_temp = template_cell._nrn_sections[sec](0.5).area()

        print(
            f"{sec:<15} | {area_sim:<15.2f} | {area_temp:<15.2f} | "
            f"{area_temp / area_sim:.4f}"
        )


def check_segment_area_scaling(net, cell_type):
    # access simulation metadata for a dendrite
    cell_meta = net.cell_types[cell_type]["cell_object"]
    sec_meta = cell_meta.sections["apical_trunk"]

    # build current template

    template_cell = pyramidal(cell_name=cell_type)
    template_cell.build(sec_name_apical="apical_trunk")
    sec_temp = template_cell._nrn_sections["apical_trunk"]

    print("--- GEOMETRY CHECK ---")
    print(f"Simulation 'apical_trunk': L={sec_meta.L}, diam={sec_meta.diam}")
    print(f"Template   'apical_trunk': L={sec_temp.L}, diam={sec_temp.diam}")

    print("\n--- AREA CALCULATION CHECK ---")
    # get nseg from recorded data

    gid = net.gid_ranges[cell_type][0]

    first_ch = next(
        (
            ch
            for ch, data in net.cell_response.transmembrane_currents.items()
            if data
            and gid in data[0]
            and "apical_trunk" in data[0][gid]
            and len(data[0][gid]["apical_trunk"]) > 0
        ),
        None,
    )

    nseg_sim = len(
        net.cell_response.transmembrane_currents[first_ch][0][gid]["apical_trunk"]
    )

    # NOTES:
    # This check uses sec(x).area() at x=0.5, which returns the area
    # associated with the middle segment (not the total area of the section
    # when nseg > 1).

    # In NEURON, this corresponds to:
    #     total_section_area / nseg

    # Thus, this check is suitable for validating segment-based area scaling
    # but ~not~ for checking full geometric integration.

    # ASSUMPTIONS:
    # - uniform diameter along the section
    # - evenly distributed segments (standard NEURON discretization)

    area_reconstructed = sec_temp(0.5).area()

    # what the area of one segment should be (total area / nseg):
    total_area_sim = np.pi * sec_meta.diam * sec_meta.L
    area_expected = total_area_sim / nseg_sim

    print(f"Simulation nseg for this section: {nseg_sim}")
    print(f" Area used in the reconstruction: {area_reconstructed:.2f} um2")
    print(f"       Expected area per segment: {area_expected:.2f} um2")
    print(
        f"               Difference Factor: {area_reconstructed / area_expected:.2f}x"
    )


def validate_geometry_and_segment_scaling(
    net,
    cell_type="L5_pyramidal",
):
    print("\n==============================")
    print("GEOMETRY + SEGMENT VALIDATION")
    print("==============================\n")

    check_section_geometry_consistency(net, cell_type)

    print("\n------------------------------\n")

    check_segment_area_scaling(
        net,
        cell_type,
    )


# ------------------------------
# check for i_pass
# ------------------------------


"""
It is documented in the foundational science for HNN, specifically the Jones et al.
2009 paper (Quantitative analysis and biophysically realistic neural modeling of the
MEG mu rhythm).[1]
In that paper, the methodology states:
"The dendrites were passive with a membrane resistance of 30,000
Ω
Ω
 cm²..."
In the NEURON simulator (which hnn-core uses), "passive" means the section has the pas
mechanism inserted. This mechanism produces the i_pas current variable. By contrast,
the soma is "active" because it has the hh2 mechanism inserted to generate action
potentials. The hh2 mechanism includes its own internal leak term, which is recorded as
il_hh2.
Why Row 4 was failing (The Missing Leak)
This distinction is the "smoking gun" for why your Row 4 diverged from Row 3:
agg_i_mem (Row 3): This is a "catch-all" variable. It records every single charge
crossing the membrane, including the current from pas mechanisms and hh2 mechanisms. It
is correct by default.
Components (Row 4): You were only summing il_hh2.
In the Soma, hh2 is present, so il_hh2 represents the leak. Row 2 matched Row 1.
In the Dendrites, hh2 is not present, so il_hh2 is zero. The leak in the dendrites is
actually i_pas. Because you weren't including i_pas in your sum, you were effectively
modeling a cell with "non-leaky" dendrites.
"""

for sec_name in ["soma", "apical_1", "apical_2", "basal_1"]:
    # Find the HOC section
    hoc_sec = next((s for s in h.allsec() if sec_name in s.name()), None)
    if hoc_sec:
        # ismembrane checks if the mechanism is inserted
        has_pas = h.ismembrane("pas", sec=hoc_sec)
        has_hh2 = h.ismembrane("hh2", sec=hoc_sec)
        print(
            f"Section {sec_name:<10} | Has pas: {bool(has_pas):<5}"
            f" | Has hh2: {bool(has_hh2)}"
        )

# ------------------------------
# check the ionic current residuals
# ------------------------------


def check_ion_residuals(net):
    cell_type = "L5_pyramidal"

    gid = net.gid_ranges[cell_type][0]
    sec = "apical_2"
    seg = "seg_1"
    trial = 0

    # get the agg currents (our "ground truth" in this case)
    agg_ina = np.array(
        net.cell_response.transmembrane_currents["agg_ina"][trial][gid][sec][seg]
    )
    agg_ik = np.array(
        net.cell_response.transmembrane_currents["agg_ik"][trial][gid][sec][seg]
    )
    imem = np.array(
        net.cell_response.transmembrane_currents["agg_i_mem"][trial][gid][sec][seg]
    )
    icap_dens = np.array(
        net.cell_response.transmembrane_currents["agg_i_cap"][trial][gid][sec][seg]
    )

    # get the component (mechanism-specific) currents
    ina_hh2 = np.array(
        net.cell_response.transmembrane_currents["ina_hh2"][trial][gid][sec][seg]
    )
    ik_hh2 = np.array(
        net.cell_response.transmembrane_currents["ik_hh2"][trial][gid][sec][seg]
    )
    ik_kca = np.array(
        net.cell_response.transmembrane_currents["ik_kca"][trial][gid][sec][seg]
    )
    ik_km = np.array(
        net.cell_response.transmembrane_currents["ik_km"][trial][gid][sec][seg]
    )

    # calcute the differentials
    #   - if these are ~not~ zero, a mechanism is writing to out agg currents that
    #     isn't accounted for
    na_residual = agg_ina - ina_hh2
    k_residual = agg_ik - (ik_hh2 + ik_kca + ik_km)

    # calculate total currents in nA
    #   - agg_i_mem = i_cap + i_na + i_k + i_ca + i_non_specific
    template_cell = pyramidal(cell_name=cell_type)
    template_cell.build(sec_name_apical="apical_trunk")
    area_cm2 = template_cell._nrn_sections[sec](0.5).area() * 1e-8

    imem_nA = imem
    ina_nA = agg_ina * area_cm2 * 1e6
    ik_nA = agg_ik * area_cm2 * 1e6
    icap_nA = icap_dens * area_cm2 * 1e6

    # calculate the remainder
    #   - if the Na and K residuals are zero, then the remainder must be
    #     accounted for by one of the other currents (ica_ca, ica_cat, il_hh2,
    #     and i_ar)
    remainder_nA = imem_nA - (icap_nA + ina_nA + ik_nA)

    print(f"--- Residuals Analysis for {sec} ---")
    print(f"   Sodium Residual:  {np.mean(na_residual):.8e} mA/cm2")
    print(f"Potassium Residual:  {np.mean(k_residual):.8e} mA/cm2")
    print(f"   Total Remainder:  {np.mean(remainder_nA):.8e} nA")


# net = net_base
# check_ion_residuals(net)

# %% --------------------------------------------------
# [DEV] population-wide ion balance check
# -----------------------------------------------------

# notes:
#   - function to calculate the balance between agg_i_mem and
#     reconstructed components for every GID over the entire simulation
#   - calculates both Max Residuals and RMSE for global and ion-specific pools
#   - printing logic: shows every 10th cell OR any aberrant cell
#     where the global residual exceeds the specified threshold
#   - threshold adjusted to 1e-2 to distinguish between baseline numerical
#     drift (~1e-4) and spike-driven discrepancies (>1.0)


def analyze_population_ion_balance(
    net,
    cell_type="L5_pyramidal",
    sec_name="soma",
    trial=0,
    threshold=1e-2,
):
    """
    Calculates residuals and RMSE for total membrane current and ion pools.
    """

    gids = net.gid_ranges[cell_type]
    trial_spike_gids = np.array(net.cell_response.spike_gids[trial])

    # get simulation metadata for area scaling
    cell_obj = net.cell_types[cell_type]["cell_object"]
    sec_meta = cell_obj.sections[sec_name]

    # determine nseg and midpoint segment key from the first cell
    nseg = len(
        net.cell_response.transmembrane_currents["agg_i_mem"][trial][gids[0]][sec_name]
    )
    seg_key = f"seg_{(nseg // 2) + 1}"

    # calculate segment area and conversion scale (mA/cm2 to nA)
    area_um2 = (np.pi * sec_meta.diam * sec_meta.L) / nseg
    scale = area_um2 * 0.01

    print(f"--- Population Ion Balance Check: {cell_type} {sec_name} ---")
    header = (
        f"{'GID':<5} | {'Spikes':<6} | {'Global RMSE':<12} | "
        f"{'Global Res':<12} | {'Na Res':<10} | {'K Res':<10} | {'Ca Res':<10}"
    )
    print(header)
    print("-" * len(header))

    results = []

    for i, gid in enumerate(gids):
        # 1. retrieve aggregate truth densities (mA/cm2)
        icap = np.array(
            net.cell_response.transmembrane_currents["agg_i_cap"][trial][gid][sec_name][
                seg_key
            ]
        )
        a_na = np.array(
            net.cell_response.transmembrane_currents["agg_ina"][trial][gid][sec_name][
                seg_key
            ]
        )
        a_k = np.array(
            net.cell_response.transmembrane_currents["agg_ik"][trial][gid][sec_name][
                seg_key
            ]
        )
        a_ca = np.array(
            net.cell_response.transmembrane_currents["agg_ica"][trial][gid][sec_name][
                seg_key
            ]
        )

        # 2. retrieve mechanism component densities (mA/cm2)
        c_na = np.array(
            net.cell_response.transmembrane_currents["ina_hh2"][trial][gid][sec_name][
                seg_key
            ]
        )
        c_k = (
            np.array(
                net.cell_response.transmembrane_currents["ik_hh2"][trial][gid][
                    sec_name
                ][seg_key]
            )
            + np.array(
                net.cell_response.transmembrane_currents["ik_kca"][trial][gid][
                    sec_name
                ][seg_key]
            )
            + np.array(
                net.cell_response.transmembrane_currents["ik_km"][trial][gid][sec_name][
                    seg_key
                ]
            )
        )
        c_ca = np.array(
            net.cell_response.transmembrane_currents["ica_ca"][trial][gid][sec_name][
                seg_key
            ]
        )

        # 3. retrieve nonspecific densities (mA/cm2)
        i_ns = (
            np.array(
                net.cell_response.transmembrane_currents["il_hh2"][trial][gid][
                    sec_name
                ][seg_key]
            )
            + np.array(
                net.cell_response.transmembrane_currents["i_ar"][trial][gid][sec_name][
                    seg_key
                ]
            )
            + np.array(
                net.cell_response.transmembrane_currents["ica_cat"][trial][gid][
                    sec_name
                ][seg_key]
            )
        )

        # 4. retrieve ground truth membrane current (nA) and synapses (nA)
        imem = np.array(
            net.cell_response.transmembrane_currents["agg_i_mem"][trial][gid][sec_name][
                seg_key
            ]
        )
        isec_data = net.cell_response.isec[trial][gid].get(sec_name, {})
        isec = (
            np.sum([np.array(v) for v in isec_data.values()], axis=0)
            if isec_data
            else 0.0
        )

        # 5. calculate balance waveforms (ignoring t=0)
        total_comp_nA = (icap + a_na + a_k + a_ca + i_ns) * scale + isec
        global_residual_waveform = imem[1:] - total_comp_nA[1:]

        res_global = np.max(np.abs(global_residual_waveform))
        rmse_global = np.sqrt(np.mean(global_residual_waveform**2))

        # ion pool balances (Truth Aggregate - Component Sum)
        res_na = np.max(np.abs(a_na[1:] - c_na[1:]))
        res_k = np.max(np.abs(a_k[1:] - c_k[1:]))
        res_ca = np.max(np.abs(a_ca[1:] - c_ca[1:]))

        n_spikes = np.sum(trial_spike_gids == gid)

        # 6. printing logic: every 10th OR aberrant
        if i % 10 == 0 or res_global > threshold:
            print(
                f"{gid:<5} | {n_spikes:<6} | {rmse_global:<12.4e} | "
                f"{res_global:<12.4e} | {res_na:<10.4e} | {res_k:<10.4e} | "
                f"{res_ca:<10.4e}"
            )

        results.append(
            {
                "gid": gid,
                "spikes": n_spikes,
                "rmse_global": rmse_global,
                "res_global": res_global,
                "res_na": res_na,
                "res_k": res_k,
                "res_ca": res_ca,
            }
        )

    return results


# ------------------------------
# check all point processes
# ------------------------------


def check_point_processes(cell_type="L5Pyr"):
    print(f"{'HOC Section Name':<30} | {'Object Type':<15} | {'Current (i)':<10}")
    print("-" * 60)

    # Iterate through every segment of every section in NEURON
    for sec in h.allsec():
        # Only look at the first L5 pyramidal cell
        if cell_type in sec.name():
            for seg in sec:
                # Look at all point processes on this segment
                for pp in seg.point_processes():
                    # We are looking for things that have a current 'i'
                    # (Synapses, TonicBias, etc.)
                    if hasattr(pp, "i"):
                        # Get the class name (e.g., 'NMDA', 'Exp2Syn')
                        obj_type = pp.hname().split("[")[0]
                        print(f"{sec.name():<30} | {obj_type:<15} | {pp.i:.4e}")


# ------------------------------
# check point processes in a section
# ------------------------------


def check_sectin_pp(net):
    # Fixed NameError: access via net metadata
    cell_type = "L5_pyramidal"
    cell_obj = net.cell_types[cell_type]["cell_object"]
    sec = "apical_2"

    print(f"--- Point Processes on {sec} ---")
    # This shows every point process (Synapses, IClamps, TonicBias)
    for syn in cell_obj.sections[sec].syns:
        # Check the name of the HOC object (e.g., 'Exp2Syn', 'TonicBias')
        print(f" - {syn}")

    # Now check what is actually inside your isec recording for that section
    gid = net.gid_ranges[cell_type][0]
    trial = 0
    recorded_syns = list(net.cell_response.isec[trial][gid].get(sec, {}).keys())
    print(f"\nReceptors recorded in isec for {sec}:")
    print(recorded_syns)

    # Find the apical_2 section
    target_sec = next((s for s in h.allsec() if sec in s.name()), None)

    if target_sec:
        print(f"\n--- Synapse Placement on {target_sec.name()} ---")
        # Iterate through all segments to see where point processes are attached
        for seg in target_sec:
            pps = seg.point_processes()
            if pps:
                print(f"\nSegment at x={seg.x}:")
                for pp in pps:
                    # Get the class name and any current variable
                    obj_name = pp.hname().split("[")[0]
                    current_val = getattr(pp, "i", "No i attribute")
                    print(f"  - {obj_name} (Current: {current_val})")
    else:
        print("Dendrite section not found.")


# net = net_base
# check_sectin_pp(net)


# %%
# %% --------------------------------------------------
# [DEV] residual analysis function
# -----------------------------------------------------

# notes:
#   - function for validating the reconstruction of the dipole moment
#     from individual components vs the 'agg_i_mem' ground truth
#   - goal is to prove: Sum(Components) == agg_i_mem
#   - dipole math magnifies segment-level errors via the vertical lever arm (zi)
#
# testing for:
#   - sign conventions: check for inverted synaptic pulses
#   - missing mechanisms: check for DC offsets (leaks, h-currents)
#   - synaptic collisions: check for shape changes from overwriting dictionary keys
#   - capacitive jitter: check for high-frequency noise during spikes (agg_i_cap)


def segment_dipole_residual_analysis(
    net,
    cell_type="L5_pyramidal",
    sec="apical_2",
    seg="seg_1",
):
    """ """

    # setup
    gid = net.gid_ranges[cell_type][0]
    trial = 0

    # get the actual segment area from metadata
    cell_obj = net.cell_types[cell_type]["cell_object"]
    sec_meta = cell_obj.sections[sec]

    # formula for segment area: (PI * diam * L) / nseg
    # use nseg from the recorded data length
    nseg_sim = len(
        net.cell_response.transmembrane_currents["agg_i_mem"][trial][gid][sec]
    )
    area_um2 = (np.pi * sec_meta.diam * sec_meta.L) / nseg_sim
    scale = area_um2 * 0.01  # Converts mA/cm2 to nA

    # aggregate transmembrane current
    imem_nA = np.array(
        net.cell_response.transmembrane_currents["agg_i_mem"][trial][gid][sec][seg]
    )

    # component transmembrane currents
    icap_nA = (
        np.array(
            net.cell_response.transmembrane_currents["agg_i_cap"][trial][gid][sec][seg]
        )
        * scale
    )
    ina_nA = (
        np.array(
            net.cell_response.transmembrane_currents["agg_ina"][trial][gid][sec][seg]
        )
        * scale
    )
    ik_nA = (
        np.array(
            net.cell_response.transmembrane_currents["agg_ik"][trial][gid][sec][seg]
        )
        * scale
    )
    ica_nA = (
        np.array(
            net.cell_response.transmembrane_currents["agg_ica"][trial][gid][sec][seg]
        )
        * scale
    )

    # nonspecific transmembrane currents (nA)
    # Summing the ones we know write to "i" or "il"
    inonspec_mechs_nA = (
        np.array(
            net.cell_response.transmembrane_currents["il_hh2"][trial][gid][sec][seg]
        )
        + np.array(
            net.cell_response.transmembrane_currents["i_ar"][trial][gid][sec][seg]
        )
        + np.array(
            net.cell_response.transmembrane_currents["ica_cat"][trial][gid][sec][seg]
        )
    ) * scale

    # 6. Synaptic Components (nA)
    isec_data = net.cell_response.isec[trial][gid].get(sec, {})
    isec_nA = (
        np.sum([np.array(v) for v in isec_data.values()], axis=0) if isec_data else 0.0
    )

    # --- THE FINAL EQUATION ---
    reconstructed_total_nA = (
        icap_nA + ina_nA + ik_nA + ica_nA + inonspec_mechs_nA + isec_nA
    )
    residual = imem_nA - reconstructed_total_nA

    # printing means from [1:] to ignore initialization transients
    print(f"--- Dipole reconstruction check for {sec} {seg} ---")
    print(f"agg_i_mem (Truth):      {np.mean(imem_nA[1:]):.8e} nA")
    print(f"Reconstructed Sum:      {np.mean(reconstructed_total_nA[1:]):.8e} nA")
    print(f"Residual Error:         {np.mean(residual[1:]):.8e} nA")

    return residual


def plot_dipole_residual_analysis(
    net,
    cell_type="L5_pyramidal",
):
    """
    Diagnostic to compare agg_i_mem dipole against component reconstruction
    and plot the residual against recorded synaptic currents.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # assume trial 0 for diagnostics
    trial = 0
    times = net.cell_response.times

    # 1. Recreate the whole-cell dipoles
    # truth: calculated from the 'catch-all' agg_i_mem variable
    dpl_truth = postproc_dipole_from_components(
        net,
        trial=trial,
        cell_type=cell_type,
        from_components=False,
    )
    # sum: calculated by manually adding ions, leak, and synapses
    dpl_sum = postproc_dipole_from_components(
        net,
        trial=trial,
        cell_type=cell_type,
        from_components=True,
    )

    # 2. Calculate the "Missing Waveform" (the residual)
    missing_waveform = dpl_truth - dpl_sum

    # 3. Sum all synaptic recordings from isec for the specified cell_type
    total_isec = np.zeros_like(missing_waveform)
    for gid in net.gid_ranges[cell_type]:
        gid_isec = net.cell_response.isec[trial].get(gid, {})
        for sec in gid_isec:
            for rec_vec in gid_isec[sec].values():
                total_isec += np.array(rec_vec)

    # 4. Plotting
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=(12, 8),
    )

    # top plot: the error (nAm)
    ax1.plot(
        times[1:],
        missing_waveform[1:],
        label='The "Missing" Part of the Waveform (Truth - Sum)',
        color="tab:blue",
    )
    ax1.set_title(f"Dipole Residual Analysis: {cell_type}")
    ax1.set_ylabel("nAm")
    ax1.axhline(0, color="black", lw=1, alpha=0.5, linestyle="--")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # bottom plot: total synaptic current (nA)
    ax2.plot(
        times,
        total_isec,
        label="Total Recorded Synaptic Current (isec)",
        color="orange",
    )
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("nA")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig


# net = net_base
# net = net_no_local_one_drive
# _ = plot_dipole_residual_analysis(net)


# %% --------------------------------------------------
# [DEV] missing charge waveform analysis
# -----------------------------------------------------


def analyze_missing_charge_waveform(
    net,
    gid=183,
    cell_type="L5_pyramidal",
    sec_name="soma",
    trial=0,
    from_agg=True,
):
    # notes:
    #   - subtracts every known component from agg_i_mem
    #   - the resulting 'missing_charge' waveform is the objective shape of what remains
    #   - we can visualize if the error tracks with spikes, drives, or something else

    # import numpy as np
    # import matplotlib.pyplot as plt

    tm_currents = net.cell_response.transmembrane_currents

    seg_key = "seg_1"
    times = net.cell_response.times

    # get area scaling (mA/cm2 -> nA)
    cell_obj = net.cell_types[cell_type]["cell_object"]
    sec_meta = cell_obj.sections[sec_name]
    nseg = len(tm_currents["agg_i_mem"][trial][gid][sec_name])
    area_um2 = (np.pi * sec_meta.diam * sec_meta.L) / nseg
    scale = area_um2 * 0.01

    # get the agg ina, ik, ica
    # we use aggregates here because we've previously validated that they match the
    # mechanisms-specific sums
    if from_agg:
        ina_nA = np.array(tm_currents["agg_ina"][trial][gid][sec_name][seg_key]) * scale
        ik_nA = np.array(tm_currents["agg_ik"][trial][gid][sec_name][seg_key]) * scale
        ica_nA = np.array(tm_currents["agg_ica"][trial][gid][sec_name][seg_key]) * scale
    else:
        ina_nA = np.array(tm_currents["ina_hh2"][trial][gid][sec_name][seg_key]) * scale
        ik_nA = (
            np.array(tm_currents["ik_hh2"][trial][gid][sec_name][seg_key])
            + np.array(tm_currents["ik_kca"][trial][gid][sec_name][seg_key])
            + np.array(tm_currents["ik_km"][trial][gid][sec_name][seg_key])
        ) * scale
        ica_nA = np.array(tm_currents["ica_ca"][trial][gid][sec_name][seg_key]) * scale

    # get the capacitive current
    icap_nA = np.array(tm_currents["agg_i_cap"][trial][gid][sec_name][seg_key]) * scale

    # get non-specific currents (`NONSPECIFIC_CURRENT` in .mod files)
    ins_nA = (
        np.array(tm_currents["il_hh2"][trial][gid][sec_name][seg_key])
        + np.array(tm_currents["i_ar"][trial][gid][sec_name][seg_key])
        + np.array(tm_currents["ica_cat"][trial][gid][sec_name][seg_key])
    ) * scale

    # get synaptic currents
    isec_data = net.cell_response.isec[trial][gid].get(sec_name, {})
    isec_nA = (
        np.sum([np.array(v) for v in isec_data.values()], axis=0) if isec_data else 0.0
    )

    # calculate the missing waveform
    imem_truth = np.array(tm_currents["agg_i_mem"][trial][gid][sec_name][seg_key])

    # residual = "truth" - (Cap + Na + K + Ca + Nonspec_Mechs + Synapses)
    component_reconstruction = icap_nA + ina_nA + ik_nA + ica_nA + ins_nA + isec_nA
    missing_charge = imem_truth - component_reconstruction

    # plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        times[1:],
        component_reconstruction[1:],
        label="Reconstructed dipole",
        alpha=0.3,
        color="orange",
    )
    ax.plot(
        times[1:],
        missing_charge[1:],
        label='Residual ("missing") dipole',
    )
    ax.plot(
        times[1:],
        imem_truth[1:],
        label='Target dipole ("truth")',
        alpha=0.3,
        color="black",
    )
    ax.set_title(f"Residual for {sec_name} GID {gid}")
    ax.set_ylabel("nA")
    ax.set_xlabel("time (ms)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return missing_charge


# execute analysis
# net = net_no_local_one_drive
# missing_wave = analyze_missing_charge_waveform(net)

# %% --------------------------------------------------
# [DEV] missing charge waveform analysis
# -----------------------------------------------------


def analyze_missing_charge_waveform_shift(
    net,
    gid=183,
    cell_type="L5_pyramidal",
    sec_name="soma",
    trial=0,
    icap_shift=0,
):
    """
    Calculates the residual error waveform with an optional time-shift
    for the capacitive current.
    """
    # notes:
    #   - subtracts every known component from agg_i_mem
    #   - the resulting 'missing_charge' waveform is the objective shape of what remains
    #   - includes icap_shift to test temporal alignment between capacitance and ions

    # import numpy as np
    # import matplotlib.pyplot as plt

    seg_key = "seg_1"
    times = net.cell_response.times

    # 1. get area scaling
    cell_obj = net.cell_types[cell_type]["cell_object"]
    sec_meta = cell_obj.sections[sec_name]
    nseg = len(
        net.cell_response.transmembrane_currents["agg_i_mem"][trial][gid][sec_name]
    )
    area_um2 = (np.pi * sec_meta.diam * sec_meta.L) / nseg
    scale = area_um2 * 0.01

    # 2. retrieve the balanced "Truths" (nA)
    ina_nA = (
        np.array(
            net.cell_response.transmembrane_currents["agg_ina"][trial][gid][sec_name][
                seg_key
            ]
        )
        * scale
    )
    ik_nA = (
        np.array(
            net.cell_response.transmembrane_currents["agg_ik"][trial][gid][sec_name][
                seg_key
            ]
        )
        * scale
    )
    ica_nA = (
        np.array(
            net.cell_response.transmembrane_currents["agg_ica"][trial][gid][sec_name][
                seg_key
            ]
        )
        * scale
    )

    # 3. retrieve non-specific mechanisms (nA)
    ins_nA = (
        np.array(
            net.cell_response.transmembrane_currents["il_hh2"][trial][gid][sec_name][
                seg_key
            ]
        )
        + np.array(
            net.cell_response.transmembrane_currents["i_ar"][trial][gid][sec_name][
                seg_key
            ]
        )
        + np.array(
            net.cell_response.transmembrane_currents["ica_cat"][trial][gid][sec_name][
                seg_key
            ]
        )
    ) * scale

    # 4. retrieve synapses (nA)
    isec_data = net.cell_response.isec[trial][gid].get(sec_name, {})
    isec_nA = (
        np.sum([np.array(v) for v in isec_data.values()], axis=0) if isec_data else 0.0
    )

    # 5. handle capacitive current and optional shift
    icap_raw = (
        np.array(
            net.cell_response.transmembrane_currents["agg_i_cap"][trial][gid][sec_name][
                seg_key
            ]
        )
        * scale
    )

    if icap_shift != 0:
        # np.roll shifts elements; we zero out the wrapped-around values for rigor
        icap_nA = np.roll(icap_raw, icap_shift)
        if icap_shift > 0:
            icap_nA[:icap_shift] = 0
        else:
            icap_nA[icap_shift:] = 0
    else:
        icap_nA = icap_raw

    # 6. calculate the missing waveform
    imem_truth = np.array(
        net.cell_response.transmembrane_currents["agg_i_mem"][trial][gid][sec_name][
            seg_key
        ]
    )

    # Residual = Truth - (Cap_shifted + Sum_of_Ions + Synapses)
    missing_charge = imem_truth - (icap_nA + ina_nA + ik_nA + ica_nA + ins_nA + isec_nA)

    # 7. plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    # slice [1:] to avoid t=0 initialization artifact
    ax.plot(
        times[1:], missing_charge[1:], label=f"Missing Current (Shift={icap_shift})"
    )
    ax.plot(
        times[1:], imem_truth[1:], label="agg_i_mem (Truth)", alpha=0.3, color="black"
    )

    ax.set_title(f"Residual Waveform Analysis (GID {gid} {sec_name})")
    ax.set_ylabel("nA")
    ax.set_xlabel("Time (ms)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return missing_charge


# execute analysis
# net = net_no_local_one_drive
# missing_wave = analyze_missing_charge_waveform_shift(
#     net,
#     icap_shift=1,
# )
