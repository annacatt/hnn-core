# %% [markdown] ###########################################################
## Setup
# #########################################################################

# %%
import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython

from hnn_core.dev_analysis.base_functions import (
    analyze_missing_charge_waveform,
    analyze_population_ion_balance,
    cell_specific_dipole_reconstruction,
    check_nrmse_bt_computed_dipoles,
    init_net_base,
    init_net_no_local_one_drive,
    postproc_dipole_from_components,
    postproc_soma_dipole,
    segment_dipole_residual_analysis,
    validate_geometry_and_segment_scaling,
)
from hnn_core.dev_analysis.plot_neuron_morphology import (
    get_celltype_plot_params,
    plot_flat_neuron,
)

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

# %%

if "net_base" not in locals():
    net_base = init_net_base()

if "net_no_local_one_drive" not in locals():
    net_no_local_one_drive = init_net_no_local_one_drive(dt=0.01)
    # net_no_local_one_drive = init_net_no_local_one_drive(dt=.005)
    # net_no_local_one_drive = init_net_no_local_one_drive()

# %%


def get_agg_component_dipoles_for_soma(net):
    from_agg = postproc_soma_dipole(
        net=net,
        from_components=False,
    )

    from_components = postproc_soma_dipole(
        net=net,
        from_components=True,
    )

    return from_agg, from_components


def get_agg_component_dipoles(
    net,
    sections=None,
):
    from_agg = postproc_dipole_from_components(
        net=net,
        from_components=False,
        sections=sections,
    )

    from_components = postproc_dipole_from_components(
        net=net,
        from_components=True,
        sections=sections,
    )

    return from_agg, from_components


def plot_agg_component_dipoles(
    from_agg,
    from_components,
    tstep_delay=1,
):
    fig, ax = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=(8, 15),
    )

    from_agg = from_agg[tstep_delay:]
    from_components = from_components[tstep_delay:]

    # get array of times from from_agg should be same length dpls but need to
    # divide by sampling_rate to get correct time values
    sampling_rate = 40
    times = np.arange(len(from_agg)) / sampling_rate

    # calculate residual and rmse
    residual = from_agg - from_components
    rmse = np.sqrt(np.mean(residual**2))

    # version 1: normalize by the peak-to-peak range of from_agg
    # dpl_range = np.max(from_agg) - np.min(from_agg)
    # nrmse_pct = (rmse / dpl_range) * 100 if dpl_range != 0 else 0

    # version 2: signal standard deviation (NRMSE = RMSE / signal_std)
    #   - read as: "error relative to natural variability of the aggregate signal"
    signal_std = np.std(from_agg)
    nrmse_pct = (rmse / signal_std) * 100 if signal_std != 0 else 0

    ax[0].plot(
        times,
        from_agg,
        label="From Aggregate TM currents",
    )

    ax[0].plot(
        times,
        from_components,
        alpha=0.5,
        # add legend
        label="From Component TM currents",
    )

    combined_data = np.concatenate(
        [from_agg, from_components],
    )

    y_min, y_max = np.min(combined_data), np.max(combined_data)
    padding = 0.15 * (y_max - y_min)

    for axis in ax:
        axis.set_ylim(y_min - padding, y_max + padding)

    ax[0].legend(loc="upper left")

    # text box for rmse
    error_text = f" RMSE: {rmse:.2f}\nNRMSE: {nrmse_pct:.2f}%"
    ax[0].text(
        x=0.02,
        y=0.02,
        s=error_text,
        transform=ax[0].transAxes,
        fontsize=10,
        fontfamily="Menlo",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.2,
        ),
    )

    ax[0].set_ylabel("nAm")
    ax[0].grid(True, alpha=0.3)
    ax[0].set_title(
        "Dipoles computes from aggregate transmembrane (TM) currents"
        "\nand from component transmembrane currents"
    )

    # residual dipole plot
    ylim = (max(abs(y_min), abs(y_max))) * 1.05

    ax[1].plot(
        times,
        residual,
        color="red",
        label="(Agg - Component), scaled",
    )
    ax[1].set_title(
        "Residual (Error)",
    )
    ax[1].set_ylabel("nAm")
    ax[1].set_ylim(-ylim, ylim)
    ax[1].grid(True, alpha=0.3)

    rightax = ax[1].twinx()
    rightax.plot(
        times,
        residual,
        alpha=0.2,
        label="(Agg - Component), zoomed",
    )
    ymin, ymax = rightax.get_ylim()
    ylim = max(abs(ymin), abs(ymax))
    rightax.set_ylim(-ylim, ylim)

    lines1, labels1 = ax[1].get_legend_handles_labels()
    lines2, labels2 = rightax.get_legend_handles_labels()

    ax[1].legend(lines1 + lines2, labels1 + labels2)

    ax[1].set_title("Residual (Error)")

    return fig


# %% [markdown] ###########################################################
## Analysis
# #########################################################################

# %%
# %matplotlib widget

from_agg, from_components = get_agg_component_dipoles_for_soma(net_base)

_ = plot_agg_component_dipoles(
    from_agg,
    from_components,
)

from_agg, from_components = get_agg_component_dipoles_for_soma(net_no_local_one_drive)

_ = plot_agg_component_dipoles(
    from_agg,
    from_components,
)

# %%
from_agg, from_components = get_agg_component_dipoles(
    net_base,
    sections=["soma"],
)

_ = plot_agg_component_dipoles(
    from_agg,
    from_components,
)

from_agg, from_components = get_agg_component_dipoles(
    net_no_local_one_drive,
    sections=["soma"],
)

_ = plot_agg_component_dipoles(
    from_agg,
    from_components,
)

# %% --------------------------------------------------
# [DEV] validation report
# -----------------------------------------------------

# notes:
#   - wrapper function to run the full suite of diagnostics
#       1. Area and nseg validation
#       2. Point processes
#       3. Ion pool balances
#       4. Recording synchronization
#       5. Population performance (Corr, Gain, Spikes per GID)


def run_comprehensive_text_report(
    net,
    cell_type="L5_pyramidal",
    sec_name="soma",
    trial=0,
    tstep_delay=1,
    plot_dipole_comparisons=False,
):
    # find first non-spiking and first spiking GID
    # gids = net.gid_ranges[cell_type]
    # trial_spike_gids = np.array(net.cell_response.spike_gids[trial])
    # gid_non_spiking = next((g for g in gids if g not in trial_spike_gids), None)
    # gid_spiking = next((g for g in gids if g in trial_spike_gids), None)

    print("#" + "=" * 70)
    print(f"# Dipole validation report for {cell_type}, {sec_name}")
    print("#" + "=" * 70)

    print("\n" + "#" + "-" * 50)
    print("# CHECK NRMSE BETWEEN DIPOLES")
    print("#" + "-" * 50)

    from_agg, from_components = get_agg_component_dipoles(
        net,
        sections=[sec_name],
    )
    from_agg = from_agg[tstep_delay:]
    from_components = from_components[tstep_delay:]

    _, _ = check_nrmse_bt_computed_dipoles(
        from_agg=from_agg,
        from_components=from_components,
        verbose=True,
    )

    if plot_dipole_comparisons:
        _ = plot_agg_component_dipoles(
            from_agg,
            from_components,
        )

    # TODO: add for segment loop here for when sec_name != "soma"
    # SEGMENT DIPOLE RESIDUAL CHECK
    print("\n" + "#" + "-" * 50)
    print("# SEGMENT DIPOLE RESIDUAL CHECK")
    print("#" + "-" * 50)
    _ = segment_dipole_residual_analysis(
        net, cell_type=cell_type, sec=sec_name, seg="seg_1"
    )

    # POPULATION PERFORMANCE
    # Waveform Corr and Gain metrics linked to Spike Counts
    print("\n" + "#" + "-" * 50)
    print("# CELL-SPECIFIC DIPOLE CHECKS")
    print("#" + "-" * 50)
    _ = cell_specific_dipole_reconstruction(net, cell_type=cell_type, trial=trial)

    # AREA / GEOMETRY VALIDATION
    # Validate simulation area vs template area and nseg assumptions
    print("\n" + "#" + "-" * 50)
    print("# AREA AND GEOMETRY VALIDATION")
    print("#" + "-" * 50)
    validate_geometry_and_segment_scaling(net, cell_type=cell_type)

    # POINT PROCESS CHECK
    # Check that point processes match what we expect
    print("\n" + "#" + "-" * 50)
    print("# CHECK POINT PROCESSES")
    print("#" + "-" * 50)
    cell_obj = net.cell_types[cell_type]["cell_object"]
    print(f"Section Locations: {cell_obj.sect_loc}")
    print(f"Attached to {sec_name}: {cell_obj.sections[sec_name].syns}")

    # ION BALANCE CHECK
    # Verify that Na, K, and Ca aggregates match the component sums
    print("\n" + "#" + "-" * 50)
    print("# CHECK ION BALANCES (AGGREGATES vs COMPONENTS)")
    print("#" + "-" * 50)
    _ = analyze_population_ion_balance(
        net,
        cell_type=cell_type,
        sec_name=sec_name,
        trial=trial,
        threshold=1e-12,  # Strict threshold for species pools
    )

    print("\n" + "=" * 70)
    print("END OF VALIDATION REPORT")
    print("=" * 70)


# %%

# execute validation
# run_comprehensive_text_report(net_no_local_one_drive)

run_comprehensive_text_report(
    net_no_local_one_drive,
    plot_dipole_comparisons=True,
)


net = net_no_local_one_drive

analyze_missing_charge_waveform(
    net,
    gid=183,
)
analyze_missing_charge_waveform(
    net,
    gid=173,
)

# notes:
#   - look at integration method for numerical differences
#   - might try really small dt or a different integration method


# %%

# %% --------------------------------------------------
# [DEV] plot dipole comparison by section
# -----------------------------------------------------

# notes:
#   - function to visualize the dipole reconstruction accuracy
#     on a per-section basis
#   - overlays 'from_agg' and 'from_components' for each section
#   - follows top-to-bottom anatomical order


def plot_dipole_reconstruction_by_section(
    net,
    tstep_delay=1,
    cell_type="L5_pyramidal",
    end_pts=None,
    default_cell_params=None,
    show_neuron_previews=False,
):
    """
    Plots dipole comparisons for each section with independent y-axis scaling.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # top to bottom section order
    section_plot_order = [
        "apical_tuft",
        "apical_2",
        "apical_1",
        "apical_trunk",
        "apical_oblique",
        "soma",
        "basal_1",
        "basal_2",
        "basal_3",
    ]

    # identify sections present in the data for this cell type
    gid = net.gid_ranges[cell_type][0]
    recorded_sections = list(
        net.cell_response.transmembrane_currents["agg_i_mem"][0][gid].keys()
    )
    
    cell_sections = [sec for sec in section_plot_order if sec in recorded_sections]

    # determine grid dimensions
    num_cols = 2 if show_neuron_previews else 1
    width_ratios = [1, 4] if show_neuron_previews else [1]

    fig, axes = plt.subplots(
        nrows=len(cell_sections),
        ncols=num_cols,
        sharex="col",
        figsize=(
            12 if show_neuron_previews else 8,
            3 * len(cell_sections),
        ),
        gridspec_kw={"width_ratios": width_ratios},
    )

    # ensure axes array is 2D
    if len(cell_sections) == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_cols == 1:
        axes = np.expand_dims(axes, axis=-1)

    # 1. handle morphology parameters
    if show_neuron_previews:
        if end_pts is None or default_cell_params is None:
            end_pts, default_cell_params = get_celltype_plot_params(
                net, 
                cell_type=cell_type
            )

    x_offsets = {
        "apical_oblique": -10,
        "basal_2": -10,
        "basal_3": 10,
    }

    # 2. plotting loop
    times = net.cell_response.times[tstep_delay:]
    
    for i, sec_name in enumerate(cell_sections):
        # calculate dipoles for this specific section
        from_agg, from_comp = get_agg_component_dipoles(
            net,
            sections=[sec_name],
        )
        d_agg = from_agg[tstep_delay:]
        d_comp = from_comp[tstep_delay:]

        # calculate NRMSE for title
        residual = d_agg - d_comp
        rmse = np.sqrt(np.mean(residual**2))
        sig_std = np.std(d_agg)
        nrmse = (rmse / sig_std) * 100 if sig_std != 0 else 0

        # plot neuron morphology preview in first column
        if show_neuron_previews:
            neuron_colors = {k: "lightgrey" for k in end_pts.keys()}
            neuron_colors[sec_name] = "#004a9e"

            plot_flat_neuron(
                end_pts,
                default_cell_params,
                x_offsets=x_offsets,
                gap=10,
                colors=neuron_colors,
                ax=axes[i, 0],
                show_labels=False,
                width_scale=1.5,
            )

        # plot section dipoles in data column
        col_idx = 1 if show_neuron_previews else 0
        ax_plot = axes[i, col_idx]
        
        ax_plot.plot(times, d_agg, label="from imem", color="black", alpha=0.7)
        ax_plot.plot(times, d_comp, label="from components", color="tab:red", linestyle="--")
        
        # calculate LOCAL y-limits for this specific section
        local_min = min(d_agg.min(), d_comp.min())
        local_max = max(d_agg.max(), d_comp.max())
        local_pad = 0.1 * (local_max - local_min) if local_max > local_min else 0.1
        
        ax_plot.set_ylim(local_min - local_pad, local_max + local_pad)
        
        ax_plot.set_title(
            f"{sec_name.replace('_', ' ').title()} (NRMSE: {nrmse:.2f}%)",
            fontsize=12
        )
        ax_plot.set_ylabel("nAm")
        ax_plot.grid(True, alpha=0.3)
        
        if i == 0:
            ax_plot.legend(loc="upper right", fontsize=8)

    celltype_title = cell_type.replace("_", " ").title()
    fig.suptitle(
        f"Sectional Dipole Reconstruction Validation: {celltype_title}\n(Independent Y-Scaling)",
        fontsize=16,
        y=0.99
    )
    
    plt.xlabel("Time (ms)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig



_ = plot_dipole_reconstruction_by_section(
    net=net_no_local_one_drive,
    tstep_delay=1,
    cell_type="L5_pyramidal",
    show_neuron_previews=True,
)
