import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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

# %% [markdown] ----------------------------------------
# plot cell morphology
# %% ---------------------------------------------------

def extract_diameters(
    section_names,
    cell_params,
):
    """
    Helper to map section names to their corresponding diameters as specified in
    params_default.py
    """
    extracted = {}
    for name in section_names:
        # adjust the section names to match the default keys
        search_key = name.replace("_", "").lower()

        # look for a key in defaults that contains search_key and "diam"
        found = False

        for cell_param, cell_val in cell_params.items():

            clean_cell_param = cell_param.lower()

            if search_key in clean_cell_param and "diam" in clean_cell_param:
                extracted[name] = cell_val
                found = True
                break

        if not found:
            raise KeyError(
                f"Could not find diameter for section '{name}' in cell_params"
            )

    return extracted


def plot_flat_neuron(
    end_pts,
    defaults,
    x_offsets=None,
    gap=0,
    colors=None,
    figsize=(6, 12),
    width_scale=1.0,
):
    # get diameters to set widths
    diameters = extract_diameters(
        end_pts.keys(),
        defaults,
    )

    # handle horizontal offsets
    x_offs = x_offsets if x_offsets else {k: 0 for k in end_pts.keys()}
    y_shift = {k: 0 for k in end_pts.keys()}

    # handle vertical offsets ("gap")

    # sort sections by how close they are to soma
    # determine the order in which vertical gaps are processed, since the gaps need
    # to be accumulated as you move away from the soma towards the ends of the
    # dendrites. E.g.:
    # - "soma" is the parent of "apical_trunk", which is the parent of "apical_1" ...
    sorted_sections = sorted(
        # sections
        end_pts.keys(),
        # absolute value of distance from soma on the Z axis
        key=lambda x: abs(end_pts[x][0][2]),
    )

    for child_section in sorted_sections:
        child_section_start = end_pts[child_section][0]

        # set shift for basal_1, which shoud move in the negative direction along
        # the Z axis
        if child_section != "soma" and child_section_start == end_pts["soma"][0]:
            y_shift[child_section] = -gap

        # as we move out from the soma, inherit the gap from the parent section
        for parent_section in end_pts.keys():
            parent_section_end = end_pts[parent_section][1]
            if child_section_start == parent_section_end:
                # for apical_oblique, inherit the parent_section shift, but add 0 gap
                # this keeps it in line with apical_trunk, while allowing apical_1
                # to movie higher
                if "oblique" in child_section:
                    y_shift[child_section] = y_shift[parent_section]
                else:
                    current_gap = -gap if "basal" in child_section else gap
                    y_shift[child_section] = y_shift[parent_section] + current_gap
                break

    # use normalized diameter to set line width for sections
    diam_values = list(diameters.values())
    d_min, d_max = min(diam_values), max(diam_values)

    # allow the user to scale the width
    def diam_to_width(d):
        if d_max > d_min:
            # apply the width_scale multiplier
            base_width = 1 + (d - d_min) / (d_max - d_min) * 14
            return base_width * width_scale
        return 5 * width_scale

    plt.figure(figsize=figsize)
    legend_elements = []

    # plot geometry
    for name, seg in end_pts.items():
        dx = x_offs.get(name, 0)
        dy = y_shift.get(name, 0)

        x = [seg[0][0] + dx, seg[1][0] + dx]
        y = [seg[0][2] + dy, seg[1][2] + dy]

        width = diam_to_width(
            diameters[name],
        )
        color = colors.get(name, "b") if colors else "b"

        # "butt" (lol) capstyle ensures sections to not overlab
        plt.plot(
            x,
            y,
            color=color,
            linewidth=width,
            solid_capstyle="butt",
        )

        # use uniform line thickness in the legend
        if colors and name in colors:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    lw=2,
                    label=name,
                ),
            )

        plt.text(
            x[1],
            y[1],
            name,
            fontsize=8,
            va="bottom",
            ha="center",
        )

    plt.axis("off")
    if colors:
        plt.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=8,
        )
    plt.show()

# defaults extracted from params_default.py for L5Pyr, but can be grabbed from
# net._params

cell_type = "L5_pyramidal"
default_cell_searchkey = net.cell_types[cell_type]["cell_object"].name
celltype_sections = list(
    net.cell_types["L5_pyramidal"]["cell_object"].sections.keys(),
)

# reduce _params down to current cell_type only
default_cell_keymatches = [
    key
    for key in list(net._params.keys())
    if default_cell_searchkey in key
]

defaults = {
    key: value for key, value in net._params.items() if key in default_cell_keymatches
}



# values pulled directly from _cell_L5Pyr in cells_default
end_pts = {
    "soma": [[0, 0, 0], [0, 0, 23]],
    "apical_trunk": [[0, 0, 23], [0, 0, 83]],
    "apical_oblique": [[0, 0, 83], [-150, 0, 83]],
    "apical_1": [[0, 0, 83], [0, 0, 483]],
    "apical_2": [[0, 0, 483], [0, 0, 883]],
    "apical_tuft": [[0, 0, 883], [0, 0, 1133]],
    "basal_1": [[0, 0, 0], [0, 0, -50]],
    "basal_2": [[0, 0, -50], [-106, 0, -156]],
    "basal_3": [[0, 0, -50], [106, 0, -156]],
}

# endpoints grabbed from procedurally-generated cell
procedural_end_pts = {}
for section in celltype_sections:
    procedural_end_pts[section] = (
        net.cell_types[cell_type]["cell_object"].sections[section].end_pts
    )

x_offsets = {
    "apical_oblique": -5,
    "apical_1": 0,
    "apical_2": 0,
    "apical_tuft": 0,
    "basal_2": -5,
    "basal_3": 5,
}


distinct_colors = {
    "soma": "orange",
    "apical_trunk": "red",
    "apical_oblique": "purple",
    "apical_1": "magenta",
    "apical_2": "pink",
    "apical_tuft": "brown",
    "basal_1": "yellow",
    "basal_2": "olive",
    "basal_3": "darkgreen",
}

blues = {
    "soma": "#4895ef",
    "apical_trunk": "#4cc9f0",
    "apical_oblique": "#caf0f8",
    "apical_1": "#90e0ef",
    "apical_2": "#ade8f4",
    "apical_tuft": "#e0f7fa",
    "basal_1": "#4cc9f0",
    "basal_2": "#90e0ef",
    "basal_3": "#ade8f4",
}

section_names = end_pts.keys()  # 'soma', 'apical_trunk', etc.
diameters_clean = extract_diameters(section_names, defaults)
print(diameters_clean)

plot_flat_neuron(
    procedural_end_pts,
    defaults,
    x_offsets=x_offsets,
    gap=0,
    colors=distinct_colors,
    width_scale=2,
)
