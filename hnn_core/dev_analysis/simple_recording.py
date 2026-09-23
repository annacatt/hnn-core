# %% [markdown] ###########################################################
## Setup
# #########################################################################

# %%
import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython

from hnn_core import (
    JoblibBackend,
    jones_2009_model,
    simulate_dipole,
)
from hnn_core.network_models import add_erp_drives_to_jones_model

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

# %% [code]
# simulate base network

tstop = 70.0
dt = 0.1
n_trials = 1
record_method = "soma"

net = jones_2009_model()
add_erp_drives_to_jones_model(net)

tm_currents={
    "agg_i_mem": {
        "type": "derived",
        "mech": None,
        "ref": "_ref_i_membrane_",
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
    "ina_hh2": {
        "type": "ionic",
        "mech": "hh2",
        "ref": "_ref_ina",
        "per_segment": True,
        "recorded_sections": record_method,
    },
}

with JoblibBackend(n_jobs=1):
    dpls = simulate_dipole(
        net,
        tstop=tstop,
        dt=dt,
        n_trials=1,
        tm_currents=tm_currents,
    )

# %% [code]

# diagnostic check for base simulation *after* updates
post_update_diagnostics = True
print_all_sections_segments = False

if post_update_diagnostics:

    gid = 170
    channels = list(net.cell_response.tm_currents.keys())

    for channel in channels:
        print(f"Channel: {channel}")
        try:
            tm_currents = net.cell_response.tm_currents
            gids = list(tm_currents[channel]["data"][0].keys())
            print(
                "GIDs: ",
                f"{min(gids)} -> {max(gids)}",
            )
            print(
                f'Checking GID "{gid}" for channel "{channel}"'
            )
            gid_data = tm_currents[channel]["data"][0][gid]
            sections = list(
                gid_data.keys()
            )
            print(
                "\nSections: ",
                sections,
            )
            sect_0 = sections[0]
            sect_0_segs = list(
                    gid_data[sect_0].keys()
                )
            if print_all_sections_segments:
                for section in sections:
                    segments = list(
                        gid_data[section].keys()
                    )
                    print(
                        f"\nSegments on {section}: ",
                        segments,
                    )
            else:
                print(
                    f'\nSegments on "{sect_0}": ',
                    sect_0_segs,
                )
            sect_0_seg_0 = gid_data[sect_0][sect_0_segs[0]]
            print(
                f'\nData check on "{sect_0}", "{sect_0_segs[0]}": ',
                sect_0_seg_0[0:5]
            )
        except Exception:
            pass
        print(
            "\n",
            "# " + "-" * 50,
            "\n",
            sep="",
        )
