
import numpy as np

from hnn_core.cells_default import pyramidal
from hnn_core.network_builder import load_custom_mechanisms

# %% [markdown] ----------------------------------------
## Recreate dipole components from non-synaptic transmembrane currents
# %% ---------------------------------------------------

def dipole_from_nonsynaptic_tm_currents(
    net,
    trial=0,
    cell_type="L5_pyramidal",
    scaling_factor=3000,
    from_components=False,
):
    """
    Function for processing transmembrane currents to recreate the dipole moment
    calculated from the axial currents in hnn_core. This can be done from either the
    total recorded transmembrane current, or from the constituent components.

    Note: isec (the transmembrane synaptic current) is part of the total transmembrane
    current, but is *not* aggregated with the other component channel currents
    in this function. This is due to the fact that isec contains *section-specific*
    currents (since synapses are placed at the section midpoint), as opposed to
    *segment-specfic* currents, which are required for this method of recreating
    the dipole.

    Parameters
    ----------
    net : Network object
        The network object containing the simulation data.
    trial : int
        The index of the trial to use
    cell_type : str
        The cell type to process
    scaling_factor : float
        The scaling factor to apply to the dipole
    from_components : bool
        if True, use agg_i_mem to reproduce the dipole. if False, use the component
        currents for either L5_pyramidal or L2_pyramidal

    Returns
    -------
    dipole : np.ndarray
        The reconstructed dipole moment from transmembrane currents.
    """

    load_custom_mechanisms()

    # initialize variable to hold dipole data
    dipole = None

    # build a template cell to get "metadata" for sections
    template_cell = pyramidal(cell_name=cell_type)
    template_cell.build(sec_name_apical="apical_trunk")

    # get the relative endpoints for each section from the template cell
    rel_endpoints = {}
    for sec_name, sec in template_cell._nrn_sections.items():
        start = np.array([sec.x3d(0), sec.y3d(0), sec.z3d(0)])
        # sec.n3d() returns the number of 3D points along a section; essentially len()
        # so "sec.n3d() - 1" is the index of the last 3D point
        end = np.array(
            [sec.x3d(sec.n3d() - 1), sec.y3d(sec.n3d() - 1), sec.z3d(sec.n3d() - 1)]
        )
        rel_endpoints[sec_name] = (start, end)

    if not from_components:
        all_tm_channels = ["agg_i_mem"]
    else:
        if cell_type == "L5_pyramidal":
            # this information would ideally be fetched from metadata but is
            # manually defined until that structure is determined
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
        elif cell_type == "L2_pyramidal":
            all_tm_channels = [
                "agg_i_cap",
                "ina_hh2",
                "ik_hh2",
                "ik_km",
                "il_hh2",
            ]
        else:
            raise ValueError(
                f"Valid channels types for {cell_type} are not known.\n"
                "Please pass the channels types as a list of str to tm_channels"
            )

    # loop through GIDs for the cell_type of interest
    for gid in net.gid_ranges[cell_type]:
        # get the updated soma position for this instantiation of the cell
        # index of the first cell: e.g., 170 for the first L5Pyr cell
        start_index = net.gid_ranges[cell_type][0]
        # get soma position from position dictionary, which uses its own indexing
        # that does not match the GID, hence the "- start_index"
        soma_pos = np.array(net.pos_dict[cell_type][gid - start_index])

        # create a dictionary of all channel data for the cell
        cell_channels = {
            ch: net.cell_response.transmembrane_currents[ch][trial][gid]
            for ch in all_tm_channels
        }

        # get the cell sections to loop over
        # the key used shouldn't matter, but we don't want to hard code it since
        # we can pass different channels to this function, so we get it dynamically
        first_key = list(cell_channels.keys())[0]
        cell_sections = list(cell_channels[first_key].keys())

        for sec_name in cell_sections:
            # offset the start/end positions by the realized soma position for this
            # cell instantiation
            start_rel, end_rel = rel_endpoints[sec_name]
            start = start_rel + soma_pos
            end = end_rel + soma_pos

            # get the normalized segment positions along the cell section
            nseg = len(cell_channels[first_key][sec_name])
            seg_positions = [(i - 0.5) / nseg for i in range(1, nseg + 1)]

            for pos, seg_key in zip(
                seg_positions,
                cell_channels[first_key][sec_name].keys(),
            ):
                # convert the normalized position to the absolute position
                # via linear interpolation
                abs_pos = start + pos * (end - start)
                # simplification: we are using the z position only here we only
                # need the vertical component of the dipole momen
                # we do *not* need to do geometric projection (via cos_theta)
                # as we do for the dipole calculation from axial currents
                z_i = abs_pos[2]

                # sum all currents for this segment
                I_t = np.zeros_like(
                    np.array(cell_channels[first_key][sec_name][seg_key])
                )
                for ch in all_tm_channels:
                    # get channel data
                    vec = np.array(cell_channels[ch][sec_name][seg_key])

                    # get segment area and convert from µm^2 to cm^2
                    seg = template_cell._nrn_sections[sec_name](pos)
                    area_um2 = seg.area()  # µm^2
                    area_cm2 = area_um2 * 1e-8  # cm^2

                    if ch == "agg_i_mem":
                        # agg_i_mem is not recorded continuously as a density; it is
                        # recorded after each timestep. Ergo, the units conversion
                        # here is not necessary as the units are already in nA
                        #
                        # multiplying the contribution by zi in um will give us fAm,
                        # so we will later need to divide by 1e6 to convert to nAm
                        I_abs = vec
                    # convert densities (mA/cm^2) to absolute currents (mA)]
                    else:
                        I_abs = vec * area_cm2  # keep as mA

                        # [WIP]
                        # Should I flip sign for the capacitive currents? I *think*
                        # so, but I haven't found direct confirmation of this ...
                        #
                        # I ideally would want to test the sign flip empirically,
                        # and confirm that we can reproduce i_mem by summing up all
                        # of its constituent components. However, we can't get the
                        # per-segment synaptic currents since we model them as point
                        # processes at the midpoint of the section (not the segment);
                        #
                        # Ergo, we are missing the synaptic piece of the total
                        # transmembrane current needed to reproduce i_mem exactly
                        #
                        # Note: isec is the *per-synapse* current, and not the
                        # *per-segment* current that we need
                        if ch == "agg_i_cap":
                            I_abs = I_abs * 1
                            # no unit/sign changes required; this if block can
                            # safely be deleted
                        # [end WIP]

                    I_t += I_abs

                # multiple by r_i per Naess 2015 Ch 2 (simplified to zi in this case)
                # for ionic currents, we have 1 mA*um = 1 nAm (correct units)
                # for i_mem, we have nA rather than mA. and 1 nA*um = 1 fAm
                contrib = I_t * z_i

                # for agg_i_mem, divide by 1e6 to convert fAm to nAm
                if not from_components:
                    contrib = contrib / 1e6 * scaling_factor
                else:
                    contrib = contrib * scaling_factor

                if dipole is None:
                    dipole = contrib.copy()
                else:
                    dipole += contrib

    return dipole




# %% [markdown] ----------------------------------------
### Recreating dipole for the soma only
# %% ---------------------------------------------------


def postproc_soma_dipole(
    net,
    trial=0,
    cell_type="L5_pyramidal",
    scaling_factor=3000,
    from_components=False,
):
    """ """

    print("Running `soma` dipole processing function")

    # this function will only handle the "soma", as it's composed of exactly one
    # segment where pos = 0.5
    sec_name = "soma"
    seg_key = "seg_1"
    pos = 0.5

    # load custom mechanisms
    load_custom_mechanisms()

    # initialize variable to hold dipole data
    dipole = None

    # build a template cell to get "metadata" for sections
    template_cell = pyramidal(cell_name=cell_type)
    template_cell.build(sec_name_apical="apical_trunk")

    # get the relative endpoints for the soma
    rel_endpoints = {}
    sec = template_cell._nrn_sections["soma"]
    start = np.array([sec.x3d(0), sec.y3d(0), sec.z3d(0)])
    end = np.array(
        [sec.x3d(sec.n3d() - 1), sec.y3d(sec.n3d() - 1), sec.z3d(sec.n3d() - 1)]
    )
    rel_endpoints[sec_name] = (start, end)

    if not from_components:
        all_tm_channels = ["agg_i_mem"]
    else:
        if cell_type == "L5_pyramidal":
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
        elif cell_type == "L2_pyramidal":
            all_tm_channels = [
                "agg_i_cap",
                "ina_hh2",
                "ik_hh2",
                "ik_km",
                "il_hh2",
            ]
        else:
            raise ValueError(
                f"Valid channels types for {cell_type} are not known.\n"
                "Please pass the channels types as a list of str to tm_channels"
            )

    # loop through GIDs for the cell_type of interest
    for gid in net.gid_ranges[cell_type]:
        # get the updated soma position for this instantiation of the cell
        # index of the first cell: e.g., 170 for the first L5Pyr cell
        start_index = net.gid_ranges[cell_type][0]
        # get soma position from position dictionary, which uses its own indexing
        # that does not match the GID, hence the "- start_index"
        soma_pos = np.array(net.pos_dict[cell_type][gid - start_index])

        # create a dictionary of all channel data for the cell
        cell_channels = {
            ch: net.cell_response.transmembrane_currents[ch][trial][gid]
            for ch in all_tm_channels
        }

        # get the cell sections to loop over
        # the key used shouldn't matter, but we don't want to hard code it since
        # we can pass different channels to this function, so we get it dynamically
        first_key = list(cell_channels.keys())[0]

        start_rel, end_rel = rel_endpoints[sec_name]
        start = start_rel + soma_pos
        end = end_rel + soma_pos

        abs_pos = start + pos * (end - start)
        z_i = abs_pos[2]

        # sum all currents for this segment
        I_t = np.zeros_like(
            np.array(cell_channels[first_key][sec_name][seg_key]),
        )

        for ch in all_tm_channels:
            # get channel data
            vec = np.array(cell_channels[ch][sec_name][seg_key])

            # get segment area and convert from µm^2 to cm^2
            seg = template_cell._nrn_sections[sec_name](pos)
            area_um2 = seg.area()  # µm^2
            area_cm2 = area_um2 * 1e-8  # cm^2

            if ch == "agg_i_mem":
                # agg_i_mem is not recorded continuously as a density; it is
                # recorded after each timestep. Ergo, the units conversion
                # here is not necessary as the units are already in nA
                #
                # multiplying the contribution by zi in um will give us fAm,
                # so we will later need to divide by 1e6 to convert to nAm
                I_abs = vec
            # convert densities (mA/cm^2) to absolute currents (mA)]
            else:
                I_abs = vec * area_cm2  # keep as mA

            I_t += I_abs

        # arround for different structure for isec when recontructing from components
        if from_components:
            soma_isec = net.cell_response.isec[trial][gid].get(sec_name, {})
            for syn_key in soma_isec:
                # isec is measured in nA, so we need to divide by 1e6 to
                # convert nA to mA before we add to I_t
                I_t += np.array(soma_isec[syn_key]) / 1e6

        # multiple by r_i per Naess 2015 Ch 2 (simplified to zi in this case)
        # for ionic currents, we have 1 mA*um = 1 nAm (correct units)
        # for i_mem, we have nA rather than mA. and 1 nA*um = 1 fAm
        contrib = I_t * z_i

        # for agg_i_mem, divide by 1e6 to convert fAm to nAm
        if not from_components:
            contrib = contrib / 1e6 * scaling_factor
        else:
            contrib = contrib * scaling_factor

        if dipole is None:
            dipole = contrib.copy()
        else:
            dipole += contrib

    return dipole


# %% --------------------------------------------------
# [DEV] baseline function
# -----------------------------------------------------

# notes:
#   - baseline function for computing dipole from imem OR from the
#     constituent currents
#   - this function adds isec recording at the section midpoint


def dipole_from_components_v1(
    net,
    trial=0,
    cell_type="L5_pyramidal",
    scaling_factor=3000,
    from_components=False,
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # [new]
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    sections=None,
    # [end new]
):
    """ """

    print("Running `baseline v1` function")

    load_custom_mechanisms()

    # initialize variable to hold dipole data
    dipole = None

    # build a template cell to get "metadata" for sections
    template_cell = pyramidal(cell_name=cell_type)
    template_cell.build(sec_name_apical="apical_trunk")

    # get the relative endpoints for each section from the template cell
    rel_endpoints = {}
    for sec_name, sec in template_cell._nrn_sections.items():
        start = np.array([sec.x3d(0), sec.y3d(0), sec.z3d(0)])
        # sec.n3d() returns the number of 3D points along a section; essentially len()
        # so "sec.n3d() - 1" is the index of the last 3D point
        end = np.array(
            [sec.x3d(sec.n3d() - 1), sec.y3d(sec.n3d() - 1), sec.z3d(sec.n3d() - 1)]
        )
        rel_endpoints[sec_name] = (start, end)

    if not from_components:
        all_tm_channels = ["agg_i_mem"]
    else:
        if cell_type == "L5_pyramidal":
            all_tm_channels = [
                "agg_i_cap",
                # "agg_ica",
                # "agg_i_non_specific",
                # "agg_ik",
                # "agg_ina",
                "ina_hh2",
                "ik_hh2",
                "ik_kca",
                "ik_km",
                "ica_ca",
                "ica_cat",
                "il_hh2",
                "i_ar",
            ]
        elif cell_type == "L2_pyramidal":
            all_tm_channels = [
                "agg_i_cap",
                "agg_ik",
                "agg_ina",
                # "ina_hh2",
                # "ik_hh2",
                # "ik_km",
                # "il_hh2",
            ]
        else:
            raise ValueError(
                f"Valid channels types for {cell_type} are not known.\n"
                "Please pass the channels types as a list of str to tm_channels"
            )

    # loop through GIDs for the cell_type of interest
    for gid in net.gid_ranges[cell_type]:
        # get the updated soma position for this instantiation of the cell
        # index of the first cell: e.g., 170 for the first L5Pyr cell
        start_index = net.gid_ranges[cell_type][0]
        # get soma position from position dictionary, which uses its own indexing
        # that does not match the GID, hence the "- start_index"
        soma_pos = np.array(net.pos_dict[cell_type][gid - start_index])

        # create a dictionary of all channel data for the cell
        cell_channels = {
            ch: net.cell_response.transmembrane_currents[ch][trial][gid]
            for ch in all_tm_channels
        }

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # [new]
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # first_key = list(cell_channels.keys())[0]
        first_key = next(iter(cell_channels))

        if isinstance(sections, list):
            cell_sections = sections
        else:
            # get the cell sections to loop over
            # the key used shouldn't matter, but we don't want to hard code it since
            # we can pass different channels to this function, so we get it dynamically
            cell_sections = list(cell_channels[first_key].keys())
        # [end new]

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # [new] Access synaptic data for this specific GID
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        cell_syn_data = net.cell_response.isec[trial][gid]
        # [end new]

        for sec_name in cell_sections:
            # offset the start/end positions by the realized soma position for this
            # cell instantiation
            start_rel, end_rel = rel_endpoints[sec_name]
            start = start_rel + soma_pos
            end = end_rel + soma_pos

            # get the normalized segment positions along the cell section
            nseg = len(cell_channels[first_key][sec_name])

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # [new]
            # notes:
            #   - no effect
            # template_cell._nrn_sections[sec_name].nseg = nseg
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            seg_positions = [(i - 0.5) / nseg for i in range(1, nseg + 1)]

            for pos, seg_key in zip(
                seg_positions,
                cell_channels[first_key][sec_name].keys(),
            ):
                # convert the normalized position to the absolute position
                # via linear interpolation
                abs_pos = start + pos * (end - start)
                # simplification: we are using the z position only here we only
                # need the vertical component of the dipole momen
                # we do *not* need to do geometric projection (via cos_theta)
                # as we do for the dipole calculation from axial currents
                z_i = abs_pos[2]

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # [new]
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # notes:
                #   - no effect compared to z_i = abs_pos[2]
                # z_i = (start_rel + pos * (end_rel - start_rel))[2]
                # [end new]

                # sum all currents for this segment
                I_t = np.zeros_like(
                    np.array(cell_channels[first_key][sec_name][seg_key])
                )
                for ch in all_tm_channels:
                    # get channel data
                    vec = np.array(cell_channels[ch][sec_name][seg_key])

                    # get segment area and convert from µm^2 to cm^2
                    seg = template_cell._nrn_sections[sec_name](pos)
                    area_um2 = seg.area()  # µm^2
                    area_cm2 = area_um2 * 1e-8  # cm^2

                    if ch == "agg_i_mem":
                        # agg_i_mem is not recorded continuously as a density; it is
                        # recorded after each timestep. Ergo, the units conversion
                        # here is not necessary as the units are already in nA
                        #
                        # multiplying the contribution by zi in um will give us fAm,
                        # so we will later need to divide by 1e6 to convert to nAm
                        I_abs = vec
                    # convert densities (mA/cm^2) to absolute currents (mA)]
                    elif ch == "agg_i_cap":
                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        # [new]
                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        # notes:
                        #   - potential issue: i_cap units are not mA/cm2. Divide by
                        #     the needed constant to convert uA to mA.
                        #   - note: this does NOT work
                        # I_abs = (vec / 1000.0) * area_cm2
                        I_abs = vec * area_cm2  # keep as is

                    else:
                        I_abs = vec * area_cm2  # keep as mA

                    I_t += I_abs

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # [new]
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # implement isec recording only at the section midpoint (0.5)
                # to prevent overcounting as synapses are point-processes [2]
                if from_components and np.isclose(pos, 0.5):
                    if sec_name in cell_syn_data:
                        for receptor_vec in cell_syn_data[sec_name].values():
                            # convert nA to mA to match ionic units
                            I_t += np.array(receptor_vec) * 1e-6
                # [end new]

                # multiple by r_i per Naess 2015 Ch 2 (simplified to zi in this case)
                # for ionic currents, we have 1 mA*um = 1 nAm (correct units)
                # for i_mem, we have nA rather than mA. and 1 nA*um = 1 fAm
                contrib = I_t * z_i

                # for agg_i_mem, divide by 1e6 to convert fAm to nAm
                if not from_components:
                    contrib = contrib / 1e6 * scaling_factor
                else:
                    contrib = contrib * scaling_factor

                if dipole is None:
                    dipole = contrib.copy()
                else:
                    dipole += contrib

    return dipole


# example usage:
# net = net_base
# baseline_fig = dev_postproc_func_test(
#     net,
#     cell_type="L5_pyramidal",
# )



# [DEV] end baseline
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# %% --------------------------------------------------
# [DEV] test_01
# -----------------------------------------------------

# notes:
#   - this is essentially a Google Gemini refactor of the "baseline" function
#     above used for testing / iteration
#   - this version yields the same results, but it is a bit more organazied and thus
#     more readable; consider adapting this version in vF


def dipole_from_components_v1_refactor(  # noqa
    net,
    trial=0,
    cell_type="L5_pyramidal",
    scaling_factor=3000,
    from_components=False,
):
    """ """

    print("Running `test_01` function")

    load_custom_mechanisms()

    # Build a template cell
    template_cell = pyramidal(cell_name=cell_type)
    template_cell.build(sec_name_apical="apical_trunk")

    # Get relative endpoints for each section
    rel_endpoints = {}
    for sec_name, sec in template_cell._nrn_sections.items():
        start = np.array([sec.x3d(0), sec.y3d(0), sec.z3d(0)])
        end = np.array(
            [sec.x3d(sec.n3d() - 1), sec.y3d(sec.n3d() - 1), sec.z3d(sec.n3d() - 1)],
        )
        rel_endpoints[sec_name] = (start, end)

    # Define which channels to aggregate
    if not from_components:
        all_tm_channels = ["agg_i_mem"]
    else:
        if cell_type == "L5_pyramidal":
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
        elif cell_type == "L2_pyramidal":
            all_tm_channels = ["agg_i_cap", "ina_hh2", "ik_hh2", "ik_km", "il_hh2"]
        else:
            raise ValueError(f"Unknown channel list for {cell_type}")

    dipole = None

    # Loop through GIDs for the specific cell type
    for gid in net.gid_ranges[cell_type]:
        start_index = net.gid_ranges[cell_type][0]
        soma_pos = np.array(net.pos_dict[cell_type][gid - start_index])

        # Access transmembrane and synaptic recordings for this GID
        cell_tm_data = {
            ch: net.cell_response.transmembrane_currents[ch][trial][gid]
            for ch in all_tm_channels
        }
        cell_syn_data = net.cell_response.isec[trial][gid]

        # Use the first recorded channel to determine the sections present
        first_ch = all_tm_channels[0]
        for sec_name in cell_tm_data[first_ch].keys():
            start_rel, end_rel = rel_endpoints[sec_name]
            start = start_rel + soma_pos
            end = end_rel + soma_pos

            # --- 1. Process Segment-based Currents (Ionic, Capacitive, or agg_i_mem)
            nseg = len(cell_tm_data[first_ch][sec_name])
            seg_indices = list(cell_tm_data[first_ch][sec_name].keys())

            # Segment centers are at (i - 0.5) / nseg
            for i, seg_key in enumerate(seg_indices):
                pos = (i + 0.5) / nseg
                abs_pos = start + pos * (end - start)
                z_i = abs_pos[2]  # Vertical component

                for ch in all_tm_channels:
                    vec = np.array(cell_tm_data[ch][sec_name][seg_key])

                    if dipole is None:
                        dipole = np.zeros_like(vec)

                    if ch == "agg_i_mem":
                        # agg_i_mem is recorded in nA.
                        # (nA * um) / 1e6 = nAm
                        dipole += (vec * z_i) / 1e6 * scaling_factor
                    else:
                        # Densities (mA/cm^2) must be converted to absolute current (mA)
                        # mA * um = nAm (no 1e6 division needed)
                        seg_nrn = template_cell._nrn_sections[sec_name](pos)
                        area_cm2 = seg_nrn.area() * 1e-8

                        I_abs = vec * area_cm2
                        dipole += (I_abs * z_i) * scaling_factor

            # --- 2. Process Section-based Point Currents (isec) ---
            # Synapses are modeled as point processes at the section midpoint (0.5)
            if from_components and sec_name in cell_syn_data:
                mid_pos = start + 0.5 * (end - start)
                z_mid = mid_pos[2]

                for receptor_vec in cell_syn_data[sec_name].values():
                    # isec is recorded in nA.
                    # (nA * um) / 1e6 = nAm
                    I_syn_nA = np.array(receptor_vec)
                    dipole += (I_syn_nA * z_mid) / 1e6 * scaling_factor

    return dipole


# test_01_fig = dev_postproc_func_test()

# [DEV] end test_01
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# %% --------------------------------------------------
# [DEV] test_02
# -----------------------------------------------------

# notes:
#   - simulation scaling applied to the template
#   - synced nseg so area() is segment-specific
#   - add isec
#   - this version yields the same results as `baseline`


def dipole_from_components_v2(  # noqa
    net,
    trial=0,
    cell_type="L5_pyramidal",
    scaling_factor=3000,
    from_components=False,
):
    print("Running `test_02` function")

    load_custom_mechanisms()

    # Build a template cell
    template_cell = pyramidal(cell_name=cell_type)
    template_cell.build(sec_name_apical="apical_trunk")

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # [new] apply simulation scaling to the template
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    p_prefix = "L5Pyr" if "L5" in cell_type else "L2Pyr"
    rel_endpoints = {}
    for sec_name, sec in template_cell._nrn_sections.items():
        # Get scaled dimensions from net._params
        new_L = net._params.get(f"{p_prefix}_{sec_name}_L", sec.L)
        sec.diam = net._params.get(f"{p_prefix}_{sec_name}_diam", sec.diam)

        start = np.array([sec.x3d(0), sec.y3d(0), sec.z3d(0)])
        end_orig = np.array(
            [sec.x3d(sec.n3d() - 1), sec.y3d(sec.n3d() - 1), sec.z3d(sec.n3d() - 1)]
        )

        # Adjust endpoint and internal Length to match simulation scaling
        if sec.L > 0:
            end = start + (end_orig - start) * (new_L / sec.L)
            sec.L = new_L
        else:
            end = end_orig
        rel_endpoints[sec_name] = (start, end)
    # [end new]

    # Define which channels to aggregate
    if not from_components:
        all_tm_channels = ["agg_i_mem"]
    else:
        if cell_type == "L5_pyramidal":
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
        elif cell_type == "L2_pyramidal":
            all_tm_channels = ["agg_i_cap", "ina_hh2", "ik_hh2", "ik_km", "il_hh2",]
        else:
            raise ValueError(f"Unknown channel list for {cell_type}")

    dipole = None

    for gid in net.gid_ranges[cell_type]:
        start_index = net.gid_ranges[cell_type][0]
        soma_pos = np.array(net.pos_dict[cell_type][gid - start_index])

        cell_tm_data = {
            ch: net.cell_response.transmembrane_currents[ch][trial][gid]
            for ch in all_tm_channels
        }
        cell_syn_data = net.cell_response.isec[trial][gid]

        first_ch = all_tm_channels[0]
        for sec_name in cell_tm_data[first_ch].keys():
            start_rel, end_rel = rel_endpoints[sec_name]
            start = start_rel + soma_pos
            end = end_rel + soma_pos

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # [new] Sync nseg so area() is segment-specific
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            nseg = len(cell_tm_data[first_ch][sec_name])
            template_cell._nrn_sections[sec_name].nseg = nseg
            # [end new]

            seg_indices = list(cell_tm_data[first_ch][sec_name].keys())
            for i, seg_key in enumerate(seg_indices):
                pos = (i + 0.5) / nseg
                abs_pos = start + pos * (end - start)
                z_i = abs_pos[2]

                for ch in all_tm_channels:
                    vec = np.array(cell_tm_data[ch][sec_name][seg_key])

                    if dipole is None:
                        dipole = np.zeros_like(vec)

                    if ch == "agg_i_mem":
                        dipole += (vec * z_i) / 1e6 * scaling_factor
                    else:
                        seg_nrn = template_cell._nrn_sections[sec_name](pos)
                        area_cm2 = seg_nrn.area() * 1e-8
                        I_abs = vec * area_cm2
                        dipole += (I_abs * z_i) * scaling_factor

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # [new] add isec once per section
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # This ensures synaptic currents are included even if nseg is even
            if from_components and sec_name in cell_syn_data:
                mid_pos = start + 0.5 * (end - start)
                z_mid = mid_pos[2]
                for receptor_vec in cell_syn_data[sec_name].values():
                    I_syn_nA = np.array(receptor_vec)
                    dipole += (I_syn_nA * z_mid) / 1e6 * scaling_factor
            # [end new]

    return dipole


# test_02_fig = dev_postproc_func_test()
