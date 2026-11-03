# Copied from hnn-tuning (jonescompneurolab/hnn-tuning, hnn_tuning/cell_builder_utils.py,
# commit fac61b4) so runscripts-rt does not depend on installing hnn-tuning.
# Unmodified below this header; re-copy if the original changes.

"""Utils to build individual pyramidal cells. 

Code taken and adapted from HNN-core.

Author: Katharina Duecker katharina.duecker@gmail.com
Last checked: 18-Oct-2026"""

import neuron
from neuron import h
from neuron.units import ms, mV
import numpy as np
from numpy.linalg import norm


# number of segments
def _get_nseg(L):
    nseg = 1
    if L > 100.:  # 100 um
        nseg = int(L / 50.)
        # make dend.nseg odd for all sections
        if not nseg % 2:
            nseg += 1
    return nseg

# get dendrites
def _get_dends(params, cell_type, section_names, end_pts, v_init = {'all': -65}, cai=5e-5):
    """Convert a flat dictionary to a nested dictionary.

    Returns
    -------
    sections : dict
        Dictionary of sections. Keys are section names
    """
    prop_names = ['L', 'diam', 'Ra', 'cm']
    sections = dict()
    for section_name in section_names:
        end_pt = end_pts[section_name]
        dend_prop = dict()
        for key in prop_names:
            if key in ['Ra', 'cm']:
                middle = 'dend'
            else:
                # map apicaltrunk -> apical_trunk etc.
                middle = section_name.replace('_', '')
            dend_prop[key] = params[f'{cell_type}_{middle}_{key}']
            if len(v_init) == 1:
                v = v_init['all']
            else:
                v = v_init[section_name]
        sections[section_name] = Section(L=dend_prop['L'],
                                        diam=dend_prop['diam'],
                                        Ra=dend_prop['Ra'],
                                        cm=dend_prop['cm'],
                                        v = v,
                                        end_pts = end_pt)
    return sections

# get soma
def _get_pyr_soma(p_all, cell_type, end_pt, v_init = -65, cai=5e-5):
    """Get somatic properties."""
    return Section(
        L=p_all[f'{cell_type}_soma_L'],
        diam=p_all[f'{cell_type}_soma_diam'],
        cm=p_all[f'{cell_type}_soma_cm'],
        Ra=p_all[f'{cell_type}_soma_Ra'],
        v = v_init,
        end_pts = end_pt
    )

# get inhibitory neuron soma
def _get_inh_soma(end_pts=None):
    if end_pts is None:
        end_pts = [[0, 0, 0], [0, 0, 20]]

    # measures from Cumberland et al., 2023
    return Section(
        L=20.,
        diam=20.,
        cm=1.,
        Ra=200.,
        end_pts=end_pts)

# angle of dendrite relative to apical for dipole calculation
def _get_cos_theta(sections, sec_name_apical):
    """Get cos(theta) to compute dipole along the apical dendrite."""
    a = (np.array(sections[sec_name_apical].end_pts[1]) -
         np.array(sections[sec_name_apical].end_pts[0]))
    cos_thetas = dict()
    for sec_name, section in sections.items():
        b = np.array(section.end_pts[1]) - np.array(section.end_pts[0])
        cos_thetas[sec_name] = np.dot(a, b) / (norm(a) * norm(b))
    return cos_thetas
    
# sections class copied from hnn-core
class Section:
    """Section class.

    Parameters
    ----------
    L : float
        length of a section in microns.
    diam : float
        diameter of a section in microns.
    cm : float
        membrane capacitance in micro-Farads.
    Ra : float
        axial resistivity in ohm-cm
    end_pts : list of [x, y, z]
        The start and stop points of the section.

    Attributes
    ----------
    mechs : dict
        Mechanisms to insert in this section. The keys
        are the names of the mechanisms and values
        are the properties. For e.g., {'ca': {'gbar_ca': 60}}
    syns : list of str
        The synaptic mechanisms to add in this section
    end_pts : list of [x, y, z]
        The start and stop points of the section. Cannot be changed.
    L : float
        length of a section in microns.
    diam : float
        diameter of a section in microns.
    cm : float
        membrane capacitance in micro-Farads.
    Ra : float
        axial resistivity in ohm-cm.
    v : float
        start value for membrane potential.
    nseg : int
        Number of segments in the section
    """
    def __init__(self, L, diam, Ra, cm, v=-65, cai = 5e-5, end_pts=None):

        self._L = L
        self._diam = diam
        self._Ra = Ra
        self._cm = cm
        self._v = v             # initial voltage for each section
        self._cai = cai
        if end_pts is None:
            end_pts = list()
        self._end_pts = end_pts

        self.mechs = dict()
        self.syns = list()

        # For distance functionality
        self.nseg = _get_nseg(self.L)

    def __repr__(self):
        return f'L={self.L}, diam={self.diam}, cm={self.cm}, Ra={self.Ra}, v={self.v}'

    @property
    def L(self):
        return self._L

    @property
    def diam(self):
        return self._diam

    @property
    def cm(self):
        return self._cm

    @property
    def Ra(self):
        return self._Ra
    
    @property
    def v(self):
        return self._v
    @property
    def cai(self):
        return self._cai

    @property
    def end_pts(self):
        return self._end_pts

# cell class from hnn-core    
class Cell: 

    def __init__(self, sections, cell_tree, cell_type, gid):

        self._gid = gid
        self.sections = sections

        self._nrn_sections = dict()

        self.cell_tree = cell_tree
        self.cell_type = cell_type
        self.dipole_pp = list()
        
    def __repr__(self):
        return f'{self.cell_type}[{self._gid}]'
    

    def _set_biophysics(self, sections):
        "Set the biophysics for the cell."

        # neuron syntax is used to set values for mechanisms
        # sec.gbar_mech = x sets value of gbar for mech to x for all segs
        # in a section. This method is significantly faster than using
        # a for loop to iterate over all segments to set mech values

        # If value depends on distance from the soma. Soma is set as
        # origin by passing cell.soma as a sec argument to h.distance()
        # Then iterate over segment nodes of dendritic sections
        # and set attribute depending on h.distance(seg.x), which returns
        # distance from the soma to this point on the CURRENTLY ACCESSED
        # SECTION!!!
        h.distance(sec=self._nrn_sections['soma'])
        for sec_name, section in sections.items():
            sec = self._nrn_sections[sec_name]
            for mech_name, p_mech in section.mechs.items():
                sec.insert(mech_name)
                for attr, val in p_mech.items():
                    if isinstance(val, list):
                        seg_xs, seg_vals = val[0], val[1]
                        for seg, seg_x, seg_val in zip(sec, seg_xs, seg_vals):
                            setattr(seg, attr, seg_val)
                    else:
                        setattr(sec, attr, val)

    def _compute_section_mechs(self):
        sections = self.sections
        for sec_name, section in sections.items():
            for mech_name, p_mech in section.mechs.items():
                for attr, val in p_mech.items():
                    if hasattr(val, '__call__'):
                        seg_xs, seg_vals = list(), list()
                        section_distance = self.distance_section(sec_name,
                                                                 ('soma', 0))
                        seg_centers = (np.linspace(0, 1, section.nseg * 2 + 1)
                                       [1::2])

                        for seg_x in seg_centers:
                            # sec_end_dist is distance between 0 end of soma to
                            # the 0 or 1 end of section (whichever is closer)
                            sec_end_dist = section_distance - (section.L / 2)
                            seg_xs.append(seg_x)
                            seg_vals.append(val(sec_end_dist +
                                                (seg_x * section.L)))
                        p_mech[attr] = [seg_xs, seg_vals]
        return self.sections

    def _create_sections(self): 
        
        for sec_name in self.sections:
            sec = h.Section(name=f'{self.cell_type}_{sec_name}', cell=self)
            self._nrn_sections[sec_name] = sec

            h.pt3dclear(sec=sec)
            h.pt3dconst(0, sec=sec)  
            for pt in self.sections[sec_name].end_pts:
                h.pt3dadd(pt[0], pt[1], pt[2], 1, sec=sec)
            # with pt3dconst==0, these will alter the 3d points defined above!
            sec.L = self.sections[sec_name].L
            sec.diam = self.sections[sec_name].diam
            sec.cm = self.sections[sec_name].cm
            sec.Ra = self.sections[sec_name].Ra
            sec.nseg = self.sections[sec_name].nseg
            sec.v = self.sections[sec_name].v

        if self.cell_tree is None:
            self.cell_tree = dict()

        # Connects sections of THIS cell together.
        for parent_node in self.cell_tree:
            for child_node in self.cell_tree[parent_node]:
                parent_sec = self._nrn_sections[parent_node[0]]
                child_sec = self._nrn_sections[child_node[0]]
                if parent_sec == child_sec:
                    continue
                parent_loc = parent_node[1]
                child_loc = child_node[1]
                child_sec.connect(parent_sec, parent_loc, child_loc)

        h.define_shape()

    def _get_mechanisms(p_all, cell_type, section_names, mechanisms):
        """Get mechanism

        Parameters
        ----------
        cell_type : str
            The cell type
        section_names : str
            The section_names
        mechanisms : dict of list
            The mechanism properties to extract

        Returns
        -------
        mech_props : dict of dict of dict
            Nested dictionary of the form
            sections -> mechanism -> mechanism properties
            used to instantiate the mechanism in Neuron
        """
        mech_props = dict()
        for sec_name in section_names:
            this_sec_prop = dict()
            for mech_name in mechanisms:
                this_mech_prop = dict()
                for mech_attr in mechanisms[mech_name]:
                    if sec_name == 'soma':
                        key = f'{cell_type}_soma_{mech_attr}'
                    elif 'basal' in sec_name or sec_name == 'apical_oblique':
                        key = f'{cell_type}_basal_{mech_attr}'
                    else:
                        key = f'{cell_type}_dend_{mech_attr}'
                    this_mech_prop[mech_attr] = p_all[key]
                this_sec_prop[mech_name] = this_mech_prop
            mech_props[sec_name] = this_sec_prop
        return mech_props

    # heavily adjusted to explore synapses of interest
    def _make_synapse(self, name, loc):
        """Create and configure one synapse of the given type at section `loc`."""
        seg = self._nrn_sections[loc](0.5)
        if name == 'Exp2Syn':
            syn = h.Exp2Syn(seg)
            syn.tau1, syn.tau2, syn.e = 0.5, 5.0, 0.0
        elif name == 'NMDA_gao2021':
            syn = h.NMDA_gao2021(seg)
        elif name == 'gabab_neymotin2016':
            syn = h.gabab_neymotin2016(seg)   # kinetic model; uses its own defaults
        elif name == 'gabaa_slow':
            syn = h.Exp2Syn(seg)
            syn.e, syn.tau1, syn.tau2 = -80, 1.5, 20
        else:
            raise ValueError(f'unknown synapse type: {name!r}')
        return syn

    def _create_synapses(self, add_syn=None, syn_loc=None):
        """Attach synapse(s) to the cell.

        `add_syn` / `syn_loc` may each be a str or a list. `self.syn` is a single
        synapse when both are strings, otherwise a list (types x locations, in
        that order).
        """
        if add_syn is None:
            return
        if syn_loc is None:
            raise ValueError('syn_loc must be specified if add_syn is given.')
        names = [add_syn] if isinstance(add_syn, str) else list(add_syn)
        locs = [syn_loc] if isinstance(syn_loc, str) else list(syn_loc)
        print(f'Adding synapse(s) {names} at {locs}')
        syns = [self._make_synapse(n, loc) for n in names for loc in locs]
        single = isinstance(add_syn, str) and isinstance(syn_loc, str)
        self.syn = syns[0] if single else syns

    def build(self, sec_name_apical=None, add_syn=None, syn_loc=None):

        self._create_sections()
        if sec_name_apical in self._nrn_sections:
            self._insert_dipole(sec_name_apical)
        if add_syn is not None:
            self._create_synapses(add_syn, syn_loc=syn_loc)

    def _insert_dipole(self, sec_name_apical):
        """Insert dipole into each section of this cell.

        Parameters
        ----------
        sec_name_apical : str
            The name of the section along which dipole moment is calculated.
        """
        self.dpl_vec = h.Vector(1)
        self.dpl_ref = self.dpl_vec._ref_x[0]
        cos_thetas = _get_cos_theta(self.sections, 'apical_trunk')

        # setting pointers and ztan values
        for sect_name in self.sections:
            sect = self._nrn_sections[sect_name]
            sect.insert('dipole')

            dpp = h.Dipole(1, sec=sect)  # defined in dipole_pp.mod
            self.dipole_pp.append(dpp)
            dpp.ri = h.ri(1, sec=sect)  # assign internal resistance
            # sets pointers in dipole mod file to the correct locations
            dpp._ref_pv = sect(0.99)._ref_v
            dpp._ref_Qtotal = self.dpl_ref
            # gives INTERNAL segments of the section, non-endpoints
            # creating this because need multiple values simultaneously
            pos_all = np.array([seg.x for seg in sect.allseg()])
            seg_lens = np.diff(pos_all) * sect.L
            seg_lens_z = seg_lens * cos_thetas[sect_name]

            # alternative procedure below with y_long(itudinal)
            # y_long = (h.y3d(1, sec=sect) - h.y3d(0, sec=sect)) * pos
            # y_diff = np.diff(y_long)

            # doing range to index multiple values of the same
            # np.array simultaneously
            for idx, pos in enumerate(pos_all[1:-1]):
                # assign the ri value to the dipole
                # ri not defined at 0 and L
                sect(pos).dipole.ri = h.ri(pos, sec=sect)
                # range variable 'dipole'
                # set pointers to previous segment's voltage, with
                # boundary condition
                sect(pos).dipole._ref_pv = sect(pos_all[idx])._ref_v

                # set aggregate pointers
                sect(pos).dipole._ref_Qsum = dpp._ref_Qsum
                sect(pos).dipole._ref_Qtotal = self.dpl_ref
                # add ztan values
                sect(pos).dipole.ztan = seg_lens_z[idx]
            # set the pp dipole's ztan value to the last value from seg_lens_z
            dpp.ztan = seg_lens_z[-1]
        self.dipole = h.Vector().record(self.dpl_ref)


##### Cell parameters (as used in HNN)
# L5ET cell
def l5_params():

    section_params = {
        # Soma
        'L5Pyr_soma_L': 39.,
        'L5Pyr_soma_diam': 28.9,
        'L5Pyr_soma_cm': 1,
        'L5Pyr_soma_Ra': 100, #Rich 495.73, Hay: 100

        # Dendrite
        'L5Pyr_dend_cm': 1,
        'L5Pyr_dend_Ra': 100, #Rich 495.73, Hay: 100

        'L5Pyr_apicaltrunk_L': 102.,
        'L5Pyr_apicaltrunk_diam': 10.2,

        'L5Pyr_apical1_L': 680.,
        'L5Pyr_apical1_diam': 7.48,

        'L5Pyr_apical2_L': 680.,
        'L5Pyr_apical2_diam': 4.93,

        'L5Pyr_apicaltuft_L': 425.,
        'L5Pyr_apicaltuft_diam': 3.4,

        'L5Pyr_apicaloblique_L': 255.,
        'L5Pyr_apicaloblique_diam': 5.1,

        'L5Pyr_basal1_L': 85.,
        'L5Pyr_basal1_diam': 6.8,

        'L5Pyr_basal2_L': 255.,
        'L5Pyr_basal2_diam': 8.5,

        'L5Pyr_basal3_L': 255.,
        'L5Pyr_basal3_diam': 8.5,
    }

    end_pts = {
            'soma': [[0, 0, 0], [0, 0, 39]],
            'apical_trunk': [[0, 0, 39], [0, 0, 141]],
            'apical_oblique': [[0, 0, 141], [-255, 0, 141]],
            'apical_1': [[0, 0, 141], [0, 0, 821]],
            'apical_2': [[0, 0, 821], [0, 0, 1501]],
            'apical_tuft': [[0, 0, 1501], [0, 0, 1926]],
            'basal_1': [[0, 0, 0], [0, 0, -85]],
            'basal_2': [[2.842170943040401e-14, 0, -84.99999999999997], [-180.3122292025696, 0, -265.3122292025696]],
            'basal_3': [[-2.842170943040401e-14, 0, -84.99999999999997], [180.3122292025696, 0, -265.312229202569]]
        }

    cell_tree = {
            ('apical_trunk', 0): [('apical_trunk', 1)],
            ('apical_1', 0): [('apical_1', 1)],
            ('apical_2', 0): [('apical_2', 1)],
            ('apical_tuft', 0): [('apical_tuft', 1)],
            ('apical_oblique', 0): [('apical_oblique', 1)],
            ('basal_1', 0): [('basal_1', 1)],
            ('basal_2', 0): [('basal_2', 1)],
            ('basal_3', 0): [('basal_3', 1)],
            # Different sections connected
            ('soma', 0): [('soma', 1), ('basal_1', 0)],
            ('soma', 1): [('apical_trunk', 0)],
            ('apical_trunk', 1): [('apical_1', 0), ('apical_oblique', 0)],
            ('apical_1', 1): [('apical_2', 0)],
            ('apical_2', 1): [('apical_tuft', 0)],
            ('basal_1', 1): [('basal_2', 0), ('basal_3', 0)]
        }

    return section_params, end_pts, cell_tree

# IT cell not included in manuscript or final model.
def l5IT_params():

    section_params = {
        # Soma
        'L5Pyr_soma_L': 39.*0.75,
        'L5Pyr_soma_diam': 28.9*0.75,
        'L5Pyr_soma_cm': 1,
        'L5Pyr_soma_Ra': 200, 

        # Dendrite
        'L5Pyr_dend_cm': 1,
        'L5Pyr_dend_Ra': 200, 

        'L5Pyr_apicaltrunk_L': 102.,
        'L5Pyr_apicaltrunk_diam': 10.2,

        'L5Pyr_apical1_L': 680.,
        'L5Pyr_apical1_diam': 7.48,

        'L5Pyr_apicaltuft_L': 425.,
        'L5Pyr_apicaltuft_diam': 3.4*0.5,

        'L5Pyr_apicaloblique_L': 255.,
        'L5Pyr_apicaloblique_diam': 5.1,

        'L5Pyr_basal1_L': 85.,
        'L5Pyr_basal1_diam': 6.8,

        'L5Pyr_basal2_L': 255.,
        'L5Pyr_basal2_diam': 8.5,

        'L5Pyr_basal3_L': 255.,
        'L5Pyr_basal3_diam': 8.5,
    }

    end_pts = {
            'soma': [[0, 0, 0], [0, 0, 23]],
            'apical_trunk': [[0, 0, 23], [0, 0, 83]],
            'apical_oblique': [[0, 0, 83], [-150, 0, 83]],
            'apical_1': [[0, 0, 83], [0, 0, 483]],
            'apical_tuft': [[0, 0, 883], [0, 0, 1133]],
            'basal_1': [[0, 0, 0], [0, 0, -50]],
            'basal_2': [[0, 0, -50], [-106, 0, -156]],
            'basal_3': [[0, 0, -50], [106, 0, -156]]
        }

    cell_tree = {
            ('apical_trunk', 0): [('apical_trunk', 1)],
            ('apical_1', 0): [('apical_1', 1)],
            ('apical_tuft', 0): [('apical_tuft', 1)],
            ('apical_oblique', 0): [('apical_oblique', 1)],
            ('basal_1', 0): [('basal_1', 1)],
            ('basal_2', 0): [('basal_2', 1)],
            ('basal_3', 0): [('basal_3', 1)],
            # Different sections connected
            ('soma', 0): [('soma', 1), ('basal_1', 0)],
            ('soma', 1): [('apical_trunk', 0)],
            ('apical_trunk', 1): [('apical_1', 0), ('apical_oblique', 0)],
            ('apical_1', 1): [('apical_tuft', 0)],
            ('basal_1', 1): [('basal_2', 0), ('basal_3', 0)]
        }

    return section_params, end_pts, cell_tree

# L2/3 pyramidal cell
def l2_params():

    section_params = {

        # Soma
        'L2Pyr_soma_L': 22.1,
        'L2Pyr_soma_diam': 23.4,
        'L2Pyr_soma_cm': 1.5,
        'L2Pyr_soma_Ra': 200,

        # Dendrites
        'L2Pyr_dend_cm': 1.5,
        'L2Pyr_dend_Ra': 200,

        'L2Pyr_apicaltrunk_L': 59.5,
        'L2Pyr_apicaltrunk_diam': 4.25,

        'L2Pyr_apical1_L': 306.,
        'L2Pyr_apical1_diam': 4.08,

        'L2Pyr_apicaltuft_L': 238.,
        'L2Pyr_apicaltuft_diam': 3.4,

        'L2Pyr_apicaloblique_L': 340.,
        'L2Pyr_apicaloblique_diam': 3.91,

        'L2Pyr_basal1_L': 85.,
        'L2Pyr_basal1_diam': 4.25,

        'L2Pyr_basal2_L': 255.,
        'L2Pyr_basal2_diam': 2.72,

        'L2Pyr_basal3_L': 255.,
        'L2Pyr_basal3_diam': 2.72,
    }

    end_pts = {
            'soma': [[0, 0, 0], [0, 0, 22.1]],
            'apical_trunk': [[0,0,22.099999999999987], [0,0,81.59999999999998]],
            'apical_oblique': [[0,0,81.59999999999998], [-340, 0, 81.59999999999998]],
            'apical_1': [[0, 0, 81.59999999999997], [0,0, 387.59999999999997]],
            'apical_tuft': [[0, 0, 387.5999999999999], [0, 0, 625.5999999999999]],
            'basal_1': [[0,0,0], [0,0,-85.0]],
            'basal_2': [[2.842170943040401e-14, 0, -84.99999999999997], [-180.3122292025696, 0, -265.3122292025696]],
            'basal_3': [[-2.842170943040401e-14, 0, -84.99999999999997], [180.3122292025696, 0, -265.312229202569]],
        }

    cell_tree = {
        ('apical_trunk', 0): [('apical_trunk', 1)],
        ('apical_1', 0): [('apical_1', 1)],
        ('apical_tuft', 0): [('apical_tuft', 1)],
        ('apical_oblique', 0): [('apical_oblique', 1)],
        ('basal_1', 0): [('basal_1', 1)],
        ('basal_2', 0): [('basal_2', 1)],
        ('basal_3', 0): [('basal_3', 1)],
        # Different sections connected
        ('soma', 0): [('soma', 1), ('basal_1', 0)],
        ('soma', 1): [('apical_trunk', 0)],
        ('apical_trunk', 1): [('apical_1', 0), ('apical_oblique', 0)],
        ('apical_1', 1): [('apical_tuft', 0)],
        ('basal_1', 1): [('basal_2', 0), ('basal_3', 0)]}
    
    return section_params, end_pts, cell_tree