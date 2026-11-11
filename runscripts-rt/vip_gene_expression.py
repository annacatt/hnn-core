"""Ion-channel gene expression of human L2/3 VIP interneurons (Lee, Dalley et al. 2023).

Data: Patch-seq cells from Lee, Dalley et al. (2023), "Signature morphoelectric
properties of diverse GABAergic interneurons in the human neocortex", Science,
https://github.com/AllenInstitute/human_patchseq_gaba

What it does
    1. downloads the data files once into runscripts-rt/data/lee_dalley/
       (about 30 MB),
    2. selects VIP cells in layers 2-3 (and, for comparison, PVALB, SST and
       LAMP5/PAX6 cells in the same layers),
    3. summarises the voltage-gated ion-channel genes (vgic_genes.csv, which
       includes HCN1-4) per subclass: mean and median expression and the
       fraction of cells in which each gene is detected,
    4. plots channel genes potentially associated with the currents in
       VIPwithT.cpp and a few candidates (see GENE_GROUPS), compares acute vs
       cultured VIP cells, and relates HCN1 expression to the measured sag,
    5. writes tables (CSV) to data/lee_dalley/results/ and figures to
       figures/vip_gene_expression/.

Expression values are log2(CPM + 1), as provided by the authors.
Patch-seq misses many transcripts in single cells, so the fraction of cells in
which a gene is detected is often more informative than one cell's value.

The presence or expression level of a channel-encoding gene does not
establish the magnitude or biophysical properties of the corresponding
current, which also depend on channel assembly, auxiliary subunits, membrane
localization, and cellular context. Use these results to select candidate
mechanisms for the model, not as evidence that a current is present or as a
basis for setting conductances: that comes from fitting the recordings.

The expression matrix is read from data/lee_dalley/expression_log2cpm.npz, a
NumPy copy of the authors' two .RData files, so only numpy, pandas and
matplotlib are needed. If that file is missing and pyreadr happens to be
installed, the script rebuilds it from the .RData files. Run it from
anywhere, or cell by cell (# %%) in VS Code / PyCharm / Jupyter.
"""

# %% imports, paths, settings
import sys
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    RT_DIR = Path(__file__).resolve().parent
except NameError:  # running interactively
    RT_DIR = Path.cwd()

DATA_DIR = RT_DIR / 'data' / 'lee_dalley'      # downloaded files (keep out of git)
OUT_DIR = DATA_DIR / 'results'                 # tables written by this script
FIG_DIR = RT_DIR / 'figures' / 'vip_gene_expression'
for d in (DATA_DIR, OUT_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

LAYERS = ['2', '3', '2_3']        # values of Cortical_layer to keep
SUBCLASSES = ['VIP', 'PVALB', 'SST', 'LAMP5/PAX6/Other']

# Channel genes to plot, grouped by current family. The tables written below
# cover every gene in vgic_genes.csv, not only these.
#
# Gene -> channel subunit follows vgic_genes.csv (IUPHAR names). Subunit ->
# current is standard channel physiology, not part of the Lee-Dalley data:
#   Nav1.1/1.2/1.3/1.6   fast Na+ current
#   Kv3.1/3.2            fast delayed-rectifier K+ current, typical of
#                        fast-spiking interneurons
#   Kv1.1/1.2/1.6        Kv1-mediated low-threshold K+ currents, which can
#                        contribute to the slowly inactivating,
#                        dendrotoxin-sensitive D-current
#   Kv7.2/7.3/7.5        M-current
#   Cav3.1-3.3           T-type (low-threshold) Ca2+ current
#   Kv4.2/4.3            A-type K+ current
#   HCN1-4               h-current
#   Cav2.1/2.2/2.3       P/Q-, N- and R-type high-voltage-activated Ca2+
#                        currents
#   KCa1.1; KCa2.2/2.3   BK current; SK current
# These mappings identify the ionic current families potentially associated
# with each channel subunit. However, the presence or expression level of a
# channel-encoding gene does not establish the magnitude or biophysical
# properties of the corresponding current, which also depend on channel
# assembly, auxiliary subunits, membrane localization, and cellular context.
#
# The first five groups relate to the currents in VIPwithT.cpp (Na, K, D, M,
# T); the rest are candidates not in the model.
GENE_GROUPS = {
    'Na (fast)':        ['SCN1A', 'SCN2A', 'SCN3A', 'SCN8A'],
    'K delayed rect. (Kv3)': ['KCNC1', 'KCNC2'],
    'Kv1 (low-threshold K)': ['KCNA1', 'KCNA2', 'KCNA6'],
    'M-current (Kv7)':  ['KCNQ2', 'KCNQ3', 'KCNQ5'],
    'T-type Ca (Cav3)': ['CACNA1G', 'CACNA1H', 'CACNA1I'],
    'A-type K (Kv4)':   ['KCND2', 'KCND3'],
    'h-current (HCN)':  ['HCN1', 'HCN2', 'HCN3', 'HCN4'],
    'HVA Ca (Cav2)':    ['CACNA1A', 'CACNA1B', 'CACNA1E'],
    'BK / SK':          ['KCNMA1', 'KCNN2', 'KCNN3'],
}

REPO_RAW = ('https://raw.githubusercontent.com/AllenInstitute/'
            'human_patchseq_gaba/main/data/')
RDATA = ['complete_patchseq_data_sets1.RData',
         'complete_patchseq_data_sets2.RData']
FILES = ['LeeDalley_manuscript_metadata_v2.csv',
         'LeeDalley_ephys_fx.csv',
         'vgic_genes.csv']
EXPR_NPZ = DATA_DIR / 'expression_log2cpm.npz'   # genes x cells, log2(CPM+1)

# colours: categorical slots 1-2 (blue, orange) and a one-hue blue ramp
BLUE, ORANGE, GREY = '#2a78d6', '#eb6834', '#8a8a86'
BLUE_RAMP = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf',
             '#184f95', '#0d366b']


# %% 1. download (only files that are missing)
def download(name):
    path = DATA_DIR / name
    if path.exists():
        return path
    print(f'downloading {name} ...')
    tmp = path.with_suffix(path.suffix + '.part')
    try:
        urllib.request.urlretrieve(REPO_RAW + name, tmp)
    except Exception as err:  # noqa: BLE001
        sys.exit(f'Could not download {name}: {err}\n'
                 'If this is an SSL certificate error on macOS, run the '
                 '"Install Certificates.command" that comes with Python, or\n'
                 f'download the file by hand from {REPO_RAW + name}\n'
                 f'and save it in {DATA_DIR}')
    tmp.rename(path)
    return path


for f in FILES:
    download(f)


# %% 2. load expression (genes x cells) and cell metadata
if not EXPR_NPZ.exists():                    # rebuild from the .RData files
    try:
        import pyreadr
    except ImportError:
        sys.exit(f'{EXPR_NPZ.name} is missing from {DATA_DIR}.\n'
                 'It is made from the two .RData files with pyreadr, which '
                 'is not installed here. Ask for the converted file, or '
                 'convert it in a separate environment.')
    for f in RDATA:
        download(f)
    part1 = pyreadr.read_r(DATA_DIR / RDATA[0])['datPatch1']
    part2 = pyreadr.read_r(DATA_DIR / RDATA[1])['datPatch2']
    both = pd.concat([part1, part2], axis=1)
    np.savez_compressed(EXPR_NPZ, values=both.to_numpy(np.float32),
                        genes=np.array(both.index, dtype=str),
                        cells=np.array(both.columns, dtype=str))
    del part1, part2, both

print('reading expression data ...')
with np.load(EXPR_NPZ) as z:
    expr = pd.DataFrame(z['values'], index=z['genes'], columns=z['cells'])

meta = pd.read_csv(DATA_DIR / 'LeeDalley_manuscript_metadata_v2.csv')
meta.columns = meta.columns.str.strip()      # one column name ends in a space
meta['Cortical_layer'] = meta['Cortical_layer'].astype(str)
meta = meta[meta['patched_cell_container'].isin(expr.columns)]

ephys = pd.read_csv(DATA_DIR / 'LeeDalley_ephys_fx.csv')
ephys['specimen_id'] = ephys['specimen_id'].astype('int64')

vgic = pd.read_csv(DATA_DIR / 'vgic_genes.csv')
vgic['family'] = vgic['Family name'].str.replace('<[^>]+>', '', regex=True)
vgic = vgic[vgic['HGNC symbol'].isin(expr.index)]

print(f'{expr.shape[0]} genes x {expr.shape[1]} cells; '
      f'{len(vgic)} channel genes found')


# %% 3. select L2/3 cells
sel = meta[meta['Cortical_layer'].isin(LAYERS)
           & meta['Revised_subclass_label'].isin(SUBCLASSES)]
vip = sel[sel['Revised_subclass_label'] == 'VIP']

print('\nL2/3 cells per subclass:')
print(sel['Revised_subclass_label'].value_counts().to_string())
print(f'\nL2/3 VIP: {len(vip)} cells '
      f"({(vip['condition'] == 'acute').sum()} acute, "
      f"{(vip['condition'] == 'culture').sum()} cultured), "
      f"{vip['specimen_id_x'].isin(ephys['specimen_id']).sum()} with ephys "
      'features')
print('\nL2/3 VIP transcriptomic types:')
print(vip['Transcriptomic_type'].value_counts().to_string())

# cells x channel genes for L2/3 VIP, with metadata and key ephys features
EPHYS_COLS = ['v_baseline', 'input_resistance', 'tau', 'rheobase_i', 'sag',
              'latency_rheo', 'fi_fit_slope', 'adapt_hero', 'avg_rate_hero',
              'threshold_v_rheo', 'width_rheo']
META_COLS = ['specimen_id_x', 'cell_name', 'patched_cell_container', 'nwb_id',
             'Donor', 'structure', 'Cortical_layer', 'Transcriptomic_type',
             'condition', 'Genes_Detected']
vip_table = (vip[META_COLS]
             .merge(ephys[['specimen_id'] + EPHYS_COLS], how='left',
                    left_on='specimen_id_x', right_on='specimen_id')
             .drop(columns='specimen_id'))
vip_genes = expr.loc[vgic['HGNC symbol'], vip['patched_cell_container']].T
vip_table = vip_table.join(vip_genes, on='patched_cell_container')
vip_table.to_csv(OUT_DIR / 'vip_L23_cells_channel_genes_ephys.csv', index=False)


# %% 4. summary per subclass: mean, median, fraction of cells detected
rows = []
for sub in SUBCLASSES:
    cells = sel.loc[sel['Revised_subclass_label'] == sub, 'patched_cell_container']
    x = expr.loc[vgic['HGNC symbol'], cells]
    rows.append(pd.DataFrame({
        'gene': x.index, 'subclass': sub, 'n_cells': len(cells),
        'mean_log2cpm': x.mean(axis=1).values,
        'median_log2cpm': x.median(axis=1).values,
        'frac_detected': (x > 0).mean(axis=1).values}))
summary = pd.concat(rows).merge(
    vgic[['HGNC symbol', 'family']], left_on='gene', right_on='HGNC symbol'
).drop(columns='HGNC symbol')
summary.to_csv(OUT_DIR / 'channel_genes_by_subclass_L23.csv', index=False)

# wide version: VIP next to the other subclasses, sorted by VIP detection
wide = summary.pivot(index=['family', 'gene'], columns='subclass',
                     values=['frac_detected', 'mean_log2cpm'])
wide = wide.sort_values(('frac_detected', 'VIP'), ascending=False).round(3)
wide.to_csv(OUT_DIR / 'channel_genes_by_subclass_L23_wide.csv')

genes_plot = [g for group in GENE_GROUPS.values() for g in group
              if g in expr.index]
print('\nL2/3 VIP, selected channel genes '
      '(fraction of cells detected, mean log2 CPM+1):')
s_vip = summary[summary['subclass'] == 'VIP'].set_index('gene')
print(s_vip.loc[genes_plot, ['frac_detected', 'mean_log2cpm']]
      .round(2).to_string())


# %% 5. dot plot: selected channel genes across L2/3 subclasses
# dot area = fraction of cells with the gene detected; colour = mean log2 CPM+1
s = summary.set_index(['gene', 'subclass'])
vmax = np.ceil(s.loc[genes_plot, 'mean_log2cpm'].max())
cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list('blue', BLUE_RAMP)

fig, (ax, lax) = plt.subplots(
    1, 2, figsize=(7.2, 0.29 * len(genes_plot) + 1.6),
    gridspec_kw={'width_ratios': [4.2, 1]}, constrained_layout=True)
lax.axis('off')                                       # column for the legends
y, ylabels, group_mid = 0, [], []
for group, genes in GENE_GROUPS.items():
    genes = [g for g in genes if g in expr.index]
    start = y
    for g in genes:
        for j, sub in enumerate(SUBCLASSES):
            frac, mean = s.loc[(g, sub), ['frac_detected', 'mean_log2cpm']]
            ax.scatter(j, y, s=12 + 230 * frac, c=[mean], cmap=cmap, vmin=0,
                       vmax=vmax, edgecolors='white', linewidths=0.8,
                       zorder=3)
        ylabels.append(g)
        y += 1
    group_mid.append(((start + y - 1) / 2, group))
    ax.axhline(y - 0.5, color='#e2e1dc', lw=0.8, zorder=0)
ax.set_yticks(range(len(ylabels)), ylabels, fontsize=8)
ax.set_ylim(len(ylabels) - 0.5, -0.5)
n_by_sub = sel['Revised_subclass_label'].value_counts()
SHORT = {'LAMP5/PAX6/Other': 'LAMP5/\nPAX6'}
ax.set_xticks(range(len(SUBCLASSES)),
              [f"{SHORT.get(sub, sub)}\n(n={n_by_sub[sub]})"
               for sub in SUBCLASSES], fontsize=8)
ax.set_xlim(-0.6, len(SUBCLASSES) + 1.5)              # room for group names
ax.tick_params(length=0)
for side in ('top', 'right', 'left', 'bottom'):
    ax.spines[side].set_visible(False)
ax.axvspan(-0.5, 0.5, color='#f0efec', zorder=-1)    # highlight VIP column
for mid, group in group_mid:                         # group names on the right
    ax.text(len(SUBCLASSES) - 0.45, mid, group, fontsize=7.5, color='#55544f',
            va='center', ha='left')
ax.set_title('Channel genes in human L2/3 interneurons (Lee, Dalley 2023)',
             fontsize=9, loc='left')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
cax = lax.inset_axes([0.05, 0.72, 0.12, 0.26])
cb = fig.colorbar(sm, cax=cax)
cb.set_label('mean log2(CPM+1)', fontsize=7.5)
cb.ax.tick_params(labelsize=7)
cb.outline.set_visible(False)
for frac in (0.25, 0.5, 1.0):                        # size legend
    lax.scatter([], [], s=12 + 230 * frac, color=GREY, edgecolors='white',
                label=f'{frac:.0%}')
lax.legend(title='cells detected', fontsize=7, title_fontsize=7.5,
           frameon=False, loc='upper left', bbox_to_anchor=(0, 0.62),
           labelspacing=1.1, alignment='left')
fig.savefig(FIG_DIR / 'channel_genes_dotplot_L23.png', dpi=200)
plt.show()


# %% 6. L2/3 VIP: acute vs cultured cells (fraction detected)
acute = vip.loc[vip['condition'] == 'acute', 'patched_cell_container']
cult = vip.loc[vip['condition'] == 'culture', 'patched_cell_container']
fa = (expr.loc[genes_plot, acute] > 0).mean(axis=1)
fc = (expr.loc[genes_plot, cult] > 0).mean(axis=1)

fig, ax = plt.subplots(figsize=(8, 3.2), constrained_layout=True)
xpos = np.arange(len(genes_plot))
w = 0.38
ax.bar(xpos - w / 2 - 0.01, fa, w, color=BLUE, label=f'acute (n={len(acute)})')
ax.bar(xpos + w / 2 + 0.01, fc, w, color=ORANGE,
       label=f'culture (n={len(cult)})')
ax.set_xticks(xpos, genes_plot, rotation=90, fontsize=7.5)
ax.set(ylabel='fraction of cells detected', ylim=(0, 1.05))
ax.set_title('L2/3 VIP cells: acute vs cultured slices', fontsize=9,
             loc='left')
ax.grid(axis='y', color='#e2e1dc', lw=0.6)
ax.set_axisbelow(True)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
ax.legend(fontsize=7.5, frameon=False, ncol=2, loc='lower right',
          bbox_to_anchor=(1, 1))
fig.savefig(FIG_DIR / 'vip_L23_acute_vs_culture.png', dpi=200)
plt.show()


# %% 7. link to electrophysiology: HCN1 expression vs sag (L2/3 VIP)
d = vip_table[['HCN1', 'sag', 'condition']].dropna()
rho = d['HCN1'].rank().corr(d['sag'].rank())     # Spearman, without scipy
print(f'\nHCN1 vs sag, L2/3 VIP: Spearman rho = {rho:.2f} (n = {len(d)})')

fig, ax = plt.subplots(figsize=(4.2, 3.4), constrained_layout=True)
for cond, color in (('acute', BLUE), ('culture', ORANGE)):
    dd = d[d['condition'] == cond]
    ax.scatter(dd['HCN1'], dd['sag'], s=40, color=color, edgecolors='white',
               linewidths=1, label=f'{cond} (n={len(dd)})', zorder=3)
ax.set(xlabel='HCN1, log2(CPM+1)', ylabel='sag ratio',
       title=f'HCN1 vs sag, L2/3 VIP (Spearman ρ = {rho:.2f})')
ax.title.set_fontsize(9)
ax.grid(color='#e2e1dc', lw=0.6)
ax.set_axisbelow(True)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
ax.legend(fontsize=7.5, frameon=False)
fig.savefig(FIG_DIR / 'vip_L23_HCN1_vs_sag.png', dpi=200)
plt.show()

print(f'\ntables saved in {OUT_DIR}\nfigures saved in {FIG_DIR}')
