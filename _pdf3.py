import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import matplotlib.cm as cm

# =========================
# Global font settings
# =========================
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 18
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'


# =========================
# RDF LOADING
# =========================
def load_rdf(path: Path):
    data = np.loadtxt(path, comments="#")
    r = data[:, 0]
    g_avg = data[:, 1]
    g_std = data[:, 2] if data.shape[1] > 2 else None
    return r, g_avg, g_std


files = [
    "rdf_data/rdf_results_L8_dr0.100.txt",
    "rdf_data/rdf_results_L12_dr0.100.txt",
    "rdf_data/rdf_results_L14_dr0.100.txt",
    "rdf_data/rdf_results_L18_dr0.100.txt",
    "rdf_data/rdf_results_L24_dr0.100.txt",
]

N = np.array([8., 12., 14., 18., 24.])


# =========================
# SEQUENCE MODEL PART
# =========================
def eps_ij(c: float) -> np.ndarray:
    E_HH = (1 - c) / 2
    E_HT = (1 - c) / 2
    E_TT = (1 + c) / 2
    return np.array([[E_HH, E_HT],
                     [E_HT, E_TT]], float)

seq_list = np.eye(8, dtype=int)
c = 0.75

def chi_av():
    chi_list = []
    for seq in seq_list:
        eps = eps_ij(c)
        S = eps[seq[:, None], seq[None, :]]
        chi = np.sum(S) / 64
        chi_list.append(chi)
    return np.array(chi_list)

def sigma2():
    i = np.arange(1, 9, dtype=float)
    j = 8 - i
    sigma2 = ((i - 1) * i * (2 * i - 1) + j * (j + 1) * (2 * j + 1)) / (18 * 8**2)
    return sigma2 / 8

def chi_de():
    chi_list = []
    sig2 = sigma2()
    for seq in seq_list:
        eps = eps_ij(c)
        S = eps[seq[:, None], seq[None, :]] / (sig2[:, None] + sig2[None, :])**(3./2.)
        chi = np.sum(S) / 64
        chi_list.append(chi)
    return np.array(chi_list)

Tc = np.array([1.6713, 1.6418, 1.6291, 1.6236,
               1.6236, 1.6291, 1.6418, 1.6713])


# =========================
# CREATE STACKED FIGURE
# =========================
fig = plt.figure(figsize=(6.4, 9.6))

pos1 = [0.12, 0.65, 0.75, 0.3]
pos2 = [0.12, 0.25, 0.75, 0.3]

ax1 = fig.add_axes(pos1)
ax2 = fig.add_axes(pos2)

# =========================
# PANEL 1 — RDF
# =========================
cmap = cm.get_cmap("viridis")
colors = cmap(np.linspace(0, 1, len(files)))

for i, fname in enumerate(files):
    path = Path(fname)
    r, g_avg, g_std = load_rdf(path)
    L = fname.split("_L")[1].split("_")[0]

    # Dash pattern becomes denser as i increases
    dash_length = 6 - i        # decreasing dash size
    gap_length = 3 - 0.75*i     # decreasing gap
    dash_pattern = (0, (dash_length, gap_length))

    ax1.plot(
        r[1:] / np.sqrt(N[i]),
        -(g_avg[1:] - 1) * np.sqrt(N[i]),
        linewidth=1.5,
        color=colors[i],
        linestyle=dash_pattern,
        label=rf"$N = {L}$"
    )

h = 1.9
l = 0.4
x = np.arange(0, 2.5, 0.01)

ax1.axhline(0, color="black", linewidth=1, linestyle="dotted")

ax1.set_xlabel(r"$N^{-1/2}r$")
ax1.set_ylabel(r"$-N^{1/2}h_\mathrm{cm}(N^{-1/2}r)$")
ax1.set_xlim([0,2.5])
ax1.legend(frameon=False)

# Panel label
ax1.text(-0.14, 1.04, "(a)", transform=ax1.transAxes, fontsize=24, fontname="Times New Roman")


# =========================
# PANEL 2 — P and Tc
# =========================
P = chi_de() / chi_av()

ax2.plot(P, 'o-', color="#8f00ff", label=r'$P$')

ax2.set_ylabel(r'$P$', rotation=0, labelpad=8)
ax2.yaxis.label.set_va('center')
ax2.spines['left'].set_color('#8f00ff')
ax2.tick_params(axis='y', colors='#8f00ff')
ax2.yaxis.label.set_color('#8f00ff')

ax3 = ax2.twinx()
ax3.plot(Tc, 's--', color='#ff8c42', label=r'$T_\mathrm{c}$')

ax3.set_ylabel(r'$T_\mathrm{c}$', rotation=0, labelpad=8)
ax3.yaxis.label.set_va('center')
ax3.spines['right'].set_color('#ff8c42')
ax3.tick_params(axis='y', colors='#ff8c42')
ax3.yaxis.label.set_color('#ff8c42')

sequences = ['ABBBBBBB', 'BABBBBBB', 'BBABBBBB', 'BBBABBBB',
             'BBBBABBB', 'BBBBBABB', 'BBBBBBAB', 'BBBBBBBA']

ax2.set_xticks(range(8))
ax2.set_xticklabels([])

circle_char = '●'

for i, seq in enumerate(sequences):
    colors = ['#0000ff' if letter == 'A' else '#cfb997' for letter in seq]
    for j, color in enumerate(colors):
        ax2.text(i, -0.05 - j*0.08, circle_char,
                 ha='center', va='top',
                 color=color, fontsize=39,
                 transform=ax2.get_xaxis_transform())

# Panel label
ax2.text(-0.14, 1.04, "(b)", transform=ax2.transAxes, fontsize=24, fontname="Times New Roman")

# Combined legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax3.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2,
           loc="center", frameon=False)

plt.tight_layout()
plt.subplots_adjust(hspace=0.5)

plt.savefig("_pdf3.pdf")
plt.show()
