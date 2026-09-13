import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit


# =====================================================================
# GLOBAL PLOT STYLE — Times New Roman
# =====================================================================

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 18
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'


def format_value_error(value, error):
    """Format value ± 1-sigma error in fixed-point notation with ~2 sig figs in the error."""
    if not np.isfinite(error) or error <= 0:
        return f"{value:.6f}", f"{error:.6f}"

    order = int(np.floor(np.log10(abs(error))))
    decimals = max(0, 1 - order)

    value_str = f"{value:.{decimals}f}"
    error_str = f"{error:.{decimals}f}"
    return value_str, error_str


# =====================================================================
# HPS-URRY PARAMETERS
# =====================================================================

# Rounded Urry hydropathy values used in the original interaction table
hydropathy = {
    'A': 0.60,
    'R': 0.56,
    'N': 0.59,
    'D': 0.29,
    'C': 0.65,
    'Q': 0.56,
    'E': 0.00,
    'G': 0.57,
    'H': 0.76,
    'I': 0.71,
    'L': 0.72,
    'K': 0.38,
    'M': 0.68,
    'F': 0.82,
    'P': 0.76,
    'S': 0.59,
    'T': 0.59,
    'W': 1.00,
    'Y': 0.90,
    'V': 0.66
}


# HPS-Urry bead diameters in Angstrom
sigma_AA = {
    'A': 5.04,
    'R': 6.56,
    'N': 5.68,
    'D': 5.58,
    'C': 5.48,
    'Q': 6.02,
    'E': 5.92,
    'G': 4.50,
    'H': 6.08,
    'I': 6.18,
    'L': 6.18,
    'K': 6.36,
    'M': 6.18,
    'F': 6.36,
    'P': 5.56,
    'S': 5.18,
    'T': 5.62,
    'W': 6.78,
    'Y': 6.46,
    'V': 5.86
}


# Charges in units of elementary charge e
charge = {
    'A': 0.0,
    'R': +1.0,
    'N': 0.0,
    'D': -1.0,
    'C': 0.0,
    'Q': 0.0,
    'E': -1.0,
    'G': 0.0,
    'H': 0.0,
    'I': 0.0,
    'L': 0.0,
    'K': +1.0,
    'M': 0.0,
    'F': 0.0,
    'P': 0.0,
    'S': 0.0,
    'T': 0.0,
    'W': 0.0,
    'Y': 0.0,
    'V': 0.0
}


# =====================================================================
# PHYSICAL PARAMETERS
# =====================================================================

epsilon_LJ = 0.2            # kcal/mol

Delta = 0.08

# Debye-Huckel
D_dielectric = 80.0
kappa = 0.1                 # Angstrom^-1
k_e = 332.06371             # kcal Angstrom / mol


# =====================================================================
# NAMES
# =====================================================================

names = [
    "WT",
    "V1",
    "V2",
    "V3",
    "protein1",
    "protein2",
    "protein3",
    "protein4",
    "protein5",
    "protein6",
    "protein7",
    "protein8",
    "protein9",
    "protein10",
    "protein11",
    "protein12",
    "protein13",
    "protein14",
    "protein15"
]


# =====================================================================
# SEQUENCES
# =====================================================================

sequences = [

    # LAF-1 RGG WT
    "MESNQSNNGGSGNAALNRGGRYVPPHLRGGDGGAAAAASAGGDDRRGGAGGGGYRRGGGNSGGGGGGGYDRGYNDNRDDRDNRGGSGGYGRDRNYEDRGYNGGGGGGGNRGYNNNRGGGGGGYNRQDRGDGGSSNFSRGGYNNRDEGSDNRGSGRSYNNDRRDNGGDG",

    # LAF-1 RGG V1
    "QRNRGYNRGNNGNGGVYGGRSRGRGGALRRGGYYPGSYNSNNRRRGSSGRNRGNRGSGGGRNSAGSDQAGGNGGAANYRRDMSHNDYGDGYNARPGESGLGRRGNDDAANNDGYGGGGDGDNGGERGDGGDAGSGGGGYFGGSNEYDGDDGRDGGNNDDGGGGRNGRG",

    # LAF-1 RGG V2
    "RGYYSGNRGRRNRGFSGRGNRLNRYRRNGSRGRPGGLRRGRPSYGRRGGEGGYVRANGGRNNSRGNGSGGGGQNGYAAGGGGGRDYDRGQGGADGYGSSGMGDHAGAGDGNNGGYNNNDAGGNNNGNDGRYEGNGANADDEDDGGSRSGGGDDSYNDSDDGGNDGGGG",

    # LAF-1 RGG V3
    "AGADDDNDGELDGDGGDDGNGDNGNDYGDNGEDGDDYYGYPGGGEYRNGGNGGDSNGNSSDNGGNYAGGGNGQDGGNAGGGGRNRGLRFRNGGNYGGGGAGGGSPARGVGYSGNYRNGRRYAANNRSGSGYGGGNSSMGNRRSRRRGRRSHQGRSRRRRRGGGARNGG",

    # protein1
    "RRYARGRRRRNRRGRRGLRGRRGSRRGHARGPGRARRRGNYYRGGNGGRGGGGGNNGGQGYGGSPGGNGNGYGGNYSGGGVRNGGGYNGGGSGNGDSYGGNNSGSSNGAANNANGSNGMAANGGQYNGGDFDGDDYDLDGNNGGDDDGDDGEADYDDGEDSNGEDSS",

    # protein2
    "RGGGGQGNGGGGGSGNGGGRRGRGGNRQSRRRGRRRRRNGRRRRRRRAGRGLGRNNRSNSRGVGGGNPHNGYGGFGYAYNNNNAMLSGDYYGYNYYGYNSAYPGASGNGGGSGAGYGANGSGDGNDNGNASNNASSNGGGDGDDEDGGGDGDGGDDDGGDGDGDDDEE",

    # protein3
    "RYAYLNRNSRRRFRRRRYRRRNGRRSGNHANRRARRRRGNGGGRPGGGAGGGGGRGRGQGSPGRGSNGYANGGGGNAGGGSGGGNGYSGGGGGGGGSNGQNGGYNGMGGAGGNYDEGSNDDGGGDGNADDGSYDDGDDDDYDSDDSEDGENSNNYANDGNYNGLNVGG",

    # protein4
    "AYYAYYFSYNNNAYADYNNVASNGDNGGSDGSNNGDDDEGADGDDGDDDSGDGEDGSGGGRGNGGGGSGDGEGDGDGGGNGGGQGRGGGGGGGGGGGGNRGGQGNYGGGGGGGSGNNRGGNNSNSRGGGNRLRNRGRRRSRRRRRRRPRRSRNYNPRRANLYAAMYRH",

    # protein5
    "RGSNANRGNNRRVRARRGSRRRGARRRRRNGGRGPGYGRNPFRGGSGGNRMNHRGRGNDRSGGGGGNGYQGGNSGRRGNGNAGDANEGSQGGNGGSSGGGSYGNGYNAYNGYNLADEGGNGDDLGNGDGGSGGGDGDADDDGDGSGYGGGAGGDDGNDYEYNDYYSDG",

    # protein6
    "DDDDDDNDDGGGEGGNDDGGEEDDRDGGGNYGGGGGMDDGGNGANGYDGGSNLANGGGGLNSYAGAPPASNGYRGNNYGNYVNYGRNGSYAYNRYSYSRFGNNSARANNGGGGNSRGGSRNRAGHGRGRGRGRGQGNRGNRGSRGGRGGRSRRRGRGRGGGGQSGGRD",

    # protein7
    "DGDGQDGDGDEGEGGGDGGDNDGGYNNGGEGGGGGNDNDDSGGGNNGSAMSADGGVGGANSYNYRNSRDGYNYGNGAYFRAYPAYGRDYNYRLANRNNRNYGARGPLGNDGSHNNSSGRGNGGGGGNGAGGGGRGSGNGRRRSGQGGRRRRGRSRRRGDSRGRGGRGD",

    # protein8
    "FLLYPYYSGSSNGDNYDANADADAAGDGDNEEGDESDDNDRSDSGDMGYNGGGGNGNDGGGDGGGGNGRRGGGDGGGGGDNGGGRGGGGGRGGQQNNRRGRGGGGGRNRRNARDRNRGGNGRRGAGRNGGGRGRRRRASGRSHGSNSGAGNRYGNYNPGNYYNSYVSY",

    # protein9
    "RYEGGSYRYRGSGGRRNRRRNSRGLRGDRGGRRRGGGGGRGGSRYRGRGRRNGHGNRSNGGGDNGYGSGGRDGNYGGSNFYNNNAAGSGGNEGGGRADPYGGGNNGGGAVDALYNSGGNNGDAGGANASDGGDGADDGGRDYQDPEDQDGMDDGDGYGNNGSNNSGNG",

    # protein10
    "RNANGRNNAGGNNGGGYGRNGASPQGGNSDARGGGLDGYGRRRGRRRHRQNRYARAGNSGDGNGRGDGRGGYRSRRSNGDGGGGGGGGRGNNSSGGGPYMGNGDGGGNGDRGGGRAGNDSGDARSGSLGGGGRGNDEVSYNDGNFGGGNDYDYDDDRYEDSYNNAEYG",

    # protein11
    "DDGGDGDRRGGDEGGDGNDGDDGGGSDNGGGGNNSGGGNNGGGDGNANSGAGAGNGRVAGNANNSYSYNGNYPSPMGSYRGAYSDNFSNNYHLSYRRNRRYLGRANARGYNGGSRYSGRRAQYRGGRNGGGGNGGRGRRGRGGGGQGGGDGRNERGGGDRGEGRDDDR",

    # protein12
    "YYYSYAYRHSANYRNRSNGRNNPGRRAGSRRRNRSGANGGGGGGGRGRRRNGRDGQRRGGGGRGGGNGGARGGDDNDAGDDGEDGDGGGGDEGGGNDDGSGDRELDGGDGDGNGNDGGGSGGGGNGARNGGSNGPNNNGSSGSRADRNGNQSVGGYNGMFLAYRNYYY",

    # protein13
    "PYGNHMGNAYYGYGYGYQPNGGSGGSGNSAGSNNAGYAGGGRGNRRDGGNNERGGDGRDGGNDRDRSDGRRGDGRDGGAEGRGRGGGDRNGNGGGDERDRNDDDRSRGGDGNGDNRAGGNRRRGGQGDNSGGRGSGGARSGSGGRGSNAGGFNGNNYGLLNVAYNSYY",

    # protein14
    "ARSSGDNLNGGDGRSARNGAGRNGGGGGYGGRSDGGPRDARYSSGVNGRNGLYGFPGDYGRDGGYDNGNNNAGRGGSDNRGRGGNGGYGGGRAGGGGSENNRDRGHDGYRGGQGGNGNGGANYSRQDRGDGNASNDEGRDGNRGDGGRGSMESNNYRGDRRGGYYAND",

    # protein15
    "RRGEDQERDGDRGDGGRGGGRGNRGGGSGGGSGGNGGSRAGGSGDGNYNAGSGNGGGSMNSSGNGNGYHAYGVNYLAPYRYYPNFRDNNYGGYYNAGGNSYGLANNSANSANSNQGRGNGAGGNGDRGGGGGGGNDGNDRGRRDRGGGGGRGRRRDGRRDRRDDDDDE"
]


# =====================================================================
# CRITICAL TEMPERATURES
# =====================================================================

Tc = np.array([
    282.60,  # WT
    302.60,  # V1
    307.70,  # V2
    319.90,  # V3
    339.08,  # protein1
    328.67,  # protein2
    331.64,  # protein3
    339.85,  # protein4
    323.79,  # protein5
    312.42,  # protein6
    328.67,  # protein7
    317.78,  # protein8
    312.49,  # protein9
    311.84,  # protein10
    302.85,  # protein11
    312.05,  # protein12
    277.46,  # protein13
    274.37,  # protein14
    283.52   # protein15
])

# =====================================================================
# Compute Pb, Pq, Plambda
# =====================================================================

def calculate_P_components(seq):

    N = len(seq)

    # ---------------------------------------------------------------
    # Gaussian-chain positional variance:
    #
    # s2_i = sigma_i^2 / (N b^2)
    # ---------------------------------------------------------------

    i = np.arange(1, N + 1, dtype=float)

    s2 = (
        ((i / N - 1.0 + 1.0 / N) / 2.0)**2
        + (1.0 - 1.0 / N**2) / 12.0
    ) / 3.0

    s2_sum = s2[:, None] + s2[None, :]

    denom = s2_sum**(3.0 / 2.0)


    # ---------------------------------------------------------------
    # Residue properties
    # ---------------------------------------------------------------

    b_i = np.array(
        [sigma_AA[aa] for aa in seq],
        dtype=float
    )

    q_i = np.array(
        [charge[aa] for aa in seq],
        dtype=float
    )

    lam_i = np.array(
        [hydropathy[aa] for aa in seq],
        dtype=float
    )


    # ---------------------------------------------------------------
    # Pair quantities
    # ---------------------------------------------------------------

    b_ij = (
        b_i[:, None] + b_i[None, :]
    ) / 2.0

    q_ij = (
        q_i[:, None] * q_i[None, :]
    )

    lambda_ij = (
        (lam_i[:, None] + lam_i[None, :]) / 2.0
        - 0.08
    )


    # ---------------------------------------------------------------
    # Pb
    # ---------------------------------------------------------------

    Pb = (
        np.sum(
            b_ij**3 / denom
        )
        / np.sum(b_ij**3)
    )


    # ---------------------------------------------------------------
    # Pq
    # ---------------------------------------------------------------

    Pq = (
        np.sum(q_ij / denom)
        / np.sum(q_ij)
    )


    # ---------------------------------------------------------------
    # Plambda
    # ---------------------------------------------------------------

    lambda_weight = lambda_ij * b_ij**3

    Plambda = (
        np.sum(lambda_weight / denom)
        / np.sum(lambda_weight)
    )


    return -Pb, -Pq, Plambda


# =====================================================================
# Calculate for all sequences
# =====================================================================

Pb_values = []
Pq_values = []
Plambda_values = []

for seq in sequences:

    Pb, Pq, Plambda = calculate_P_components(seq)

    Pb_values.append(Pb)
    Pq_values.append(Pq)
    Plambda_values.append(Plambda)


# =====================================================================
# Print
# =====================================================================

print(
    f"{'name':12s}"
    f"{'N':>6s}"
    f"{'Pb':>16s}"
    f"{'Pq':>16s}"
    f"{'Plambda':>16s}"
)

print("-" * 66)

for name, seq, Pb, Pq, Plambda in zip(
    names,
    sequences,
    Pb_values,
    Pq_values,
    Plambda_values
):

    print(
        f"{name:12s}"
        f"{len(seq):6d}"
        f"{Pb:16.6f}"
        f"{Pq:16.6f}"
        f"{Plambda:16.6f}"
    )


# =====================================================================
# Mean and variance across sequences
# =====================================================================

Pb_values = np.asarray(Pb_values)
Pq_values = np.asarray(Pq_values)
Plambda_values = np.asarray(Plambda_values)

print("\nStatistics across sequences")
print("--------------------------------")

print(
    f"Pb:       mean = {np.mean(Pb_values):.6f}, "
    f"std dev = {np.std(Pb_values):.6f}"
)

print(
    f"Pq:       mean = {np.mean(Pq_values):.6f}, "
    f"std dev = {np.std(Pq_values):.6f}"
)

print(
    f"Plambda:  mean = {np.mean(Plambda_values):.6f}, "
    f"std dev = {np.std(Plambda_values):.6f}"
)

# =====================================================================
# Plot Tc vs Pb
# =====================================================================

def plot_Tc_vs_Pb(Pb_values, Tc):
    X = np.asarray(Pb_values, dtype=float)
    Y = np.asarray(Tc, dtype=float)
    (slope, intercept), cov = np.polyfit(X, Y, 1, cov=True)
    slope_err, intercept_err = np.sqrt(np.diag(cov))
    Yfit = slope * X + intercept
    SS_res = np.sum((Y - Yfit)**2)
    SS_tot = np.sum((Y - np.mean(Y))**2)
    R2 = 1.0 - SS_res / SS_tot

    print("\nTc vs Pb")
    slope_str, slope_err_str = format_value_error(slope, slope_err)
    intercept_str, intercept_err_str = format_value_error(intercept, intercept_err)
    print(f"slope     = {slope_str} +/- {slope_err_str}")
    print(f"intercept = {intercept_str} +/- {intercept_err_str}")
    print(f"R^2       = {R2:.6f}")

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.scatter(X, Y, s=20, linewidths=0)
    xs = np.linspace(X.min(), X.max(), 300)
    ax.plot(xs, slope * xs + intercept, color="crimson", lw=1.8)
    ax.set_xlabel(r"$P_b$")
    ax.set_ylabel(r"$T_{\mathrm{c}}\;[\mathrm{K}]$")

    slope_str, slope_err_str = format_value_error(slope, slope_err)
    intercept_str, intercept_err_str = format_value_error(intercept, intercept_err)

    annotation = (
        rf"$m = {slope_str} \pm {slope_err_str} [\mathrm{{K}}]$" "\n"
        rf"$b = {intercept_str} \pm {intercept_err_str} [\mathrm{{K}}]$" "\n"
        rf"$R^2 = {R2:.4f}$"
    )
    ax.text(0.5, 0.98, annotation, transform=ax.transAxes,
            ha="center", va="top", fontsize=13)
    ax.grid(False)
    fig.tight_layout()
    plt.savefig("_Tc_vs_Pb.pdf", bbox_inches="tight")
    plt.show()

plot_Tc_vs_Pb(
    Pb_values,
    Tc
)

# =====================================================================
# Plot Tc vs Pq
# =====================================================================

def plot_Tc_vs_Pq(Pq_values, Tc):
    X = np.asarray(Pq_values, dtype=float)
    Y = np.asarray(Tc, dtype=float)
    (slope, intercept), cov = np.polyfit(X, Y, 1, cov=True)
    slope_err, intercept_err = np.sqrt(np.diag(cov))
    Yfit = slope * X + intercept
    SS_res = np.sum((Y - Yfit)**2)
    SS_tot = np.sum((Y - np.mean(Y))**2)
    R2 = 1.0 - SS_res / SS_tot

    print("\nTc vs Pq")
    slope_str, slope_err_str = format_value_error(slope, slope_err)
    intercept_str, intercept_err_str = format_value_error(intercept, intercept_err)
    print(f"slope     = {slope_str} +/- {slope_err_str} [K]")
    print(f"intercept = {intercept_str} +/- {intercept_err_str} [K]")
    print(f"R^2       = {R2:.6f}")

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.scatter(X, Y, s=20, linewidths=0)
    xs = np.linspace(X.min(), X.max(), 300)
    ax.plot(xs, slope * xs + intercept, color="crimson", lw=1.8)
    ax.set_xlabel(r"$P_q$")
    ax.set_ylabel(r"$T_{\mathrm{c}}\;[\mathrm{K}]$")

    slope_str, slope_err_str = format_value_error(slope, slope_err)
    intercept_str, intercept_err_str = format_value_error(intercept, intercept_err)

    annotation = (
        rf"$m = {slope_str} \pm {slope_err_str} [\mathrm{{K}}]$" "\n"
        rf"$b = {intercept_str} \pm {intercept_err_str} [\mathrm{{K}}]$" "\n"
        rf"$R^2 = {R2:.4f}$"
    )
    ax.text(0.5, 0.98, annotation, transform=ax.transAxes,
            ha="center", va="top", fontsize=13)
    ax.grid(False)
    fig.tight_layout()
    plt.savefig("_Tc_vs_Pq.pdf", bbox_inches="tight")
    plt.show()

plot_Tc_vs_Pq(
    Pq_values,
    Tc
)

# =====================================================================
# Plot Tc vs Plambda
# =====================================================================

def plot_Tc_vs_Plambda(Plambda_values, Tc):
    X = np.asarray(Plambda_values, dtype=float)
    Y = np.asarray(Tc, dtype=float)
    (slope, intercept), cov = np.polyfit(X, Y, 1, cov=True)
    slope_err, intercept_err = np.sqrt(np.diag(cov))
    Yfit = slope * X + intercept
    SS_res = np.sum((Y - Yfit)**2)
    SS_tot = np.sum((Y - np.mean(Y))**2)
    R2 = 1.0 - SS_res / SS_tot

    print("\nTc vs Plambda")
    slope_str, slope_err_str = format_value_error(slope, slope_err)
    intercept_str, intercept_err_str = format_value_error(intercept, intercept_err)
    print(f"slope     = {slope_str} +/- {slope_err_str}")
    print(f"intercept = {intercept_str} +/- {intercept_err_str}")
    print(f"R^2       = {R2:.6f}")

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.scatter(X, Y, s=20, linewidths=0)
    xs = np.linspace(X.min(), X.max(), 300)
    ax.plot(xs, slope * xs + intercept, color="crimson", lw=1.8)
    ax.set_xlabel(r"$P_{\lambda}$")
    ax.set_ylabel(r"$T_{\mathrm{c}}\;[\mathrm{K}]$")

    slope_str, slope_err_str = format_value_error(slope, slope_err)
    intercept_str, intercept_err_str = format_value_error(intercept, intercept_err)

    annotation = (
        rf"$m = {slope_str} \pm {slope_err_str} [\mathrm{{K}}]$" "\n"
        rf"$b = {intercept_str} \pm {intercept_err_str} [\mathrm{{K}}]$" "\n"
        rf"$R^2 = {R2:.4f}$"
    )
    ax.text(0.5, 0.98, annotation, transform=ax.transAxes,
            ha="center", va="top", fontsize=13)
    ax.grid(False)
    fig.tight_layout()
    plt.savefig("_Tc_vs_Plambda.pdf", bbox_inches="tight")
    plt.show()

plot_Tc_vs_Plambda(
    Plambda_values,
    Tc
)

# =====================================================================
# Compute combined P_{q+lambda}
# =====================================================================

e_charge = 1.602176634e-19       # C
epsilon0 = 8.8541878128e-12      # F/m
epsilon_w = 80.0

NA = 6.02214076e23

# U0 = 0.2 kcal/mol converted to J per molecule
U0 = 0.2 * 4184.0 / NA           # J

kappa_SI = 1.0e9                  # m^-1


def calculate_Pqlambda(seq):

    N = len(seq)

    i = np.arange(1, N + 1, dtype=float)

    s2 = (
        ((i / N - 1.0 + 1.0 / N) / 2.0)**2
        + (1.0 - 1.0 / N**2) / 12.0
    ) / 3.0

    denom = (
        s2[:, None] + s2[None, :]
    )**(3.0 / 2.0)


    b_i = np.array(
        [sigma_AA[aa] for aa in seq]
    )

    q_i = np.array(
        [charge[aa] for aa in seq]
    )

    lam_i = np.array(
        [hydropathy[aa] for aa in seq]
    )


    b_ij = (
        b_i[:, None] + b_i[None, :]
    ) / 2.0

    q_ij = q_i[:, None] * q_i[None, :]

    lambda_ij = (
        (lam_i[:, None] + lam_i[None, :]) / 2.0
        - 0.08
    )


    # Coulomb prefactor, converted from m^3 to Angstrom^3
    Cq = (
        e_charge**2
        / (
            4.0 * np.pi
            * epsilon0
            * epsilon_w
            * U0
            * kappa_SI**2
        )
        * 1.0e30
    )


    pair_term = (
        - Cq * q_ij
        +
        (8.0 * np.sqrt(2.0) / 9.0)
        * lambda_ij
        * b_ij**3
    )

    Pqlambda = (
        np.sum(pair_term / denom)
        / np.sum(pair_term)
    )

    return Pqlambda


Pqlambda_values = [
    calculate_Pqlambda(seq)
    for seq in sequences
]


# =====================================================================
# Plot Tc vs combined P
# =====================================================================

def plot_Tc_vs_Pqlambda(Pqlambda_values, Tc):
    X = np.asarray(Pqlambda_values, dtype=float)
    Y = np.asarray(Tc, dtype=float)
    (slope, intercept), cov = np.polyfit(X, Y, 1, cov=True)
    slope_err, intercept_err = np.sqrt(np.diag(cov))
    Yfit = slope * X + intercept
    SS_res = np.sum((Y - Yfit)**2)
    SS_tot = np.sum((Y - np.mean(Y))**2)
    R2 = 1.0 - SS_res / SS_tot

    print("\nTc vs Pq+lambda")
    slope_str, slope_err_str = format_value_error(slope, slope_err)
    intercept_str, intercept_err_str = format_value_error(intercept, intercept_err)
    print(f"slope     = {slope_str} +/- {slope_err_str}")
    print(f"intercept = {intercept_str} +/- {intercept_err_str}")
    print(f"R^2       = {R2:.6f}")

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.scatter(X, Y, s=20, linewidths=0)
    xs = np.linspace(X.min(), X.max(), 300)
    ax.plot(xs, slope * xs + intercept, color="crimson", lw=1.8)
    ax.set_xlabel(r"$P_{q+\lambda}$")
    ax.set_ylabel(r"$T_{\mathrm{c}}\;[\mathrm{K}]$")

    slope_str, slope_err_str = format_value_error(slope, slope_err)
    intercept_str, intercept_err_str = format_value_error(intercept, intercept_err)

    annotation = (
        rf"$m = {slope_str} \pm {slope_err_str} [\mathrm{{K}}]$" "\n"
        rf"$b = {intercept_str} \pm {intercept_err_str} [\mathrm{{K}}]$" "\n"
        rf"$R^2 = {R2:.4f}$"
    )
    ax.text(0.5, 0.98, annotation, transform=ax.transAxes,
            ha="center", va="top", fontsize=13)
    ax.grid(False)
    fig.tight_layout()
    plt.savefig("_Tc_vs_Pqlambda.pdf", bbox_inches="tight")
    plt.show()

plot_Tc_vs_Pqlambda(
    Pqlambda_values,
    Tc
)

# =====================================================================
# FULL THEORY FIT
#
# Fit
#
#   Tc = [4 pi U0 / (vm kB (1 + N^{-1/2})^2)]
#        * (1/N^2) sum_ij A_ij(Ireg)
#        * [1 - c / (s_i^2 + s_j^2)^{3/2}]
#
# with
#
#   A_ij(Ireg) =
#       -sqrt(2) (1/3 + 4 Ireg) b_ij^3
#       - Cq q_i q_j
#       + (8 sqrt(2)/9) lambda_ij b_ij^3
#
# and
#
#   s_i^2 = sigma_i^2/(N b^2)
#
# Unknown fit parameters:
#   vm   : molecular/reference volume [Angstrom^3]
#   Ireg : regularized-core integral [dimensionless]
#   c    : accessibility coefficient [dimensionless]
# =====================================================================

kB = 1.380649e-23               # J/K


# ---------------------------------------------------------------------
# Precompute sequence-dependent quantities used in the nonlinear fit
# ---------------------------------------------------------------------

def build_fit_data(seq):

    N = len(seq)

    i = np.arange(1, N + 1, dtype=float)

    # s_i^2 = sigma_i^2 / (N b^2)
    s2 = (
        ((i / N - 1.0 + 1.0 / N) / 2.0)**2
        + (1.0 - 1.0 / N**2) / 12.0
    ) / 3.0

    s2_sum = s2[:, None] + s2[None, :]

    b_i = np.array(
        [sigma_AA[aa] for aa in seq],
        dtype=float
    )

    q_i = np.array(
        [charge[aa] for aa in seq],
        dtype=float
    )

    lam_i = np.array(
        [hydropathy[aa] for aa in seq],
        dtype=float
    )

    b_ij = (
        b_i[:, None] + b_i[None, :]
    ) / 2.0

    q_ij = q_i[:, None] * q_i[None, :]

    lambda_ij = (
        (lam_i[:, None] + lam_i[None, :]) / 2.0
        - Delta
    )

    return {
        "N": N,
        "s2_sum": s2_sum,
        "b_ij": b_ij,
        "q_ij": q_ij,
        "lambda_ij": lambda_ij
    }


fit_data = [
    build_fit_data(seq)
    for seq in sequences
]


# ---------------------------------------------------------------------
# Coulomb coefficient
#
# q_i in the charge dictionary is dimensionless valence, so e^2 is
# included explicitly.  Convert m^3 -> Angstrom^3.
# ---------------------------------------------------------------------

Cq_fit = (
    e_charge**2
    / (
        4.0 * np.pi
        * epsilon0
        * epsilon_w
        * U0
        * kappa_SI**2
    )
    * 1.0e30
)


# ---------------------------------------------------------------------
# Tc for one sequence
# ---------------------------------------------------------------------

def Tc_theory_one(data, vm, Ireg, c):

    N = data["N"]
    s2_sum = data["s2_sum"]
    b_ij = data["b_ij"]
    q_ij = data["q_ij"]
    lambda_ij = data["lambda_ij"]

    Aij = (
        -np.sqrt(2.0)
        * (1.0 / 3.0 + 4.0 * Ireg)
        * b_ij**3

        - Cq_fit * q_ij

        + (8.0 * np.sqrt(2.0) / 9.0)
        * lambda_ij
        * b_ij**3
    )

    accessibility = (
        1.0
        - c / s2_sum**(3.0 / 2.0)
    )

    pair_average = (
        np.sum(Aij * accessibility)
        / N**2
    )

    prefactor = (
        4.0 * np.pi * U0
        / (
            vm
            * kB
            * (1.0 + N**(-0.5))**2
        )
    )

    return prefactor * pair_average


# ---------------------------------------------------------------------
# Vector-valued model for curve_fit
# ---------------------------------------------------------------------

def Tc_theory_model(dummy_x, vm, Ireg, c):

    return np.array([
        Tc_theory_one(data, vm, Ireg, c)
        for data in fit_data
    ])


# ---------------------------------------------------------------------
# Nonlinear least-squares fit
# ---------------------------------------------------------------------

Y = np.asarray(Tc, dtype=float)
X_dummy = np.arange(len(Y), dtype=float)

# Starting values.  vm is constrained to remain positive.
p0 = [
    100.0,      # vm [Angstrom^3]
    -0.10,      # Ireg
    1.0e-3      # c
]

popt, pcov = curve_fit(
    Tc_theory_model,
    X_dummy,
    Y,
    p0=p0,
    bounds=(
        [1.0e-12, -np.inf, -np.inf],
        [np.inf,   np.inf,  np.inf]
    ),
    maxfev=500000
)

vm_fit, Ireg_fit, c_fit = popt

Tc_fit = Tc_theory_model(
    X_dummy,
    vm_fit,
    Ireg_fit,
    c_fit
)


# ---------------------------------------------------------------------
# Fit quality
# ---------------------------------------------------------------------

SS_res = np.sum((Y - Tc_fit)**2)
SS_tot = np.sum((Y - np.mean(Y))**2)

R2 = 1.0 - SS_res / SS_tot
RMSE = np.sqrt(np.mean((Y - Tc_fit)**2))


# ---------------------------------------------------------------------
# One-standard-error uncertainties from the local covariance matrix
# ---------------------------------------------------------------------

stderr = np.sqrt(np.diag(pcov))
vm_err, Ireg_err, c_err = stderr


# ---------------------------------------------------------------------
# Parameter correlation matrix
# ---------------------------------------------------------------------

with np.errstate(divide='ignore', invalid='ignore'):
    corr = pcov / np.outer(stderr, stderr)


# ---------------------------------------------------------------------
# Print fitted parameters
# ---------------------------------------------------------------------

print("\n============================================================")
print("FULL THEORY FIT")
print("============================================================")

vm_str, vm_err_str = format_value_error(vm_fit, vm_err)
Ireg_str, Ireg_err_str = format_value_error(Ireg_fit, Ireg_err)
c_str, c_err_str = format_value_error(c_fit, c_err)

print(f"vm    = {vm_str} +/- {vm_err_str} Angstrom^3")
print(f"Ireg  = {Ireg_str} +/- {Ireg_err_str}")
print(f"c     = {c_str} +/- {c_err_str}")
print(f"R^2   = {R2:.8f}")
print(f"RMSE  = {RMSE:.8f} K")

print("\nParameter correlation matrix")
print("rows/columns = [vm, Ireg, c]")
print(corr)


# ---------------------------------------------------------------------
# Print observed and predicted Tc values
# ---------------------------------------------------------------------

print("\nObserved and fitted critical temperatures")
print("------------------------------------------------------------")
print(f"{'name':12s}{'Tc data':>14s}{'Tc fit':>14s}{'residual':>14s}")

for name, y, yfit in zip(names, Y, Tc_fit):
    print(
        f"{name:12s}"
        f"{y:14.4f}"
        f"{yfit:14.4f}"
        f"{(y - yfit):14.4f}"
    )


# ---------------------------------------------------------------------
# Plot observed Tc versus fitted Tc
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6.4, 4.8))

ax.scatter(
    Y,
    Tc_fit,
    s=25,
    linewidths=0
)

xy_min = min(Y.min(), Tc_fit.min())
xy_max = max(Y.max(), Tc_fit.max())

ax.plot(
    [xy_min, xy_max],
    [xy_min, xy_max],
    color="crimson",
    lw=1.8
)

ax.set_xlabel(r"Observed $T_{\mathrm{c}}\;[\mathrm{K}]$")
ax.set_ylabel(r"Fitted $T_{\mathrm{c}}\;[\mathrm{K}]$")

annotation = (
    rf"$\upsilon_m = {vm_str} \pm {vm_err_str}\,\AA^3$" "\n"
    rf"$I_{{\mathrm{{reg}}}} = {Ireg_str} \pm {Ireg_err_str}$" "\n"
    rf"$c = {c_str} \pm {c_err_str}$" "\n"
    rf"$R^2 = {R2:.4f}$"
)
ax.text(
    0.5, 0.98, annotation,
    transform=ax.transAxes,
    ha="center", va="top",
    fontsize=12
)

ax.grid(False)
fig.tight_layout()

plt.savefig(
    "_Tc_full_theory_fit.pdf",
    bbox_inches="tight"
)

plt.show()

