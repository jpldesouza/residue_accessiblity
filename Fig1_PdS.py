import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre


def generate_laguerre_polynomials(num_terms):
    # Generate Laguerre polynomials up to the specified number of terms
    laguerre_polynomials = [genlaguerre(n - 1, 1) for n in range(1, num_terms + 1)]
    return laguerre_polynomials

def hinv(hx, laguerre_polynomials, num_terms=100):
    logeps = np.zeros_like(hx)  # Initialize eps as an array of zeros with the same shape as hx

    # Conditions for different ranges of hx
    cond1 = hx < 1.076
    cond2 = hx > 5
    cond3 = ~cond1 & ~cond2  # hx between 1.06 and 5

    # Apply the first condition
    logeps[cond1] = np.log(1 - np.sqrt(((637875 * hx[cond1] + np.sqrt((637875 * hx[cond1] - 557172)**2 + 5833096416) - 557172)**(1/3) / (45 * 2**(1/3)) - (126 * 2**(1/3)) / (5 * (637875 * hx[cond1] + np.sqrt((637875 * hx[cond1] - 557172)**2 + 5833096416) - 557172)**(1/3)) - 7/15)))

    # Apply the second condition
    logeps[cond2] = np.log(2) - 2 * hx[cond2]

    # Apply the third condition using a vectorized approach
    hx_cond3 = hx[cond3]
    logeps_cond3 = np.zeros_like(hx_cond3)
    for n in range(num_terms):
        L_n_minus_1_1 = laguerre_polynomials[n]  # Use precomputed Laguerre polynomial
        term = -(-1) ** (n + 1) * (2 * np.exp(-2 * hx_cond3 * (n + 1)) / (n + 1)) * L_n_minus_1_1(4 * hx_cond3 * (n + 1))
        logeps_cond3 += term
    logeps_cond3 = np.log(logeps_cond3)
    logeps[cond3] = logeps_cond3

    return logeps


def compute_binodal2(logepsvec, N, laguerre_polynomials):
    # Function that takes in the vector of epsilon=1-y values and the value of N then outputs the the binodal curve in the form of
    # phi_right_vec, phi_left_vec, and chi_vec
    epsvec=np.exp(logepsvec)
    hyvec = 0.5*(np.log(2.0-epsvec)-logepsvec) / (1.0-epsvec)
    hzvec=hyvec / N - 1 / N +1.0
    logepsZvec = hinv(hzvec, laguerre_polynomials)
    epsZvec = np.exp(logepsZvec)
    zvec = 1 - epsZvec
    yvec=1.0-epsvec
    Bvec = 2.0 * zvec / (1.0-epsvec + zvec)
    chi_vec = (1 / N - 1.0) / Bvec  +(- np.log(1-zvec) +np.log (1+zvec)) / (Bvec ** 2 * (1.0-epsvec))
    phi_right_vec = zvec * (1.0 + (1.0-epsvec)) / (zvec + (1.0-epsvec))
    phi_left_vec = zvec * (epsvec) / (zvec + (1.0-epsvec))


    return chi_vec, phi_right_vec, phi_left_vec

# ============================================================
# Parameters
# ============================================================
N = 101      # number of monomers
b = 1.0      # Kuhn length (step length)

# Monomer index (1 ... N)
i = np.arange(1, N + 1, dtype=float)

# ============================================================
# Theory: position-dependent variance relative to COM
# ============================================================
sigma2_theory = (
    b**2 / (18.0 * N**2)
    * ((i - 1) * i * (2 * i - 1)
       + (N - i) * (N - i + 1) * (2 * (N - i) + 1))
)

# Normalization
x = i / N
y = sigma2_theory / (b**2 * N)

# ============================================================
# Plot helpers
# ============================================================
def panel_label(ax, label):
    ax.text(
        -0.05, 1.25, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=18
    )

# ============================================================
# Figure: 2x2 layout, plot is panel (c)
# ============================================================
plt.rcParams.update({
    "font.size": 18,
    "lines.linewidth": 4,
    "axes.linewidth": 1.5,
})

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axA, axB, axC, axD = axes.ravel()

# ----------------------------
# (a) placeholder
# ----------------------------
axA.set_axis_off()
panel_label(axA, "(a)")

# ----------------------------
# (b) placeholder
# ----------------------------
axB.set_axis_off()
panel_label(axB, "(b)")

# ----------------------------
# (c) σ_i^2 plot
# ----------------------------
axC.plot(x, y, color="black")

axC.set_xlabel(r"Sequence position, $i/N$")
axC.set_ylabel(r"Variance, $\sigma_i^2 / (b^2 N)$")
axC.set_title("Position-dependent variance")

axC.tick_params(direction="in", top=True, right=True)
panel_label(axC, "(c)")

# ----------------------------
# (d) placeholder
# ----------------------------
panel_label(axD, "(d)")

# Generate Laguerre polynomials
num_terms = 100
laguerre_polynomials = generate_laguerre_polynomials(num_terms)

# Define epsvec=1-y so that it samples values close to 0
logepsvec = np.linspace(-30, np.log(0.99), 1000)

# Run and plot the function outputs
N = 50  # Define N

# Find the exact solution:
chi_vec, phi_right_vec, phi_left_vec = compute_binodal2(logepsvec, N, laguerre_polynomials)

phi_target = 0.5 * 1 / (1 + np.sqrt(N))
print(np.interp(phi_target, phi_left_vec, chi_vec))

phic = 1 / (1 + np.sqrt(N))
chic = 0.5 * (1 + 1 / np.sqrt(N)) ** 2

chi_vec       = np.append(chi_vec, chic)
phi_right_vec = np.append(phi_right_vec, phic)
phi_left_vec  = np.append(phi_left_vec, phic)

phispinA = (1 + (-1 + 2 * chi_vec) * N - np.sqrt(1 - 2 * (1 + 2 * chi_vec) * N + (1 - 2 * chi_vec)**2 * N**2)) / (4 * chi_vec * N)
phispinB = (1 + (-1 + 2 * chi_vec) * N + np.sqrt(1 - 2 * (1 + 2 * chi_vec) * N + (1 - 2 * chi_vec)**2 * N**2)) / (4 * chi_vec * N)

print(np.interp(phi_target, phispinA, chi_vec))

# --- Curves (on axD) ---
axD.plot(phi_right_vec, 1/chi_vec, color='black', linewidth=2)
axD.plot(phi_left_vec,  1/chi_vec, color='black', linewidth=2)
axD.plot(phi_right_vec, 1.2*(1/chi_vec), color='black', linewidth=2)
axD.plot(phi_left_vec,  1.2*(1/chi_vec), color='black', linewidth=2)

# --- Shading BELOW the curves (on axD) ---
axD.fill_between(phi_right_vec, 0, 1.2*(1/chi_vec), color='#bdd7e7', alpha=1)
axD.fill_between(phi_left_vec,  0, 1.2*(1/chi_vec), color='#bdd7e7', alpha=1)

axD.fill_between(phi_right_vec, 0, 1/chi_vec, color='#6baed6', alpha=1)
axD.fill_between(phi_left_vec,  0, 1/chi_vec, color='#6baed6', alpha=1)

# --- Critical points ---
axD.scatter(1/(1+np.sqrt(N)), 1/(0.5*(1+1/np.sqrt(N))**2), color='black', marker='o', s=50, zorder=5)
axD.scatter(1/(1+np.sqrt(N)), 1.2*1/(0.5*(1+1/np.sqrt(N))**2), color='black', marker='o', s=50, zorder=5)
# Target point
x_pt = 1 / (1 + np.sqrt(N))
y_pt = 1.2 * 1 / (0.5 * (1 + 1 / np.sqrt(N))**2)

# Annotation
axD.annotate(
    "more accessible\ninteractions",
    xy=(x_pt, y_pt),
    xycoords="data",
    xytext=(0.4, 1.9),
    textcoords="data",
    arrowprops=dict(
        arrowstyle="->",
        linewidth=2,
        color="black"
    ),
    fontsize=16,
    ha="left",
    va="center"
)
# --- Styling ---
axD.grid(False)
axD.tick_params(direction="in", top=True, right=True)

axD.set_xlabel(r'Volume fraction, $\phi$', fontsize=20)
axD.set_ylabel(r'$T_c$', fontsize=20)
axD.set_title("Phase diagram")

axD.set_xlim(0, 1)
axD.set_ylim(1.0, 2.0)

panel_label(axD, "(d)")

fig.tight_layout()
plt.savefig("Fig1_PdS.png", dpi=300)
plt.show()