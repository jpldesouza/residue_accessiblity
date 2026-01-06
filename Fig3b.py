import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# =====================================================================
# GLOBAL PLOT STYLE — Times New Roman
# =====================================================================
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


# =====================================================================
# Global settings
# =====================================================================
PLOT_DIR = "plot_pdfs"
DEFAULT_EXCEL = "./HTseq_results_FINAL.xlsx"
PHI_COL_INDEX = 17   # column R, 0-based index


# =====================================================================
# Utilities
# =====================================================================
def ensure_output_dir(path: str = PLOT_DIR) -> None:
    os.makedirs(path, exist_ok=True)


def decode_sequence(seq_string: str) -> str:
    mapping = {"1": "H", "2": "T"}
    return "".join(mapping[c] for c in str(seq_string).strip())


def interaction_matrix_from_c(c: float) -> np.ndarray:
    E_HH = -(1 - c) / 2
    E_HT = -(1 - c) / 2
    E_TT = -(1 + c) / 2
    return np.array([[E_HH, E_HT],
                     [E_HT, E_TT]], float)


def gaussian_params_for_chain(N: int, b: float = 1.0) -> Tuple[np.ndarray, None, None]:
    i = np.arange(1, N + 1, dtype=float)
    j = N - i
    sigma2 = b**2 * ((i - 1) * i * (2 * i - 1) +
                     j * (j + 1) * (2 * j + 1)) / (18 * N**2)
    return sigma2, None, None


def delta_chi_and_sum(S: np.ndarray, sigma: np.ndarray) -> Tuple[float, float]:
    sigma2 = sigma**2
    denom = (sigma2[:, None] + sigma2[None, :])**1.5
    delta_S = -S / denom
    val = float(np.sum(delta_S))
    return val, val


def fit_slope_fixed_intercept(
    X: np.ndarray,
    Y: np.ndarray,
    intercept: float = 1.0
) -> Optional[Tuple[float, float]]:
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)

    mask = np.isfinite(X) & np.isfinite(Y)
    if mask.sum() < 2:
        return None

    Xv = X[mask]
    Yv = Y[mask]

    denom = np.sum(Xv**2)
    if denom == 0:
        return None

    m = np.sum(Xv * (Yv - intercept)) / denom

    Y_pred = m * Xv + intercept
    SS_res = np.sum((Yv - Y_pred)**2)
    SS_tot = np.sum((Yv - np.mean(Yv))**2)
    R2 = 1 - SS_res / SS_tot if SS_tot > 0 else np.nan

    return float(m), float(R2)


# =====================================================================
# Data processing
# =====================================================================
def process_all_chains(
    filename: str,
    b: float = 1.0,
    phi_col_index: int = PHI_COL_INDEX
) -> pd.DataFrame:
    sheets = pd.read_excel(filename, sheet_name=None)
    all_rows = []

    for _, df_sheet in sheets.items():
        df_sheet.columns = [c.strip().lower() for c in df_sheet.columns]
        Tc_key = "tc" if "tc" in df_sheet.columns else "t"

        for _, row in df_sheet.iterrows():
            seq = decode_sequence(str(row["sequence"]))
            N = len(seq)
            c_val = float(row["c"])

            E = interaction_matrix_from_c(c_val)
            type_idx = np.array([0 if s == "H" else 1 for s in seq], dtype=int)
            S = E[type_idx[:, None], type_idx[None, :]]

            interaction_sum = float(np.sum(S))
            sigma2, _, _ = gaussian_params_for_chain(N, b)
            sigma = np.sqrt(sigma2)
            _, delta_sum = delta_chi_and_sum(S, sigma)

            Tc_raw = float(row[Tc_key])
            Tc_scaled = (
                -Tc_raw / interaction_sum
                * (1.0 + 1.0 / np.sqrt(N))**2
                * N**2 / 26
            )

            phi_c = float(row.iloc[phi_col_index])

            all_rows.append({
                "length": int(N),
                "interaction_sum": interaction_sum,
                "delta_interaction_sum": delta_sum,
                "Tc": Tc_scaled,
                "phi_c": phi_c,
            })

    df = pd.DataFrame(all_rows)

    N_arr = df["length"].astype(float).values
    delta_sum = df["delta_interaction_sum"].values
    interaction_sum = df["interaction_sum"].values

    with np.errstate(divide="ignore", invalid="ignore"):
        P = -N_arr**1.5 * (delta_sum / interaction_sum)

    P0_inf = 32.7617
    df["P"] = P / P0_inf

    return df


# =====================================================================
# Density smoothing utilities
# =====================================================================
def gaussian_kernel1d(sigma_bins: float) -> np.ndarray:
    radius = int(np.ceil(3.0 * sigma_bins))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
    return k / np.sum(k)


def smooth2d_separable(H: np.ndarray, sigma_bins: float) -> np.ndarray:
    k = gaussian_kernel1d(sigma_bins)
    pad = len(k) // 2

    Hx = np.pad(H, ((0, 0), (pad, pad)), mode="reflect")
    tmp = np.array([np.convolve(row, k, mode="valid") for row in Hx])

    Hy = np.pad(tmp, ((pad, pad), (0, 0)), mode="reflect")
    out = np.array([np.convolve(Hy[:, j], k, mode="valid")
                    for j in range(Hy.shape[1])]).T
    return out


# =====================================================================
# Plotting
# =====================================================================
def plot_Tc_vs_P_scatter(df: pd.DataFrame) -> Tuple[float, float]:
    X = df["P"].values
    Y = df["Tc"].values
    Nvals = df["length"].astype(int).values

    mask = np.isfinite(X) & np.isfinite(Y)
    X, Y, Nvals = X[mask], Y[mask], Nvals[mask]

    unique_N = np.sort(np.unique(Nvals))
    cmap = plt.get_cmap("viridis", len(unique_N))
    color_map = {N: cmap(i) for i, N in enumerate(unique_N)}

    fig, ax = plt.subplots(figsize=(4, 3))

    for N in unique_N:
        mN = Nvals == N
        ax.scatter(
            X[mN], Y[mN],
            s=14,
            alpha=0.9,
            color=color_map[N],
            linewidths=0,
            label=f"N = {N}",
        )

    intercept = 1.0
    fit_info = fit_slope_fixed_intercept(X, Y, intercept=intercept)
    slope = np.nan

    if fit_info is not None:
        slope, R2 = fit_info
        xs = np.linspace(X.min(), X.max(), 300)
        ax.plot(xs, slope * xs + intercept, color="crimson", lw=1.8)

        print("Fixed-intercept fit (y = m x + 1):")
        print(f"  slope m = {slope:.6f}")
        print(f"  R^2     = {R2:.6f}")

    ax.set_xlabel(r"$P/P_{0,\infty}$")
    ax.set_ylabel(r"$T_c\left(1 + 1/\sqrt{N}\right)^2/(z\bar{\epsilon})$")
    ax.grid(False)

    if len(unique_N) <= 6:
        ax.legend(frameon=False, fontsize=8, loc="best")
    else:
        ax.legend(frameon=False, fontsize=7,
                  bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    plt.savefig("Fig3b.png", dpi=1200, bbox_inches="tight")
    plt.close(fig)

    return slope, intercept


def plot_Tc_vs_P_smoothed_density_contourf(
    df: pd.DataFrame,
    fit_slope: float,
    fit_intercept: float,
    n_bins: int = 80,
    smooth_sigma_bins: float = 1.5,
    n_levels: int = 12,
) -> None:
    X = df["P"].values
    Y = df["Tc"].values

    mask = np.isfinite(X) & np.isfinite(Y)
    X, Y = X[mask], Y[mask]

    H, xedges, yedges = np.histogram2d(X, Y, bins=n_bins, density=True)
    H = H.T

    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    Xc, Yc = np.meshgrid(xc, yc, indexing="xy")

    Hs = smooth2d_separable(H, sigma_bins=smooth_sigma_bins)

    fig, ax = plt.subplots(figsize=(4, 3))

    finite = np.log(Hs[Hs > 0])
    levels = np.linspace(2.0, finite.max(), n_levels)

    cf = ax.contourf(Xc, Yc, np.log(Hs), levels=levels, cmap="viridis")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("log PDF (smoothed)")

    xs = np.linspace(X.min(), X.max(), 300)
    ax.plot(xs, fit_slope * xs + fit_intercept, color="crimson", lw=1.4)

    ax.set_xlabel(r"$P/P_{0,\infty}$")
    ax.set_ylabel(r"$T_c\left(1 + 1/\sqrt{N}\right)^2/(z\bar{\epsilon})$")
    ax.grid(False)

    fig.tight_layout()
    plt.savefig("Fig3b_density_contourf.png", dpi=1200, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================
def main():
    ensure_output_dir(PLOT_DIR)

    df = process_all_chains(DEFAULT_EXCEL)
    slope, intercept = plot_Tc_vs_P_scatter(df)

    plot_Tc_vs_P_smoothed_density_contourf(
        df,
        fit_slope=slope,
        fit_intercept=intercept,
        n_bins=80,
        smooth_sigma_bins=1.5,
        n_levels=12,
    )


if __name__ == "__main__":
    main()
    print("12233434636")
