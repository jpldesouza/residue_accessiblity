import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import matplotlib as mpl

def load_data(filepath: str) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    excel_file = pd.ExcelFile(filepath)
    
    all_decoded_sequences = []
    all_c_values = []
    all_T_values = []
    all_phi_values = []
    
    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        
        sequences = df.iloc[:, 1].tolist()
        c_values = df.iloc[:, 3].values
        T_values = df.iloc[:, 7].values
        phi_values = df.iloc[:, 17].values
        
        # Decode sequences
        for i, seq in enumerate(sequences):
            seq_str = str(seq).strip()
            if seq_str and seq_str != 'nan':
                mapping = {'1': 0, '2': 1}
                decoded_seq = np.array([mapping[c] for c in seq_str], dtype=int)
                all_decoded_sequences.append(decoded_seq)
                all_c_values.append(c_values[i])
                all_T_values.append(T_values[i])
                all_phi_values.append(phi_values[i])
    
    return (all_decoded_sequences, 
            np.array(all_c_values), 
            np.array(all_T_values), 
            np.array(all_phi_values))

def eps_ij(c: float) -> np.ndarray:
    E_HH = (1 - c) / 2
    E_HT = (1 - c) / 2
    E_TT = (1 + c) / 2
    return np.array([[E_HH, E_HT],
                     [E_HT, E_TT]], float)

seq_list, c_list, Tc_list, phic_list = load_data('HTseq_results_FINAL.xlsx')
N_list = np.array([len(seq_list[i]) for i in range(len(seq_list))])
N_unique = np.unique(N_list) 
c_unique = np.unique(c_list)
phic_pwr=0.4
Tc_pwr=0.5

############ accesibility stuff

def chi_av(seq_list, c_list):
    chi_list = []
    for seq, c_val in zip(seq_list, c_list):
        eps = eps_ij(c_val)
        S = eps[seq[:, None], seq[None, :]]
        chi = np.sum(S)/len(seq)**2
        chi_list.append(chi)
    return np.array(chi_list)

av_list=chi_av(seq_list,c_list)


mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'

def plot_violin_for_N(N, ax, color):
    """Plot violin plot for a given N value"""
    mask = (N_list == N)
    
    # Calculate x and y
    x_raw = np.sqrt((av_list - 0.5)/c_list + 0.5)[mask]
    y_raw = (Tc_list * (1 + 1 / N_list**Tc_pwr)**2 / av_list)[mask]/26
    
    # Remove NaN and Inf values
    valid_mask = np.isfinite(x_raw) & np.isfinite(y_raw)
    x_raw = x_raw[valid_mask]
    y_raw = y_raw[valid_mask]

    a, b = np.polyfit(x_raw, y_raw, 1)
    y_pred = a * x_raw + b
    ss_res = np.sum((y_raw - y_pred)**2)
    ss_tot = np.sum((y_raw - np.mean(y_raw))**2)
    r2 = 1 - ss_res / ss_tot

    print("R^2="+str(r2)+f" for N={N}")

    # Get unique x values
    x_unique = np.unique(x_raw)
    print(f"\nN={N}:")
    print(f"Unique x values: {x_unique}")
    print(f"Total data points: {len(x_raw)}")
    
    # Separate data into single points and violin data
    violin_data = []
    violin_positions = []
    
    for x_val in x_unique:
        mask_x = np.isclose(x_raw, x_val, rtol=1e-9)
        y_at_x = y_raw[mask_x]
        
        print(f"x={x_val:.4f}: {len(y_at_x)} points")
        
        # Check if x is 0 or 1 (within tolerance)
        is_endpoint = np.isclose(x_val, 0.0, atol=1e-6) or np.isclose(x_val, 1.0, atol=1e-6)
        
        if is_endpoint or len(y_at_x) == 1:
            # Endpoint or single point - plot as scatter only
            ax.scatter([x_val]*len(y_at_x), y_at_x, s=10, c=color, edgecolors='none', alpha=0.5, zorder=3)
        elif len(y_at_x) > 1:
            # Multiple points at intermediate x - prepare for violin
            violin_data.append(y_at_x) 
            violin_positions.append(x_val)
    
    # Create violin plots only if we have data
    if len(violin_data) > 0:
        # Check that all violin data arrays have multiple points
        valid_violin_data = []
        valid_positions = []
        
        for data, pos in zip(violin_data, violin_positions):
            if len(data) > 1:
                valid_violin_data.append(data)
                valid_positions.append(pos)
        
        if len(valid_violin_data) > 0:
            parts = ax.violinplot(valid_violin_data, positions=valid_positions, 
                                  widths=0.8/N, showmeans=False, showmedians=False)
            
            # Customize violin colors
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_edgecolor('black')
                pc.set_alpha(0.3)
            
            # Plot raw points on top of violins
            for data, pos in zip(valid_violin_data, valid_positions):
                # No jitter - exact x positions
                ax.scatter([pos]*len(data), data, s=10, c=color, edgecolors='none', alpha=0.5, zorder=2)
    else:
        print(f"Warning: No violin plots to display for N={N} (all x values have <=1 point)")

# Create figure
fig, ax = plt.subplots(figsize=(4, 3))

# Plot N=8 in blue
plot_violin_for_N(8, ax, color='#8f00ff')

# Plot N=24 in orange
plot_violin_for_N(24, ax, color='#ff8c42')

# Labels and formatting
ax.set_xlabel(r'$f_\mathrm{A}$', fontsize=14)
ax.set_ylabel(r'$T_\mathrm{c}(1 + 1/N^{1/2})^2 / (z\bar{\epsilon})$', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')

# Create legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#8f00ff', alpha=1, label=r'$N=8$'),
                   Patch(facecolor='#ff8c42', alpha=1, label=r'$N=24$')]
ax.legend(handles=legend_elements, loc='best')

plt.tight_layout()
plt.savefig('Fig2.png', dpi=1200)
plt.show()