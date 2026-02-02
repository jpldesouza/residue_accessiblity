import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors

index_c=2
index_t=1

def eps_ij(c: float) -> np.ndarray:
    E_HH = (1 - c) / 2
    E_HT = (1 - c) / 2
    E_TT = (1 + c) / 2
    if index_t == 0:
        return np.array([[E_HH, E_HT],
                     [E_HT, E_TT]], float)
    return np.array([[E_TT, E_HT],
                     [E_HT, E_HH]], float)
    

arrays = []
arrays.append(np.concatenate([np.ones(6), np.zeros(42)]))

# next three arrays: shifting block of ones
for k in range(1, 4):
    arrays.append(
        np.concatenate([
            np.zeros(6 * k),
            np.ones(6),
            np.zeros(48 - 6 * (k + 1))
        ])
    )

seq_list = np.array(arrays).astype(int)
c=np.arange(0,1.01,0.25)

def chi_av():
    chi_list = []
    for seq in seq_list:
        eps = eps_ij(c[index_c])
        S = eps[seq[:, None], seq[None, :]]
        chi = np.sum(S)/48**2
        chi_list.append(chi)
    return np.array(chi_list)

def sigma2():
    i = np.arange(1, 49, dtype=float)
    j = 48 - i
    sigma2 = ((i - 1) * i * (2 * i - 1) + j * (j + 1) * (2 * j + 1)) / (18 * 48**2)
    return sigma2/48

def chi_de():
    chi_list = []
    for seq in seq_list:
        eps = eps_ij(c[index_c])
        sig2 = sigma2()
        S = eps[seq[:, None], seq[None, :]]/(sig2[:,None]+sig2[None,:])**(3./2.)
        chi = np.sum(S)/48**2
        chi_list.append(chi)
    return np.array(chi_list)

sig2=sigma2()
eps=eps_ij(c[index_c])
P0 = 32.76

Tc=np.array([[[8.0938,8.0938,8.0938,8.0938],
              [6.1543,6.1406,6.1338,6.1293],
              [4.2938,4.2166,4.2005,4.1947],
              [np.nan,np.nan,np.nan,np.nan],
              [np.nan,np.nan,np.nan,np.nan]],
             [[8.0938,8.0938,8.0938,8.0938],
              [9.0478,9.1648,9.2091,9.2247],
              [9.9728,10.2364,10.3332,10.3694],
              [np.nan,11.3233,11.4851,11.5455],
              [np.nan,12.4095,12.6642,12.7501]]])

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'

fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(np.concatenate((chi_de()/chi_av(),(chi_de()/chi_av())[::-1])), 'o-', color="#8f00ff",label=r'$P$')
ylabel1 = ax.set_ylabel(r'$P$', fontsize=14, rotation=0, labelpad=8)
ylabel1.set_va('center')  # Set vertical alignment to center

ax.spines['left'].set_color('#8f00ff')
ax.tick_params(axis='y', colors='#8f00ff')
ax.yaxis.label.set_color('#8f00ff')

ax2 = ax.twinx()
cmap = cm.get_cmap('YlGnBu')   # beige → blue vibe
norm = mcolors.Normalize(vmin=0, vmax=4)
ax2.plot(np.concatenate((Tc[index_t,index_c],Tc[index_t,index_c][::-1])), 's--', color='#ff8c42', label=r'$T_\mathrm{c}$')
ylabel2 = ax2.set_ylabel(r'$T_\mathrm{c}$', fontsize=14, rotation=0, labelpad=8)
ylabel2.set_va('center')  # Set vertical alignment to center

ax2.spines['right'].set_color('#ff8c42')
ax2.tick_params(axis='y', colors='#ff8c42')
ax2.yaxis.label.set_color('#ff8c42')

sequences = ['ABBBBBBB', 
             'BABBBBBB', 
             'BBABBBBB', 
             'BBBABBBB',
             'BBBBABBB',
             'BBBBBABB',
             'BBBBBBAB',
             'BBBBBBBA']


ax.set_xticks(range(8))
ax.set_xticklabels([]) 

circle_char = '●'  # Unicode filled circle
for i, seq in enumerate(sequences):
    label_text = '\n'.join([circle_char for _ in seq])
    if index_t == 0:
        colors = ['#0000ff' if letter == 'A' else '#cfb997' for letter in seq]
    else:
        colors = ['#0000ff' if letter == 'B' else '#cfb997' for letter in seq]
    
    # Create text with colors
    for j, (char, color) in enumerate(zip(seq, colors)):
        ax.text(i, -0.03 - j*0.06, circle_char, 
                ha='center', va='top', 
                color=color, fontsize=18,
                transform=ax.get_xaxis_transform())
        ax.text(i-0.006, -0.066 - j*0.06, '6',
                ha='center', va='top',
                color='black', fontsize=8,
                transform=ax.get_xaxis_transform(),
                zorder=3)
plt.subplots_adjust(bottom=0.4)
plt.tight_layout()
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="center")
plt.savefig('Fig3a48.png', dpi=1200)
plt.show()