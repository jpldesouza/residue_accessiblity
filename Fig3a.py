import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def eps_ij(c: float) -> np.ndarray:
    E_HH = (1 - c) / 2
    E_HT = (1 - c) / 2
    E_TT = (1 + c) / 2
    return np.array([[E_HH, E_HT],
                     [E_HT, E_TT]], float)

seq_list=np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],
                   [0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])
c=0.75

def chi_av():
    chi_list = []
    for seq in seq_list:
        eps = eps_ij(c)
        S = eps[seq[:, None], seq[None, :]]
        chi = np.sum(S)/64
        chi_list.append(chi)
    return np.array(chi_list)

def sigma2():
    i = np.arange(1, 9, dtype=float)
    j = 8 - i
    sigma2 = ((i - 1) * i * (2 * i - 1) + j * (j + 1) * (2 * j + 1)) / (18 * 8**2)
    return sigma2/8

def chi_de():
    chi_list = []
    for seq in seq_list:
        eps = eps_ij(c)
        sig2 = sigma2()
        S = eps[seq[:, None], seq[None, :]]/(sig2[:,None]+sig2[None,:])**(3./2.)
        chi = np.sum(S)/64
        chi_list.append(chi)
    return np.array(chi_list)

sig2=sigma2()
seq=np.array([0,0,0,0,0,0,0,0])
eps=eps_ij(0)
S = eps[seq[:, None], seq[None, :]]/(sig2[:,None]+sig2[None,:])**(3./2.)
P0 = 32.76 #np.sum(S)/np.sum(eps[seq[:, None], seq[None, :]])

Tc=np.array([1.6713,1.6418,1.6291,1.6236,1.6236,1.6291,1.6418,1.6713])

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'

fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(chi_de()/chi_av(), 'o-', color="#8f00ff",label=r'$P$')
ylabel1 = ax.set_ylabel(r'$P$', fontsize=14, rotation=0, labelpad=8)
ylabel1.set_va('center')  # Set vertical alignment to center

ax.spines['left'].set_color('#8f00ff')
ax.tick_params(axis='y', colors='#8f00ff')
ax.yaxis.label.set_color('#8f00ff')

ax2 = ax.twinx()
ax2.plot(Tc, 's--', color='#ff8c42',label=r'$T_\mathrm{c}$')
ylabel2 = ax2.set_ylabel(r'$T_\mathrm{c}$', fontsize=14, rotation=0, labelpad=8)
ylabel2.set_va('center')  # Set vertical alignment to center

ax2.spines['right'].set_color('#ff8c42')
ax2.tick_params(axis='y', colors='#ff8c42')
ax2.yaxis.label.set_color('#ff8c42')

sequences = ['ABBBBBBB', 'BABBBBBB', 'BBABBBBB', 'BBBABBBB',
             'BBBBABBB', 'BBBBBABB', 'BBBBBBAB', 'BBBBBBBA']

ax.set_xticks(range(8))
ax.set_xticklabels([]) 

circle_char = '●'  # Unicode filled circle
for i, seq in enumerate(sequences):
    label_text = '\n'.join([circle_char for _ in seq])
    colors = ['#0000ff' if letter == 'A' else '#cfb997' for letter in seq]
    
    # Create text with colors
    for j, (char, color) in enumerate(zip(seq, colors)):
        ax.text(i, -0.03 - j*0.06, circle_char, 
                ha='center', va='top', 
                color=color, fontsize=18,
                transform=ax.get_xaxis_transform())
plt.subplots_adjust(bottom=0.4)
plt.tight_layout()
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="center")
plt.savefig('Fig3a.png', dpi=1200)
plt.show()