import os
import numpy as np
import matplotlib.pyplot as plt
import qnm
from plot_config import PlotConfig

config = PlotConfig()
config.apply_style()

qnm.download_data()

s = -2
l_max = 3
n_max = 1

qnm_mode_cache = {}
for l in np.arange(2, l_max+1):
  for m in np.arange(-l, l+1):
    for n in np.arange(0, n_max+1):
      qnm_mode_cache[(l, m, n)] = qnm.modes_cache(s=s,
                                                  l=l,
                                                  m=m,
                                                  n=n)

def omega(chi, l, m, n):
  ans, _, _ = qnm_mode_cache[(l, m, n)](a=chi)
  return ans

chi_grid = np.linspace(0., 0.99, 100)

fig, ax = plt.subplots(figsize=(config.fig_width, config.fig_height), dpi=300)

for l in np.arange(2, l_max+1):
  for m in np.arange(-l, l+1):
    for n in np.arange(0, n_max+1):
      if l==3 and n==1:
        pass
      else:
        om = np.array([omega(chi, l, m, n) for chi in chi_grid])
        recip_tau = -om.imag
        f = om.real / (2*np.pi)
        label = ( "QNMs" if
                          (l==2 and m==2 and n==0)
                            else None )
        ax.plot(f, recip_tau, c='k', lw=0.7, ls='-', label=label)
        if l==m:
          for chi in [0, 0.2, 0.4, 0.6, 0.8, 0.99]:
            om = omega(chi, l, m, n)
            recip_tau = -om.imag
            f = om.real / (2*np.pi)
            ax.scatter(f, recip_tau, c='k', s=6, marker='o')

def Omega_H(chi):
  r_plus = 1+np.sqrt(1-chi**2)
  return chi / ( r_plus**2 + chi**2 )

def kappa(chi):
  r_plus = 1+np.sqrt(1-chi**2)
  return np.sqrt(1-chi**2) / ( 2 * r_plus )

recip_tau = np.array([kappa(chi) for chi in chi_grid])
f = 2 * np.array([Omega_H(chi) for chi in chi_grid]) / (2*np.pi)
ax.plot(f, recip_tau, c=config.color_dw, lw=1.5, ls='-', label=r"DW ($m=2$)")
for chi in [0, 0.2, 0.4, 0.6, 0.8, 0.99]:
  recip_tau = kappa(chi)
  f = 2 * Omega_H(chi) / (2*np.pi)
  ax.scatter(f, recip_tau, c=config.color_dw, s=12, marker='o')

ax.set_xlabel(r'$Mf$')
ax.set_ylabel(r'$M/\tau$')

f_max = 0.2
recip_tau_max = 0.3

ax.set_xlim(0, f_max)
ax.set_ylim(0, recip_tau_max)

ax.set_xticks(np.arange(0, f_max+1.0e-3, 0.05))
ax.set_yticks(np.arange(0, recip_tau_max+1.0e-3, 0.05))

ax.annotate(r"DW $\chi=0.8$",
            (Omega_H(0.8)/np.pi, kappa(0.8)),
            (0.015, 0.13),
            color=config.color_dw,
            size=8,
            arrowprops=dict(facecolor=config.color_dw,
                            edgecolor=config.color_dw,
                            width=0.5,
                            shrink=0.15,
                            headwidth=2,
                            headlength=3))

ax.annotate(r"(2,2,0,+) $\chi=0.8$",
            (omega(0.8, 2, 2, 0).real/2/np.pi,
             -omega(0.8, 2, 2, 0).imag - 0.005),
            (0.06, 0.015),
            color='k',
            size=8,
            arrowprops=dict(facecolor='k',
                            edgecolor='k',
                            width=0.5,
                            shrink=0.15,
                            headwidth=2,
                            headlength=3))

ax.annotate(r"(2,2,1,+) $\chi=0.8$",
            (omega(0.8, 2, 2, 1).real/2/np.pi,
             -omega(0.8, 2, 2, 1).imag),
            (0.12, 0.19),
            color='k',
            size=8,
            arrowprops=dict(facecolor='k',
                            edgecolor='k',
                            width=0.5,
                            shrink=0.15,
                            headwidth=2,
                            headlength=3))

ax.annotate(r"(3,3,0,+) $\chi=0.8$",
            (omega(0.8, 3, 3, 0).real/2/np.pi,
             -omega(0.8, 3, 3, 0).imag),
            (0.16, 0.15),
            color='k',
            size=8,
            ha='center',
            arrowprops=dict(facecolor='k',
                            edgecolor='k',
                            width=0.5,
                            shrink=0.15,
                            headwidth=2,
                            headlength=3))

ax.legend(loc='upper right', fontsize=7, frameon=False)
plt.tight_layout()
os.makedirs("figs", exist_ok=True)
plt.savefig('figs/0_frequencies.pdf', bbox_inches='tight')
plt.clf()
