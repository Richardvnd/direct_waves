"""
Free-frequency (nonlinear least-squares) figure for the (3,3) DW mode.

Mirrors 3a_free_frequency_plot.py but without violin plots: scatter only.

Output: figs/5b_free_frequency_{SXS_ID}.pdf
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import AutoMinorLocator
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.optimize import minimize

import bgp_qnm_fits as bgp
from plot_config import PlotConfig

config = PlotConfig()
config.apply_style()


# ── Configuration ──────────────────────────────────────────────────────────────

SXS_IDS = ["0010"]
# SXS_IDS = [f"{i:04d}" for i in range(1, 14)]

DATA_TYPE = "news"

TARGET_MODE     = (3, 3, "DW")
SPHERICAL_MODES = [(3, 3), (4, 3)]

T_GRID  = 100.0
T0_VALS = np.arange(-10.0, 50.1, 1.0)

OMEGA_R_RANGE = (0.4, 1.5)
KAPPA_RANGE   = (0.01, 0.8)

MARKER_MODES = [(3, 3, n, 1) for n in range(4)] + [(4, 3, n, 1) for n in range(4)]


# ── Content loader ─────────────────────────────────────────────────────────────

def load_content_modes(path, target_mode):
    with open(path) as f:
        data = json.load(f)
    target_tuple  = tuple(target_mode)
    content_t0s   = np.array(data["times"], dtype=float)
    content_modes = []
    for mode_list in data["modes"]:
        fixed = []
        for m in mode_list:
            t = tuple(m)
            if len(t) != 4:
                continue
            if t == target_tuple:
                continue
            fixed.append(t)
        content_modes.append(fixed)
    return content_t0s, content_modes


# ── Objective factory ──────────────────────────────────────────────────────────

def make_objective(times, data_dict, fixed_modes, target_mode,
                   chif, Mf, t0, T, spherical_modes):
    mask  = (times >= t0 - 1e-9) & (times < t0 + T - 1e-9)
    t_win = times[mask]
    if len(t_win) < 2:
        return None

    data_arr = np.array([data_dict[lm][mask] for lm in spherical_modes])
    data     = data_arr.ravel()

    all_modes = fixed_modes + [target_mode]
    n_fixed   = len(fixed_modes)
    if n_fixed == 0:
        return None

    fixed_freqs   = np.array(bgp.qnm.omega_list(fixed_modes, chif, Mf))
    indices_lists = [[lm + mode for mode in all_modes] for lm in spherical_modes]
    mu_lists      = [bgp.qnm.mu_list(indices, chif) for indices in indices_lists]

    a_fixed = np.concatenate([
        np.array([
            mu_lists[i][j] * np.exp(-1j * fixed_freqs[j] * (t_win - t0))
            for j in range(n_fixed)
        ]).T
        for i in range(len(spherical_modes))
    ])
    mu_free = [mu_lists[i][n_fixed] for i in range(len(spherical_modes))]

    def objective(x):
        omega_free = x[0] - 1j * x[1]
        a_free = np.concatenate([
            (mu_free[ii] * np.exp(-1j * omega_free * (t_win - t0))).reshape(-1, 1)
            for ii in range(len(spherical_modes))
        ])
        a = np.hstack([a_fixed, a_free])
        C, *_ = np.linalg.lstsq(a, data, rcond=None)
        model = np.einsum("ij,j->i", a, C).reshape(len(spherical_modes), len(t_win))
        return float(bgp.mismatch(model, data_arr))

    return objective


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot_scatter(omega_r_fits, kappa_fits, t0_vals,
                 omega_dw, omega_markers, outpath):

    t0_arr = np.asarray(t0_vals)
    vmin, vmax = t0_arr.min(), t0_arr.max()
    scatter_kw = dict(c=t0_arr, cmap="plasma", vmin=vmin, vmax=vmax,
                      s=4, zorder=3, edgecolors="k", linewidths=0.1)

    total_w = config.fig_width_2
    total_h = config.fig_height_2 * 0.8
    fig = plt.figure(figsize=(total_w, total_h), dpi=300)

    gs = GridSpec(2, 2, figure=fig,
                  width_ratios=[1.0, 1.1], hspace=0., wspace=0.2)
    ax_re  = fig.add_subplot(gs[0, 0])
    ax_im  = fig.add_subplot(gs[1, 0], sharex=ax_re)
    ax_cmp = fig.add_subplot(gs[:, 1])

    # ── Left: Re(ω) panel ─────────────────────────────────────────────────────
    ax_re.scatter(t0_arr, omega_r_fits, **scatter_kw)
    ax_re.axhline(omega_dw.real, color=config.color_dw,
                  lw=0.9, ls="--", zorder=2,
                  label=r"$\mathrm{Re}(\omega_{\rm DW}^{(3,3)})$")
    ax_re.set_ylabel(r"$\mathrm{Re}(\omega)\;[M^{-1}]$")
    ax_re.set_xlim(t0_arr[0], t0_arr[-1])
    ax_re.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax_re.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax_re.legend(frameon=False, loc="upper left")
    plt.setp(ax_re.get_xticklabels(), visible=False)

    # ── Left: κ panel ─────────────────────────────────────────────────────────
    ax_im.scatter(t0_arr, kappa_fits, **scatter_kw)
    ax_im.axhline(-omega_dw.imag, color=config.color_dw,
                  lw=0.9, ls="--", zorder=2,
                  label=r"$-\mathrm{Im}(\omega_{\rm DW}^{(3,3)})$")
    ax_im.set_xlabel(r"Start time $t_0 \, [M]$")
    ax_im.set_ylabel(r"$-\mathrm{Im}(\omega)\;[M^{-1}]$")
    ax_im.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax_im.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax_im.legend(frameon=False, loc="upper left")

    # ── Right: complex-plane scatter ───────────────────────────────────────────
    family_marker = {(3, 3): ("x", 7), (4, 3): ("+", 8)}
    ax_cmp.plot(omega_dw.real, -omega_dw.imag,
                "*", color=config.color_dw, ms=9, zorder=7,
                path_effects=[pe.Stroke(linewidth=1.2, foreground="k"), pe.Normal()])
    for mode, om in zip(MARKER_MODES, omega_markers):
        l, m, n_ot, _ = mode
        mk, ms_mk = family_marker.get((l, m), ("x", 7))
        ax_cmp.plot(om.real, -om.imag,
                    mk, color=config.colors[n_ot], ms=ms_mk, mew=1.3, zorder=6,
                    path_effects=[pe.Stroke(linewidth=1.5, foreground="k"), pe.Normal()])
    sc = ax_cmp.scatter(omega_r_fits, kappa_fits, **{**scatter_kw, "zorder": 8})
    ax_cmp.set_xlim(*OMEGA_R_RANGE)
    ax_cmp.set_ylim(*KAPPA_RANGE)
    ax_cmp.set_xlabel(r"$\mathrm{Re}(\omega)\;[M^{-1}]$")
    ax_cmp.set_ylabel(r"$-\mathrm{Im}(\omega)\;[M^{-1}]$")
    ax_cmp.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax_cmp.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax_cmp.set_box_aspect(1)

    style_handles = [
        Line2D([0], [0], marker="*", ls="none", ms=7,
               markerfacecolor=config.color_dw, markeredgecolor="k",
               markeredgewidth=0.1, label=fr"$\omega_{{\rm DW}}^{{(3,3)}}$"),
        Line2D([0], [0], marker="x", color="k", ls="none",
               ms=6, mew=1, label="$(3,3,n)$"),
        Line2D([0], [0], marker="+", color="k", ls="none",
               ms=6, mew=1, label="$(4,3,n)$"),
    ]
    ax_cmp.legend(handles=style_handles, frameon=False,
                  loc="upper left", ncols=3, columnspacing=0.8, handletextpad=0.01)

    cb = fig.colorbar(sc, ax=ax_cmp, location="right", shrink=0.7, pad=0.04, aspect=25)
    cb.set_label(r"$t_0\;[M]$")
    cb.ax.tick_params(labelsize=6)

    # Align left panels to the square's vertical extent
    fig.canvas.draw()
    sq_pos      = ax_cmp.get_position()
    re_pos      = ax_re.get_position()
    im_pos      = ax_im.get_position()
    left_height = re_pos.y1 - im_pos.y0
    scale       = sq_pos.height / left_height
    im_h        = im_pos.height * scale
    re_h        = re_pos.height * scale
    ax_im.set_position([im_pos.x0, sq_pos.y0,        im_pos.width, im_h])
    ax_re.set_position([re_pos.x0, sq_pos.y0 + im_h, re_pos.width, re_h])

    os.makedirs(os.path.dirname(outpath) if os.path.dirname(outpath) else ".", exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    print(f"Saved {outpath}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    for sxs_id in SXS_IDS:
        print(f"\n{'='*60}", flush=True)
        print(f"SXS:BBH:{sxs_id}", flush=True)
        print(f"{'='*60}", flush=True)

        try:
            sim = bgp.SXS_CCE(sxs_id, type=DATA_TYPE, lev="Lev5", radius="R2")
        except Exception as e:
            print(f"  SKIP — failed to load: {e}")
            continue

        times = np.asarray(sim.times)
        Mf    = float(sim.Mf)
        chif  = float(sim.chif_mag)
        data_dict = {lm: np.asarray(sim.h[lm]) for lm in SPHERICAL_MODES}

        omega_dw      = bgp.qnm.omega_list([TARGET_MODE], chif, Mf)[0]
        omega_markers = bgp.qnm.omega_list(MARKER_MODES, chif, Mf)

        print(f"  Mf={Mf:.4f}  chif={chif:.4f}")
        print(f"  DW(3,3):  Re(omega) = {omega_dw.real:.4f}  kappa = {-omega_dw.imag:.4f}")

        content_file = f"mode_content_files/dw_{sxs_id}_33_with_dw.json"
        if not os.path.exists(content_file):
            print(f"  SKIP — {content_file} not found")
            continue
        content_t0s, content_modes_list = load_content_modes(content_file, TARGET_MODE)
        print(f"  Loaded content from {content_file}  ({len(content_t0s)} t0 entries)")

        bounds = [OMEGA_R_RANGE, KAPPA_RANGE]
        x0     = [omega_dw.real, -omega_dw.imag]

        omega_r_fits = []
        kappa_fits   = []
        valid_t0s    = []

        print(f"  Minimising over {len(T0_VALS)} start times  T={T_GRID} M")
        for k, t0 in enumerate(T0_VALS):
            idx         = int(np.argmin(np.abs(content_t0s - t0)))
            fixed_modes = content_modes_list[idx]

            obj = make_objective(
                times, data_dict, fixed_modes, TARGET_MODE,
                chif, Mf, t0, T_GRID, SPHERICAL_MODES,
            )
            if obj is None:
                continue

            result = minimize(obj, x0, method="Nelder-Mead", bounds=bounds)
            omega_r_min, kappa_min = result.x
            print(f"  [{k+1:02d}/{len(T0_VALS)}]  t0={t0:+.1f}"
                  f"  n_fixed={len(fixed_modes)}"
                  f"  min={result.fun:.2e}"
                  f"  at (omega_r={omega_r_min:.4f}, kappa={kappa_min:.4f})"
                  f"  nit={result.nit}", flush=True)
            omega_r_fits.append(omega_r_min)
            kappa_fits.append(kappa_min)
            valid_t0s.append(t0)

        out_dir = "figs" if sxs_id == "0010" else "diagnostic_figs"
        plot_scatter(
            np.array(omega_r_fits), np.array(kappa_fits), np.array(valid_t0s),
            omega_dw, omega_markers,
            f"{out_dir}/5b_free_frequency_{sxs_id}.pdf",
        )


if __name__ == "__main__":
    main()
