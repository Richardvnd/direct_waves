"""
Combined free-frequency figure (nonlinear least-squares and bayesian)

Output: figs/3a_free_frequency_{SXS_ID}.pdf
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import AutoMinorLocator
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.optimize import minimize

import bgp_qnm_fits as bgp
from plot_config import PlotConfig

config = PlotConfig()
config.apply_style()


# ── Configuration ──────────────────────────────────────────────────────────────

SXS_IDS   = ["0004"]
#SXS_IDS = [f"{i:04d}" for i in range(1, 14)]

DATA_TYPE = "news"

TARGET_MODE     = (2, 2, "DW")
SPHERICAL_MODES = [(2, 2), (3, 2)]

# (CONTENT_FILE and MCMC_SAMPLES_FILE are constructed per-simulation in main())

T_GRID  = 100.0
T0_VALS = np.arange(-10.0, 50.1, 1.0)

OMEGA_R_RANGE = (0.3, 1)
KAPPA_RANGE   = (0.01, 0.8)

# MCMC prior bounds (drawn as reference lines on the violin panels)
MCMC_OMEGA_R_PRIOR = (0.4, 0.85)
MCMC_KAPPA_PRIOR   = (0.01, 0.3)

# Modes marked in the complex plane and as horizontal reference lines.
# (2,2) family only — cleanest for the 1-D panels.
MARKER_MODES = [(2, 2, n, 1) for n in range(4)] + [(3, 2, n, 1) for n in range(4)]

# ── Content loader ─────────────────────────────────────────────────────────────

def load_content_modes(path, target_mode):
    """
    Read the per-t0 mode lists produced by 2_DW_content.py.

    Keeps only valid QNM 4-tuples (l, m, n, p) — drops 2-tuple BGP reference
    entries and the target mode itself (which is the free parameter here).

    Returns
    -------
    content_t0s : np.ndarray   shape (N,)
    content_modes : list[list[tuple]]  length N, one list of fixed-mode tuples per t0
    """
    with open(path) as f:
        data = json.load(f)

    target_tuple = tuple(target_mode)
    content_t0s  = np.array(data["times"], dtype=float)
    content_modes = []

    for mode_list in data["modes"]:
        fixed = []
        for m in mode_list:
            t = tuple(m)
            if len(t) != 4:          # drop 2-tuple BGP entries and DW 3-tuples
                continue
            if t == target_tuple:    # drop the target mode (free parameter)
                continue
            fixed.append(t)
        content_modes.append(fixed)

    return content_t0s, content_modes


def load_mcmc_samples(path):
    """Load posterior samples saved by 3_free_frequency.py."""
    data = np.load(path, allow_pickle=True).item()
    return data


# ── Objective factory ──────────────────────────────────────────────────────────

def make_objective(times, data_dict, fixed_modes, target_mode,
                   chif, Mf, t0, T, spherical_modes):
    mask  = (times >= t0 - 1e-9) & (times < t0 + T - 1e-9)
    t_win = times[mask]
    if len(t_win) < 2:
        return None

    data     = np.concatenate([data_dict[lm][mask] for lm in spherical_modes])
    data_arr = np.array([data_dict[lm][mask] for lm in spherical_modes])

    all_modes   = fixed_modes + [target_mode]
    n_fixed     = len(fixed_modes)
    if n_fixed == 0:
        return None   # nothing to fix; free-only fit not meaningful here

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


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_combined(omega_r_fits, kappa_fits, t0_vals,
                  omega_dw, omega_markers, outpath, mcmc_data=None):

    t0_arr = np.asarray(t0_vals)
    vmin, vmax = t0_arr.min(), t0_arr.max()
    cmap = "plasma"
    scatter_kw = dict(c=t0_arr, cmap=cmap, vmin=vmin, vmax=vmax,
                      s=4, zorder=3, edgecolors="k", linewidths=0.1)

    # ── Figure / layout ────────────────────────────────────────────────────────
    # Left col slightly narrower than right; right panel forced square via aspect.
    total_w = config.fig_width_2
    total_h = config.fig_height_2 * 0.8
    fig = plt.figure(figsize=(total_w, total_h), dpi=300)

    gs = GridSpec(
        2, 2,
        figure=fig,
        width_ratios=[1.0, 1.1],
        hspace=0.,
        wspace=0.2,
    )
    ax_re  = fig.add_subplot(gs[0, 0])
    ax_im  = fig.add_subplot(gs[1, 0], sharex=ax_re)
    ax_cmp = fig.add_subplot(gs[:, 1])   # spans both rows → square via set_box_aspect

    # ── MCMC violin helper ─────────────────────────────────────────────────────
    cmap_fn = plt.get_cmap("plasma")
    norm_t0 = plt.Normalize(vmin=vmin, vmax=vmax)

    def draw_violin(ax, t0_m, samples, y_range, width=2.5):
        clipped = samples[(samples >= y_range[0]) & (samples <= y_range[1])]
        if len(clipped) < 10:
            return
        color = cmap_fn(norm_t0(t0_m))
        vp = ax.violinplot([clipped], positions=[t0_m], widths=width,
                           showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(1)
            body.set_edgecolor("k")
            body.set_linewidth(0.5)
            body.set_zorder(4)
        ax.axvline(t0_m, color="0.0", lw=0.5, ls="-", alpha=0.1, zorder=1)
        #vp["cmedians"].set_color(color)
        #vp["cmedians"].set_linewidth(1.2)
        #vp["cmedians"].set_zorder(6)

    # ── Left: Re(ω) panel ─────────────────────────────────────────────────────
    if mcmc_data is not None:
        for t0_m, or_samp in zip(mcmc_data["t0_vals"], mcmc_data["omega_r_samples"]):
            if t0_m < -9:
                continue
            draw_violin(ax_re, t0_m, or_samp, OMEGA_R_RANGE)
    for bound in MCMC_OMEGA_R_PRIOR:
        ax_re.axhline(bound, color="0.0", lw=0.6, ls="-", alpha=0.1, zorder=1)

    ax_re.scatter(t0_arr, omega_r_fits, **scatter_kw)
    ax_re.axhline(omega_dw.real, color=config.color_dw,
                  lw=0.9, ls="--", zorder=2, label=r"$\mathrm{Re}(\omega_{\rm DW}^{(2,2)})$")
    ax_re.set_ylabel(r"$\mathrm{Re}(\omega)\;[M^{-1}]$")
    ax_re.set_xlim(t0_arr[0], t0_arr[-1])
    ax_re.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax_re.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax_re.legend(frameon=False, loc="upper left")
    plt.setp(ax_re.get_xticklabels(), visible=False)

    # ── Left: κ panel ─────────────────────────────────────────────────────────
    if mcmc_data is not None:
        for t0_m, k_samp in zip(mcmc_data["t0_vals"], mcmc_data["kappa_samples"]):
            if t0_m < -9:
                continue
            draw_violin(ax_im, t0_m, k_samp, KAPPA_RANGE)
    for bound in MCMC_KAPPA_PRIOR:
        ax_im.axhline(bound, color="0.0", lw=0.6, ls="-", alpha=0.1, zorder=1)

    ax_im.scatter(t0_arr, kappa_fits, **scatter_kw)
    ax_im.axhline(-omega_dw.imag, color=config.color_dw,
                  lw=0.9, ls="--", zorder=2, label=r"$-\mathrm{Im}(\omega_{\rm DW}^{(2,2)})$")
    ax_im.set_xlabel(r"Start time $t_0 \, [M]$")
    ax_im.set_ylabel(r"$-\mathrm{Im}(\omega)\;[M^{-1}]$")
    ax_im.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax_im.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax_im.legend(frameon=False, loc="upper left")

    # ── Right: complex-plane scatter ───────────────────────────────────────────
    prior_rect = Rectangle(
        (MCMC_OMEGA_R_PRIOR[0], MCMC_KAPPA_PRIOR[0]),
        MCMC_OMEGA_R_PRIOR[1] - MCMC_OMEGA_R_PRIOR[0],
        MCMC_KAPPA_PRIOR[1]   - MCMC_KAPPA_PRIOR[0],
        linewidth=0.6, edgecolor=(0, 0, 0, 0.1), facecolor=(0, 0, 0, 0.03), zorder=0,
    )
    ax_cmp.add_patch(prior_rect)

    family_marker = {(2, 2): ("x", 7), (3, 2): ("+", 8)}
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

    # Colour legend: overtone number n
    n_vals = sorted({mode[2] for mode in MARKER_MODES})
    color_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=config.colors[n], markeredgecolor="k",
               markeredgewidth=0.1, markersize=5, label=f"$n={n}$")
        for n in n_vals
    ]

    # Style legend: marker family + DW star
    style_handles = [
        Line2D([0], [0], marker="*", ls="none", ms=7,
               markerfacecolor=config.color_dw, markeredgecolor="k",
               markeredgewidth=0.1, label=fr"$\omega_{{\rm DW}}^{{(2,2)}}$"),
        Line2D([0], [0], marker="x", color="k", ls="none",
               ms=6, mew=1, label="$(2,2,n)$"),
        Line2D([0], [0], marker="+", color="k", ls="none",
               ms=6, mew=1, label="$(3,2,n)$"),
    ]
    ax_cmp.legend(handles=style_handles, frameon=False,
                  loc="upper left", ncols=3, columnspacing=0.8, handletextpad=0.01)

    # ── Colorbar (vertical, right edge) ───────────────────────────────────────
    cb = fig.colorbar(sc, ax=ax_cmp, location='right', shrink=0.7, pad=0.04, aspect=25)
    cb.set_label(r"$t_0\;[M]$")
    cb.ax.tick_params(labelsize=6)

    # ── Align left panels to square height ────────────────────────────────────
    # Render once so set_box_aspect(1) resolves the square's actual position,
    # then rescale both left axes to span exactly the same vertical extent.
    fig.canvas.draw()
    sq_pos      = ax_cmp.get_position()
    re_pos      = ax_re.get_position()
    im_pos      = ax_im.get_position()
    left_height = re_pos.y1 - im_pos.y0
    scale       = sq_pos.height / left_height
    im_h        = im_pos.height * scale
    re_h        = re_pos.height * scale
    ax_im.set_position([im_pos.x0, sq_pos.y0,          im_pos.width, im_h])
    ax_re.set_position([re_pos.x0, sq_pos.y0 + im_h,   re_pos.width, re_h])

    fig.savefig(outpath, bbox_inches="tight")
    print(f"Saved {outpath}")


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
        print(f"  DW:  Re(omega) = {omega_dw.real:.4f}  kappa = {-omega_dw.imag:.4f}")

        content_file = f"mode_content_files/dw_{sxs_id}_with_dw.json"
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

        mcmc_samples_file = f"mcmc/free_freq_{sxs_id}.npy"
        mcmc_data = None
        if os.path.exists(mcmc_samples_file):
            mcmc_data = load_mcmc_samples(mcmc_samples_file)
            print(f"  Loaded MCMC samples from {mcmc_samples_file}")

        out_dir = "figs" if sxs_id == "0004" else "diagnostic_figs"
        os.makedirs(out_dir, exist_ok=True)
        plot_combined(
            np.array(omega_r_fits), np.array(kappa_fits), np.array(valid_t0s),
            omega_dw, omega_markers,
            f"{out_dir}/3a_free_frequency_{sxs_id}.pdf",
            mcmc_data=mcmc_data,
        )


if __name__ == "__main__":
    main()
