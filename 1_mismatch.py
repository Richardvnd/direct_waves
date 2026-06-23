"""
Mismatch vs start time with/without direct wave.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import bgp_qnm_fits as bgp
from bgp_qnm_fits.qnmfits_funcs import multimode_ringdown_fit

from plot_config import PlotConfig

# ── Configuration ──────────────────────────────────────────────────────────────

SXS_IDS   = ["0004"]
# SXS_IDS = [f"{i:04d}" for i in range(1, 14)]

DATA_TYPE = "news"
N_MAX     = 6
T_FIT     = 100.0
T0_ARRAY  = np.arange(-10.0, 60.1, 0.1)
DW_MODE   = (2, 2, "DW")

SPHERICAL_MODES = [(2, 2)]

# ── Mismatch sweep ─────────────────────────────────────────────────────────────

def mismatch_sweep(t, data_dict, modes, Mf, chif):
    mm = []
    for t0 in T0_ARRAY:
        fit = multimode_ringdown_fit(
            t, data_dict, modes, Mf, chif, t0=t0, T=T_FIT,
            spherical_modes=SPHERICAL_MODES,
        )
        mm.append(fit["mismatch"])
    return np.array(mm)


# ── Resolution uncertainty band ────────────────────────────────────────────────

def compute_resolution_band(sxs_id):
    """
    Estimate numerical uncertainty by comparing Lev5 and Lev4 waveforms.
    """

    sim_L5 = bgp.SXS_CCE(sxs_id, type=DATA_TYPE, lev="Lev5", radius="R2")
    sim_L4 = bgp.SXS_CCE(sxs_id, type=DATA_TYPE, lev="Lev4", radius="R2")

    time_shift = bgp.get_time_shift(
        sim_L5, sim_L4,
        modes=SPHERICAL_MODES,
        delta=0.1, alpha=0.1, t0=-100, T=100,
    )
    sim_L4.zero_time = -time_shift
    sim_L4.time_shift()

    L5_times = np.asarray(sim_L5.times)
    L4_times = np.asarray(sim_L4.times)

    mm_res = []
    for t0 in T0_ARRAY:
        # Mask each level independently (matching get_residuals)
        mask_L5 = (L5_times >= t0 - 1e-9) & (L5_times < t0 + T_FIT - 1e-9)
        mask_L4 = (L4_times >= t0 - 1e-9) & (L4_times < t0 + T_FIT - 1e-9)

        t_L5_win = L5_times[mask_L5]
        t_L4_win = L4_times[mask_L4]

        if len(t_L5_win) < 2 or len(t_L4_win) < 2:
            mm_res.append(np.nan)
            continue

        # dt = native Lev5 step at t0 — matches get_residuals dt=None logic
        idx = np.argmin(np.abs(t_L5_win - t0))
        dt  = t_L5_win[min(idx + 1, len(t_L5_win) - 1)] - t_L5_win[idx]

        # Interpolate both levels onto a common regular grid
        new_times = np.arange(t0, t0 + T_FIT, dt)

        L5_win = {lm: np.asarray(sim_L5.h[lm])[mask_L5] for lm in SPHERICAL_MODES}
        L4_win = {lm: np.asarray(sim_L4.h[lm])[mask_L4] for lm in SPHERICAL_MODES}

        L5_interp = bgp.sim_interpolator_data(L5_win, t_L5_win, new_times)
        L4_interp = bgp.sim_interpolator_data(L4_win, t_L4_win, new_times)

        wf_L5 = np.array([L5_interp[lm] for lm in SPHERICAL_MODES])
        wf_L4 = np.array([L4_interp[lm] for lm in SPHERICAL_MODES])
        mm_res.append(float(bgp.mismatch(wf_L5, wf_L4)))

    return np.array(mm_res)


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot_mismatch(results, cfg, uncertainty=None):
    fig, ax = plt.subplots(figsize=(cfg.fig_width, cfg.fig_height))

    # Resolution uncertainty band (drawn first so lines sit on top)
    if uncertainty is not None:
        ax.fill_between(T0_ARRAY, 0, uncertainty,
                        color="lightgrey", alpha=0.8, zorder=0)

    for n, mm in results["qnm"]:
        ax.semilogy(T0_ARRAY, mm, color=cfg.colors[n], label=rf"$N={n}$")

    ax.semilogy(T0_ARRAY, results["dw"], color=cfg.colors[6],
                lw=1.2, ls="--", label=rf"$N={N_MAX}+\rm DW$")

    ax.semilogy(T0_ARRAY, results["extra"], color=cfg.colors[0],
                lw=1.2, ls="--", label=rf"$N=0+\rm DW$")

    ax.set_xlabel(r"Start time $t_0 \, [M]$")
    ax.set_ylabel(r"Mismatch")
    ax.set_ylim(1e-8, 1)
    ax.set_xlim(T0_ARRAY[0], T0_ARRAY[-1])

    # Colour legend: one entry per N value
    color_handles = [
        Line2D([0], [0], color=cfg.colors[n], lw=1, label=rf"$N={n}$")
        for n in range(N_MAX + 1)
    ]
    leg1 = ax.legend(handles=color_handles, loc="upper right",
                     fontsize=6, ncols=2, frameon=False)
    ax.add_artist(leg1)

    # Linestyle / uncertainty legend
    style_handles = [
        Line2D([0], [0], color="k", lw=1, ls="-",  label="No DW"),
        Line2D([0], [0], color="k", lw=1, ls="--", label="With DW"),
    ]
    leg2 = ax.legend(handles=style_handles, loc="upper right",
                     fontsize=6, frameon=False, bbox_to_anchor=(1.0, 0.65))

    fig.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    cfg = PlotConfig()
    cfg.apply_style()

    for sxs_id in SXS_IDS:
        print(f"\nSXS:BBH:{sxs_id} (CCE, {DATA_TYPE})...", flush=True)

        sim = bgp.SXS_CCE(sxs_id, type=DATA_TYPE, lev="Lev5", radius="R2")

        t    = np.asarray(sim.times)
        h22  = np.asarray(sim.h[(2, 2)])
        Mf   = float(sim.Mf)
        chif = float(sim.chif_mag)

        data_dict = {(2, 2): h22}

        results = {"qnm": [], "dw": None, "extra": None}

        for n in range(N_MAX + 1):
            modes = [(2, 2, k, 1) for k in range(n + 1)]
            print(f"  n_max={n} ({len(modes)} modes)...", flush=True)
            results["qnm"].append((n, mismatch_sweep(t, data_dict, modes, Mf, chif)))

        modes_dw = [(2, 2, k, 1) for k in range(N_MAX + 1)] + [DW_MODE]
        print(f"  n_max={N_MAX} + DW ({len(modes_dw)} modes)...", flush=True)
        results["dw"] = mismatch_sweep(t, data_dict, modes_dw, Mf, chif)

        modes_extra = [(2, 2, 0, 1)] + [DW_MODE]
        print(f"  n_max={N_MAX} + extra ({len(modes_extra)} modes)...", flush=True)
        results["extra"] = mismatch_sweep(t, data_dict, modes_extra, Mf, chif)

        try:
            uncertainty = compute_resolution_band(sxs_id)
        except Exception as e:
            print(f"  Resolution band skipped: {e}")
            uncertainty = None

        out_dir = "figs" if sxs_id == "0004" else "diagnostic_figs"
        os.makedirs(out_dir, exist_ok=True)
        fig = plot_mismatch(results, cfg, uncertainty=uncertainty)
        outpath = f"{out_dir}/1_mismatch_{sxs_id}.pdf"
        fig.savefig(outpath)
        print(f"Saved {outpath}")
        plt.close(fig)


if __name__ == "__main__":
    main()
