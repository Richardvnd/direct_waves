"""
Amplitude figure for the BGP_select mode sets from 4_DW_content.py.

A BGP_fit is run at each t0 using the mode set selected at that start time.
Two stacked panels (no-DW top, with-DW bottom) show the decay-corrected
(2,2,n,+1) amplitudes with 25/75% posterior bands, and the DW amplitude in
config.color_dw.

Loads:  mode_content_files/dw_0004_no_dw.json
        mode_content_files/dw_0004_with_dw.json
Output: figs/2b_amplitudes_{SXS_ID}.pdf
"""

import json
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import bgp_qnm_fits as bgp
from plot_config import PlotConfig

PPC_THRESHOLD = 0.5

# Earliest t0 considered when searching for the first PPC crossing below threshold.
PPC_CUTOFF = {
    "0001": -5, "0002": -5, "0003": -5, "0004": -5,
    "0005": -5, "0006": -5, "0007": -5, "0008": -10,
    "0009": 10, "0010": -5, "0011":  0, "0012":  20, "0013": 10,
}

config = PlotConfig()
config.apply_style()

SXS_IDS   = ["0004"]
# SXS_IDS = [f"{i:04d}" for i in range(1, 14)]

DATA_TYPE = "news"
T_FIT     = 100
INCLUDE_CHIF = False
INCLUDE_MF   = False


# ── Data loading ───────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        d = json.load(f)
    modes_list = [list(map(tuple, inner)) for inner in d["modes"]]
    sph_modes  = [tuple(m) for m in d["spherical_modes"]]
    ppc = {
        "median": d.get("ppc_median"),
        "q25":    d.get("ppc_q25"),
        "q75":    d.get("ppc_q75"),
    }
    return modes_list, np.array(d["times"]), sph_modes, float(d["Mf"]), float(d["chif"]), ppc


# ── Fitting ────────────────────────────────────────────────────────────────────

def get_fits(sim, sxs_id, t0_vals, full_modes_list, spherical_modes, Mf, chif):
    tuned = bgp.get_tuned_param_dict("GP", data_type=DATA_TYPE)[sxs_id].copy()
    fits = []
    for i, t0 in enumerate(t0_vals):
        modes = full_modes_list[i]
        print(f"  BGP_fit t0={t0:.1f}  modes={len(modes)}", flush=True)
        if not modes:
            fits.append(None)
            continue
        fits.append(bgp.BGP_fit(
            sim.times, sim.h, modes,
            Mf, chif,
            tuned, bgp.kernel_GP,
            t0=t0, T=T_FIT,
            decay_corrected=True,
            spherical_modes=spherical_modes,
            include_chif=INCLUDE_CHIF,
            include_Mf=INCLUDE_MF,
            data_type=DATA_TYPE,
        ))
    return fits


def masks(mode, t0_vals, full_modes_list):
    """Index arrays for each contiguous t0 range where mode is selected."""
    present = np.array([mode in full_modes_list[i] for i in range(len(t0_vals))])
    changes = np.diff(present.astype(int))
    starts  = list(np.where(changes == 1)[0] + 1)
    ends    = list(np.where(changes == -1)[0] + 1)
    if present[0]:
        starts.insert(0, 0)
    if present[-1]:
        ends.append(len(present))
    return [np.arange(s, e) for s, e in zip(starts, ends)]


# ── Plotting ───────────────────────────────────────────────────────────────────

def draw_amplitudes(ax, t0_vals, full_modes_list, fits):
    unique_modes = set(m for ms in full_modes_list for m in ms)
    plot_modes = sorted(
        [m for m in unique_modes if len(m) == 4 and m[0] == 2 and m[1] == 2 and m[3] == 1],
        key=lambda m: m[2],
    ) + [m for m in unique_modes if len(m) == 3]

    dt = float(np.median(np.diff(t0_vals)))

    for mode in plot_modes:
        is_dw = len(mode) == 3
        if is_dw:
            color, lw, alpha = config.color_dw, 1.8, 1
        else:
            n = mode[2]
            color = config.colors[n]
            lw    = 0.7
            alpha = 1

        for run in masks(mode, t0_vals, full_modes_list):
            amps, lowers, uppers, t_vals = [], [], [], []
            for tidx in run:
                if fits[tidx] is None:
                    continue
                idx = full_modes_list[tidx].index(mode)
                q = fits[tidx].fit["unweighted_quantiles"]
                med = float(q[0.5][idx])
                lo  = float(q[0.25][idx])
                hi  = float(q[0.75][idx])
                if not (np.isfinite(med) and np.isfinite(lo) and np.isfinite(hi) and med > 0):
                    continue
                lowers.append(lo)
                amps.append(med)
                uppers.append(hi)
                t_vals.append(float(t0_vals[tidx]))

            if not amps:
                continue
            t_run = np.array(t_vals)
            amps, lowers, uppers = np.array(amps), np.array(lowers), np.array(uppers)

            if len(t_run) == 1:
                t_run  = np.array([t_run[0] - dt / 2, t_run[0] + dt / 2])
                amps   = np.array([amps[0],   amps[0]])
                lowers = np.array([lowers[0], lowers[0]])
                uppers = np.array([uppers[0], uppers[0]])

            ax.plot(t_run, amps, color=color, lw=lw, alpha=alpha)
            ax.fill_between(t_run, lowers, uppers,
                            color=color, alpha=0.15, linewidth=0)


def add_ppc_span(ax, t0_vals, ppc, cutoff):
    """Grey axvspan from the plot left edge to the first t0 >= cutoff where PPC
    crosses below PPC_THRESHOLD."""
    if ppc["median"] is None:
        return
    dt = float(np.median(np.diff(t0_vals)))
    med = [v if v is not None else np.nan for v in ppc["median"]]
    threshold_idx = next(
        (i for i in range(len(med))
         if t0_vals[i] >= cutoff and not np.isnan(med[i]) and med[i] < PPC_THRESHOLD),
        None,
    )
    if threshold_idx is not None:
        ax.axvspan(t0_vals[0] - 0.5 * dt, t0_vals[threshold_idx] - 0.5 * dt,
                   color="grey", alpha=0.2, zorder=0)


def plot_amplitudes(no_dw_fits, dw_fits, no_dw_modes, dw_modes, t0_vals,
                    no_dw_ppc, dw_ppc, outpath, ppc_cutoff=-5.0):
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(config.fig_width, config.fig_height*1.6),
        sharex=True, dpi=300,
    )

    draw_amplitudes(ax_top, t0_vals, no_dw_modes, no_dw_fits)
    draw_amplitudes(ax_bot, t0_vals, dw_modes,    dw_fits)

    add_ppc_span(ax_top, t0_vals, no_dw_ppc, cutoff=ppc_cutoff)
    add_ppc_span(ax_bot, t0_vals, dw_ppc,    cutoff=ppc_cutoff)

    # QNM legend on top panel
    qnm_ns = sorted(set(
        m[2] for ms in no_dw_modes for m in ms
        if len(m) == 4 and m[0] == 2 and m[1] == 2 and m[3] == 1
    ))
    qnm_handles = [
        Line2D([0], [0], color=config.colors[n], lw=1.2, label=rf"$n={n}$")
        for n in qnm_ns
    ]
    ax_top.legend(handles=qnm_handles, fontsize=6, loc="upper right",
                  ncol=2, frameon=False)

    # DW legend on bottom panel
    dw_handle = Line2D([0], [0], color=config.color_dw, lw=1.8, label=r"$\rm DW$")
    ax_bot.legend(handles=[dw_handle], fontsize=6, loc="upper right", ncol=2, frameon=False)

    dt = float(np.median(np.diff(t0_vals)))
    for ax in (ax_top, ax_bot):
        ax.set_xlim(t0_vals[0] - 0.5 * dt, t0_vals[-1] + 0.5 * dt)
        ax.set_ylabel(r"$|\hat{C}_\alpha|$")
        ax.set_yscale("log")

    # Panel labels in bottom-right corner instead of titles
    for ax, label in [(ax_top, "No DW"), (ax_bot, "With DW")]:
        ax.text(0.98, 0.04, label, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="k")

    ax_bot.set_xlabel(r"Start time $t_0 \, [M]$")

    ax_top.set_xlim(-9 - 0.5 * dt, t0_vals[-1] + 0.5 * dt)
    ax_bot.set_xlim(-9 - 0.5 * dt, t0_vals[-1] + 0.5 * dt)

    plt.subplots_adjust(hspace=0.08)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    print(f"Saved {outpath}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    for sxs_id in SXS_IDS:
        no_dw_path   = f"mode_content_files/dw_{sxs_id}_no_dw.json"
        with_dw_path = f"mode_content_files/dw_{sxs_id}_with_dw.json"
        if not os.path.exists(no_dw_path) or not os.path.exists(with_dw_path):
            print(f"SKIP {sxs_id} — JSON files not found")
            continue

        no_dw_modes, t0_vals, sph_modes, Mf, chif, no_dw_ppc = load_json(no_dw_path)
        dw_modes, _, _, _, _, dw_ppc = load_json(with_dw_path)

        print(f"\nLoading SXS:BBH:{sxs_id} (CCE)...", flush=True)
        try:
            sim = bgp.SXS_CCE(sxs_id, type=DATA_TYPE, lev="Lev5", radius="R2")
        except Exception as e:
            print(f"  SKIP — failed to load: {e}")
            continue

        print("Fitting amplitudes [no DW]...", flush=True)
        no_dw_fits = get_fits(sim, sxs_id, t0_vals, no_dw_modes, sph_modes, Mf, chif)

        print("Fitting amplitudes [with DW]...", flush=True)
        dw_fits = get_fits(sim, sxs_id, t0_vals, dw_modes, sph_modes, Mf, chif)

        out_dir = "figs" if sxs_id == "0004" else "diagnostic_figs/mode_amplitudes"
        plot_amplitudes(
            no_dw_fits, dw_fits, no_dw_modes, dw_modes, t0_vals,
            no_dw_ppc, dw_ppc,
            f"{out_dir}/2b_amplitudes_{sxs_id}.pdf",
            ppc_cutoff=PPC_CUTOFF.get(sxs_id, -5.0),
        )


if __name__ == "__main__":
    main()
