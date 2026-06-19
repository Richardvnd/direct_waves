"""
Mode content figure for SXS:BBH:0004 (CCE, news), no-DW vs with-DW.

Loads:  mode_content_files/dw_0004_no_dw.json
        mode_content_files/dw_0004_with_dw.json
Output: figs/2a_DW_content_{SXS_ID}.png
"""

import json
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator

from plot_config import PlotConfig

config = PlotConfig()
config.apply_style()

#SXS_IDS = ["0004"]
SXS_IDS = [f"{i:04d}" for i in range(1, 14)]

BAR_H    = 0.07   # half-height of each sub-bar
OFFSET   = 0.08   # sub-bar centre offset from row centre
ROW_STEP = 0.35   # vertical spacing between rows

PPC_THRESHOLD = 0.5

# Earliest t0 considered when searching for the first PPC crossing below threshold.
# Values earlier than this are ignored to avoid spurious crossings.
PPC_CUTOFF = {
    "0001": -5, "0002": -5, "0003": -5, "0004": -5,
    "0005": -5, "0006": -5, "0007": -5, "0008": -10,
    "0009": 10, "0010": -5, "0011":  0, "0012":  20, "0013": 10,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        d = json.load(f)
    modes   = [list(map(tuple, inner)) for inner in d["modes"]]
    t0_vals = np.array(d["times"])
    ppc = {
        "median": d.get("ppc_median"),
        "q25":    d.get("ppc_q25"),
        "q75":    d.get("ppc_q75"),
    }
    return modes, t0_vals, ppc


def build_layout(no_dw_modes, dw_modes):
    """Flat y-position map: DW at bottom, then (2,2,n=6..0) at top."""
    all_modes = set(m for ms in no_dw_modes + dw_modes for m in ms)

    qnm_pro = sorted(
        [m for m in all_modes if len(m) == 4 and m[0] == 2 and m[1] == 2 and m[3] == 1],
        key=lambda m: m[2], reverse=True,
    )
    dw = [m for m in all_modes if len(m) == 3]

    ordered = dw + qnm_pro
    key_positions = {m: i * ROW_STEP for i, m in enumerate(ordered)}
    return key_positions, len(ordered) * ROW_STEP


# ── Drawing ────────────────────────────────────────────────────────────────────

def draw_bars(ax, t0_vals, no_dw_modes, dw_modes, key_positions):
    dt = float(np.median(np.diff(t0_vals)))

    for t_idx, t0 in enumerate(t0_vals):
        width = (t0_vals[t_idx + 1] - t0) if t_idx < len(t0_vals) - 1 else dt
        x0 = t0 - width / 2
        no_dw_set = set(no_dw_modes[t_idx])
        dw_set    = set(dw_modes[t_idx])

        for mode, y in key_positions.items():
            if len(mode) == 4:
                l, m, n, p = mode
                if p != 1:
                    continue
                alpha = 0.8 #max(0.2, 1.0 - 0.12 * n)
                color = config.colors[n]
                for dy, mode_set in [(+OFFSET, no_dw_set), (-OFFSET, dw_set)]:
                    if mode not in mode_set:
                        continue
                    retro = (l, m, n, -1) in mode_set
                    ax.broken_barh(
                        [(x0, width)], (y + dy - BAR_H, 2 * BAR_H),
                        facecolors=color, alpha=alpha, edgecolor="none",
                        hatch="///////////" if retro else None,
                    )

            elif len(mode) == 3:
                if mode not in dw_set:
                    continue
                ax.broken_barh(
                    [(x0, width)], (y - OFFSET - BAR_H, 2 * BAR_H),
                    facecolors=config.color_dw, alpha=0.8, edgecolor="none",
                )


def annotate_axes(ax, key_positions, t0_vals):
    dt = float(np.median(np.diff(t0_vals)))

    for mode, y in key_positions.items():
        if len(mode) == 4 and mode[3] == 1:
            n = mode[2]
            ax.text(t0_vals[0] - 0.5 * dt, y, rf"$n={n}$",
                    fontsize=8, va="center", ha="right")
        elif len(mode) == 3:
            ax.text(t0_vals[0] - 0.5 * dt, y - OFFSET, r"$\rm DW$",
                    fontsize=8, va="center", ha="right")

    # Sub-row labels beside the n=0 row
    n0 = next((m for m in key_positions if len(m) == 4 and m[2] == 0 and m[3] == 1), None)
    if n0:
        y_top = key_positions[n0]
        for dy, label in [(+OFFSET, "No DW"), (-OFFSET, "With DW")]:
            ax.text(
                t0_vals[-1] + 0.3 * dt, y_top + dy, label, va="center", ha="left", color="black", clip_on=False,
            )


def add_ppc_span(ax, t0_vals, ppc, cutoff, alpha=0.2):
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
        ax.axvspan(
            t0_vals[0],
            t0_vals[threshold_idx] - dt / 2,
            color="grey", alpha=alpha, zorder=0,
        )


# ── Main plot ──────────────────────────────────────────────────────────────────

def plot_content(no_dw_modes, dw_modes, t0_vals, no_dw_ppc, dw_ppc, outpath, ppc_cutoff=-5.0):
    key_positions, y_total = build_layout(no_dw_modes, dw_modes)
    dt = float(np.median(np.diff(t0_vals)))

    fig, ax = plt.subplots(
        figsize=(config.fig_width_2 * 1.2, config.fig_height_2 * (len(key_positions) / 14)),
        dpi=300,
    )

    draw_bars(ax, t0_vals, no_dw_modes, dw_modes, key_positions)
    annotate_axes(ax, key_positions, t0_vals)
    add_ppc_span(ax, t0_vals, dw_ppc, cutoff=ppc_cutoff, alpha=0.15)

    ax.set_yticks(list(key_positions.values()))
    ax.set_yticklabels([""] * len(key_positions))
    ax.set_xlabel(r"Start time $t_0 \, [M]$")
    ax.set_ylabel(r"Ringdown mode $(2, 2, n)$", labelpad=23)
    ax.set_xlim(t0_vals[0], t0_vals[-1])
    ax.set_ylim(-ROW_STEP * 0.7, y_total)
    ax.tick_params(axis="y", direction="out", which="both", right=False)
    ax.xaxis.set_minor_locator(FixedLocator(np.arange(t0_vals[0], t0_vals[-1] + dt, 5)))

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    print(f"Saved {outpath}")


def main():
    for sxs_id in SXS_IDS:
        no_dw_path   = f"mode_content_files/dw_{sxs_id}_no_dw.json"
        with_dw_path = f"mode_content_files/dw_{sxs_id}_with_dw.json"
        if not os.path.exists(no_dw_path) or not os.path.exists(with_dw_path):
            print(f"SKIP {sxs_id} — JSON files not found")
            continue
        no_dw_modes, t0_vals, no_dw_ppc = load_json(no_dw_path)
        dw_modes, _,          dw_ppc    = load_json(with_dw_path)
        out_dir = "figs" if sxs_id == "0004" else "diagnostic_figs/mode_content"
        plot_content(no_dw_modes, dw_modes, t0_vals, no_dw_ppc, dw_ppc,
                     f"{out_dir}/2a_DW_content_{sxs_id}.png",
                     ppc_cutoff=PPC_CUTOFF.get(sxs_id, -5.0))


if __name__ == "__main__":
    main()
