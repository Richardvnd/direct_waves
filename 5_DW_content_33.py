"""
Run BGP_select for each simulation in SXS_IDS (CCE, news) with and without
the direct wave candidate mode in the (3,3) sector.  Outputs two JSON files
per simulation:
  mode_content_files/dw_{SXS_ID}_33_no_dw.json
  mode_content_files/dw_{SXS_ID}_33_with_dw.json
"""

import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import bgp_qnm_fits as bgp

from plot_config import PlotConfig

config = PlotConfig()
config.apply_style()

# ── Configuration ──────────────────────────────────────────────────────────────

SXS_IDS = ["0010"]
# SXS_IDS = [f"{i:04d}" for i in range(1, 14)]

DATA_TYPE = "news"
NOISE_MODEL = "GP"
N_MAX     = 6
T_FIT     = 100
THRESHOLD = 0.9999
INCLUDE_RETRO = True
INCLUDE_CHIF  = False
INCLUDE_MF    = False
N_DRAWS       = 1000
PPC_SAMPLES   = 1000

T0_VALS = np.arange(-10.0, 60.1, 1.0)
SPHERICAL_MODES = [(3, 3), (4, 3)]
DW_MODE = (3, 3, "DW")

OUTPUT_DIR = "mode_content_files"


# ── Selection + PPC ───────────────────────────────────────────────────────────

def run_bgp_select(sim, sxs_id, candidate_modes):
    tuned = bgp.get_tuned_param_dict(NOISE_MODEL, data_type=DATA_TYPE)[sxs_id].copy()
    noise_model = bgp.kernel_GP if NOISE_MODEL == "GP" else bgp.kernel_WN

    full_modes_list = []
    ppc_median_list, ppc_q25_list, ppc_q75_list = [], [], []

    for t0 in T0_VALS:
        print(f"  t0={t0:.1f}", flush=True)
        sel = bgp.BGP_select(
            sim.times,
            sim.h,
            [],
            sim.Mf, sim.chif_mag,
            tuned, noise_model,
            t0=t0,
            candidate_modes=candidate_modes,
            log_threshold=np.log(THRESHOLD),
            candidate_type="prograde_sequential",
            num_draws=N_DRAWS,
            T=T_FIT,
            spherical_modes=SPHERICAL_MODES,
            include_chif=INCLUDE_CHIF,
            include_Mf=INCLUDE_MF,
            data_type=DATA_TYPE,
        )
        modes = list(sel.full_modes)
        full_modes_list.append([list(m) for m in modes])

        if modes:
            fit_obj = bgp.BGP_fit(
                sim.times, sim.h, modes,
                sim.Mf, sim.chif_mag,
                tuned.copy(), noise_model,
                t0=t0, T=T_FIT,
                num_samples=PPC_SAMPLES,
                spherical_modes=SPHERICAL_MODES,
                include_chif=INCLUDE_CHIF,
                include_Mf=INCLUDE_MF,
                data_type=DATA_TYPE,
            )
            pvs = np.array(fit_obj.fit["p_values"])
            ppc_median_list.append(float(np.median(pvs)))
            ppc_q25_list.append(float(np.percentile(pvs, 25)))
            ppc_q75_list.append(float(np.percentile(pvs, 75)))
        else:
            ppc_median_list.append(None)
            ppc_q25_list.append(None)
            ppc_q75_list.append(None)

    return full_modes_list, ppc_median_list, ppc_q25_list, ppc_q75_list


# ── Saving ────────────────────────────────────────────────────────────────────

def save_json(path, sxs_id, sim, candidate_modes, full_modes_list,
              ppc_median, ppc_q25, ppc_q75, run_time):
    data = {
        "sim_id":          sxs_id,
        "times":           T0_VALS.tolist(),
        "Mf":              float(sim.Mf),
        "chif":            float(sim.chif_mag),
        "spherical_modes": SPHERICAL_MODES,
        "threshold":       THRESHOLD,
        "initial_modes":   [],
        "candidate_modes": [list(m) for m in candidate_modes],
        "include_chif":    INCLUDE_CHIF,
        "include_Mf":      INCLUDE_MF,
        "modes":           full_modes_list,
        "p_values":        [None] * len(T0_VALS),
        "ppc_median":      ppc_median,
        "ppc_q25":         ppc_q25,
        "ppc_q75":         ppc_q75,
        "run_time":        run_time,
    }
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"Saved {path}")


# ── PPC check plot ────────────────────────────────────────────────────────────

def plot_ppc_check(sxs_id, no_dw_ppc, dw_ppc, outpath):
    fig, ax = plt.subplots()

    for label, ppc, color, ls in [
        ("no DW",    no_dw_ppc, "steelblue",    "-"),
        ("with DW",  dw_ppc,    config.color_dw, "--"),
    ]:
        med = np.array([v if v is not None else np.nan for v in ppc["median"]])
        q25 = np.array([v if v is not None else np.nan for v in ppc["q25"]])
        q75 = np.array([v if v is not None else np.nan for v in ppc["q75"]])
        ax.plot(T0_VALS, med, color=color, ls=ls, lw=1.2, label=label)
        ax.fill_between(T0_VALS, q25, q75, color=color, alpha=0.2, linewidth=0)

    ax.axhline(0.5, color="k", lw=0.7, ls=":", alpha=0.5, label="p=0.5")
    ax.set_xlabel("Start time $t_0$ [M]")
    ax.set_ylabel("PPC (median p-value)")
    ax.set_ylim(0, 1)
    ax.set_xlim(T0_VALS[0], T0_VALS[-1])
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f"SXS:BBH:{sxs_id} — posterior predictive check (3,3)", fontsize=9)

    os.makedirs("diagnostic_figs/ppcs", exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"Saved {outpath}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    parities = (1, -1) if INCLUDE_RETRO else (1,)
    base_candidates = (
        [(3, 3, n, s) for n in range(N_MAX + 1) for s in parities] +
        [(4, 3, n, s) for n in range(N_MAX + 1) for s in parities] +
        [(3, 3), (4, 3)]
    )
    dw_candidates = base_candidates + [DW_MODE]

    for sxs_id in SXS_IDS:
        print(f"\n{'='*60}", flush=True)
        print(f"SXS:BBH:{sxs_id}", flush=True)
        print(f"{'='*60}", flush=True)

        try:
            sim = bgp.SXS_CCE(sxs_id, type=DATA_TYPE, lev="Lev5", radius="R2")
            print(f"  Mf={sim.Mf:.6f}  chif={sim.chif_mag:.6f}")
        except Exception as e:
            print(f"  SKIP — failed to load: {e}")
            continue

        print("Running BGP_select [no DW]...", flush=True)
        t_start = time.time()
        modes_no_dw, ppc_med_no_dw, ppc_q25_no_dw, ppc_q75_no_dw = run_bgp_select(
            sim, sxs_id, base_candidates
        )
        save_json(
            f"{OUTPUT_DIR}/dw_{sxs_id}_33_no_dw.json",
            sxs_id, sim, base_candidates, modes_no_dw,
            ppc_med_no_dw, ppc_q25_no_dw, ppc_q75_no_dw,
            time.time() - t_start,
        )

        print("Running BGP_select [with DW]...", flush=True)
        t_start = time.time()
        modes_with_dw, ppc_med_dw, ppc_q25_dw, ppc_q75_dw = run_bgp_select(
            sim, sxs_id, dw_candidates
        )
        save_json(
            f"{OUTPUT_DIR}/dw_{sxs_id}_33_with_dw.json",
            sxs_id, sim, dw_candidates, modes_with_dw,
            ppc_med_dw, ppc_q25_dw, ppc_q75_dw,
            time.time() - t_start,
        )

        no_dw_ppc = {"median": ppc_med_no_dw, "q25": ppc_q25_no_dw, "q75": ppc_q75_no_dw}
        dw_ppc    = {"median": ppc_med_dw,    "q25": ppc_q25_dw,    "q75": ppc_q75_dw}
        plot_ppc_check(sxs_id, no_dw_ppc, dw_ppc, f"diagnostic_figs/ppcs/5_ppc_check_{sxs_id}.pdf")


if __name__ == "__main__":
    main()
