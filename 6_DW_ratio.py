"""
DW / (2,2,0,+) decay-corrected amplitude ratio for each simulation 
and final time in content selection.
"""

import json
import os
import numpy as np
import bgp_qnm_fits as bgp

SXS_IDS         = [f"{i:04d}" for i in range(1, 14)]
DATA_TYPE       = "news"
SPHERICAL_MODES = [(2, 2), (3, 2)]
T_FIT           = 100
DW_MODE         = (2, 2, "DW")
FUND_MODE       = (2, 2, 0, 1)
T0_REF          = 5.0

INCLUDE_CHIF = False
INCLUDE_MF   = False

# Sims where T0_REF falls in a degenerate region; ratio is linearly
# interpolated between the nearest valid bracket points either side.
SXS_IDS_INTERPOLATE = {"0012"}

def load_content(sxs_id):
    path = f"mode_content_files/dw_{sxs_id}_with_dw.json"
    with open(path) as f:
        d = json.load(f)
    t0_vals    = np.array(d["times"])
    modes_list = [[tuple(m) for m in ms] for ms in d["modes"]]
    Mf         = float(d["Mf"])
    chif       = float(d["chif"])
    ppc_med    = d.get("ppc_median", [None] * len(t0_vals))
    return t0_vals, modes_list, Mf, chif, ppc_med


def dropout_times(t0_vals, modes_list):
    """t0 values where DW transitions from present to absent."""
    present = [DW_MODE in ms for ms in modes_list]
    return [
        float(t0_vals[i])
        for i in range(len(present) - 1)
        if present[i] and not present[i + 1]
    ]


def bgp_ratio_interpolated(sim, sxs_id, modes_list, t0_vals, ppc_med, Mf, chif, t0_ref):
    """Linearly interpolate ratio at t0_ref from nearest valid bracket points.

    Uses ppc_median from the content JSON to skip known-degenerate times,
    then verifies each candidate with an actual BGP_fit.
    """
    PPC_MIN = 0.5
    ref_idx = int(np.argmin(np.abs(t0_vals - t0_ref)))

    def valid_candidate(i):
        p = ppc_med[i]
        return (
            p is not None and p >= PPC_MIN
            and DW_MODE in modes_list[i]
            and FUND_MODE in modes_list[i]
        )

    def try_ratio(i):
        r = bgp_ratio(sim, sxs_id, modes_list[i], Mf, chif, float(t0_vals[i]))
        return r if np.isfinite(r) and r > 0 else None

    lo_t, lo_r = None, None
    for i in range(ref_idx, -1, -1):
        if not valid_candidate(i):
            continue
        r = try_ratio(i)
        if r is not None:
            lo_t, lo_r = float(t0_vals[i]), r
            break

    hi_t, hi_r = None, None
    for i in range(ref_idx + 1, len(t0_vals)):
        if not valid_candidate(i):
            continue
        r = try_ratio(i)
        if r is not None:
            hi_t, hi_r = float(t0_vals[i]), r
            break

    if lo_t is None or hi_t is None:
        return None, "interpolation bracket not found"

    frac  = (t0_ref - lo_t) / (hi_t - lo_t)
    ratio = lo_r + frac * (hi_r - lo_r)
    note  = (f"t0={t0_ref:.0f}M interpolated "
             f"[t0={lo_t:.0f}M: {lo_r:.4f}] → [t0={hi_t:.0f}M: {hi_r:.4f}]")
    return ratio, note


def bgp_ratio(sim, sxs_id, modes, Mf, chif, t0):
    """Decay-corrected median |A_DW| / median |A_(2,2,0,1)| from a BGP_fit."""
    tuned = bgp.get_tuned_param_dict("GP", data_type=DATA_TYPE)[sxs_id].copy()
    fit_obj = bgp.BGP_fit(
        sim.times, sim.h, modes,
        Mf, chif,
        tuned, bgp.kernel_GP,
        t0=t0, T=T_FIT,
        decay_corrected=True,
        spherical_modes=SPHERICAL_MODES,
        include_chif=INCLUDE_CHIF,
        include_Mf=INCLUDE_MF,
        data_type=DATA_TYPE,
    )
    q        = fit_obj.fit["unweighted_quantiles"]
    dw_amp   = float(q[0.5][modes.index(DW_MODE)])
    fund_amp = float(q[0.5][modes.index(FUND_MODE)])
    return dw_amp / fund_amp


def main():
    results = {}   # sxs_id → dict with ratio, fit_t0, dropouts, note

    for sxs_id in SXS_IDS:
        path = f"mode_content_files/dw_{sxs_id}_with_dw.json"
        if not os.path.exists(path):
            print(f"SKIP {sxs_id} — JSON not found")
            continue

        t0_vals, modes_list, Mf, chif, ppc_med = load_content(sxs_id)
        dw_present = [DW_MODE in ms for ms in modes_list]

        if not any(dw_present):
            print(f"SKIP {sxs_id} — DW never selected")
            continue

        dropouts = dropout_times(t0_vals, modes_list)

        print(f"Loading SXS:BBH:{sxs_id}...", flush=True)
        try:
            sim = bgp.SXS_CCE(sxs_id, type=DATA_TYPE, lev="Lev5", radius="R2")
        except Exception as e:
            print(f"  SKIP — failed to load: {e}")
            continue

        if sxs_id in SXS_IDS_INTERPOLATE:
            ratio, fit_note = bgp_ratio_interpolated(
                sim, sxs_id, modes_list, t0_vals, ppc_med, Mf, chif, T0_REF
            )
            if ratio is None:
                print(f"  SKIP — {fit_note}")
                continue
        else:
            # Determine fit time: T0_REF if DW present there, else latest available
            ref_idx = int(np.argmin(np.abs(t0_vals - T0_REF)))
            if dw_present[ref_idx]:
                fit_idx  = ref_idx
                fit_note = f"t0={t0_vals[fit_idx]:.0f}M (reference)"
            else:
                fit_idx  = max(i for i, v in enumerate(dw_present) if v)
                fit_note = f"t0={t0_vals[fit_idx]:.0f}M (latest available — DW absent at ref)"

            modes_at_fit = modes_list[fit_idx]
            if FUND_MODE not in modes_at_fit:
                print(f"SKIP {sxs_id} — (2,2,0,1) absent at fit time {t0_vals[fit_idx]:.0f}M")
                continue

            ratio = bgp_ratio(sim, sxs_id, modes_at_fit, Mf, chif,
                              float(t0_vals[fit_idx]))

        results[sxs_id] = {
            "ratio":    ratio,
            "fit_t0":   T0_REF if sxs_id in SXS_IDS_INTERPOLATE else float(t0_vals[fit_idx]),
            "dropouts": dropouts,
            "note":     fit_note,
        }
        print(f"  {fit_note}  ratio={ratio:.4f}"
              + (f"  dropouts={dropouts}" if dropouts else "  (no dropout)"))

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'SXS':>6}  {'fit t0':>8}  {'ratio':>8}  dropout t0(s)")
    print("-" * 70)
    for sxs_id in SXS_IDS:
        if sxs_id not in results:
            continue
        r = results[sxs_id]
        dropout_str = ", ".join(f"{t:.0f}M" for t in r["dropouts"]) or "—"
        flag = " *" if "latest" in r["note"] else (" †" if "interpolated" in r["note"] else "")
        print(f"{sxs_id:>6}  {r['fit_t0']:>7.0f}M  {r['ratio']:>8.4f}  {dropout_str}{flag}")

    print()
    print("* fit performed at latest available t0 (DW absent at reference time)")
    print("† ratio linearly interpolated between nearest valid bracket points (unreliable)")
    print("\nRatios:")
    print([f"{results[s]['ratio']:.4f}" for s in SXS_IDS if s in results])


if __name__ == "__main__":
    main()
