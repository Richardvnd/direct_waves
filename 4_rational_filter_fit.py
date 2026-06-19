"""
Rational-filter then BGP_fit the direct wave.

The (2,2,0,1), (2, 2, 1, 1), and (3,2,0,1) fundamentals are removed via a rational filter.
A BGP_fit with the single DW mode (2,2,"DW") is then run on the residual signal. 

Output: figs/4_rational_filter_fit_{SXS_ID}.pdf
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import qnmfits
import bgp_qnm_fits as bgp
from plot_config import PlotConfig

config = PlotConfig()
config.apply_style()


# ── Configuration ──────────────────────────────────────────────────────────────

SXS_IDS      = ["0004"]
# SXS_IDS = [f"{i:04d}" for i in range(1, 14)]

DATA_TYPE    = "news"
FILTER_MODES = [(2, 2, 0, 1), (2, 2, 1, 1), (3, 2, 0, 1)]   # modes to remove via rational filter
M_MODE       = 2

FIT_T0   = 0.0   # BGP_fit start time [M]
FIT_T    = 30.0  # BGP_fit window duration [M]

T_START = -90.0  # rational-filter resampling grid start [M]
T_TAPER =  40.0  # taper length at start of grid [M]

PLOT_XLIM      = (-20, 80)
PLOT_YLIM      = (1e-5, 1.0)
N_BAND_SAMPLES = 1000

SPHERICAL_MODES = [(2, 2)]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    for sxs_id in SXS_IDS:
        print(f"\nSXS:BBH:{sxs_id}", flush=True)

        try:
            sim = bgp.SXS_CCE(sxs_id, type=DATA_TYPE, lev="Lev5", radius="R2")
        except Exception as e:
            print(f"  SKIP — failed to load: {e}")
            continue

        times = np.asarray(sim.times)
        h22   = np.asarray(sim.h[(2, 2)])
        Mf    = float(sim.Mf)
        chif  = float(sim.chif_mag)

        # ── Rational filter ──────────────────────────────────────────────────
        print(f"  Filtering: {FILTER_MODES}", flush=True)
        dt = float(np.min(np.diff(times)))
        t_filt, h_filt = qnmfits.rational_filter(
            times=times, data=h22, modes=FILTER_MODES,
            Mf=Mf, chif=chif,
            t_start=T_START, dt=dt, t_taper=T_TAPER,
        )

        # ── BGP_fit on filtered signal — fixed ω_H ──────────────────────────
        tuned   = bgp.get_tuned_param_dict("GP", data_type=DATA_TYPE)[sxs_id].copy()
        fit_obj = bgp.BGP_fit(
            t_filt, {(2, 2): h_filt}, [(2, 2, "DW")],
            Mf, chif,
            tuned, bgp.kernel_GP,
            t0=FIT_T0, T=FIT_T,
            include_chif=False,
            include_Mf=False,
            spherical_modes=SPHERICAL_MODES,
            data_type=DATA_TYPE,
        )
        fit = fit_obj.fit

        analysis_t    = np.array(fit["analysis_times"])
        model_complex = np.array(fit["model_array_linear"][0])

        samples    = np.array(fit["samples"])[:N_BAND_SAMPLES]
        const      = np.array(fit["constant_term"])
        ref_params = np.array(fit["ref_params"])
        mterms     = np.array(fit["model_terms"])
        sample_re_abs = np.abs(np.real(np.array([
            (const + np.einsum("p,stp->st", s - ref_params, mterms))[0]
            for s in samples
        ])))
        band_lo = np.percentile(sample_re_abs, 25, axis=0)
        band_hi = np.percentile(sample_re_abs, 75, axis=0)

        mean_amp = float(fit["mean_amplitude"][0])
        print(f"  DW amplitude (posterior mean): {mean_amp:.4e}")

        # ── Plot ─────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(config.fig_width, config.fig_height), dpi=300)

        ax.semilogy(times, np.abs(np.real(h22)),
                    color="0.75", lw=0.7, label="Original")
        ax.semilogy(t_filt, np.abs(np.real(h_filt)),
                    color=config.colors[0], lw=0.9, label="Filtered")
        ax.semilogy(analysis_t, np.abs(np.real(model_complex)),
                    color=config.color_dw, lw=1.3, ls="--", label=r"DW fit")
        ax.fill_between(analysis_t, band_lo, band_hi,
                        color=config.color_dw, alpha=0.5, linewidth=1.3)

        ax.set_xlim(*PLOT_XLIM)
        ax.set_ylim(*PLOT_YLIM)
        ax.set_xlabel(r"$t\;[M]$")
        ax.set_ylabel(r"$|\mathrm{Re}\,\mathcal{N}^{22}|$")
        ax.legend(fontsize=7, frameon=False, loc="upper right")

        out_dir = "figs" if sxs_id == "0004" else "diagnostic_figs"
        os.makedirs(out_dir, exist_ok=True)
        fig.tight_layout()
        out_path = f"{out_dir}/4_rational_filter_fit_{sxs_id}.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
