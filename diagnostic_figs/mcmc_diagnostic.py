"""
MCMC diagnostic plots for 3_free_frequency.py output.

Figures produced (one PDF per t0):
  1. Trace plots: Re(omega) and kappa walker chains vs step, burn-in shaded.
  2. Corner plot: all 4 sampled parameters [A_real, A_imag, Re(omega), kappa]
     with DW reference values marked.

Samples are read from  mcmc/free_freq_{SXS_ID}.npy
Output goes to         diagnostic_figs/mcmc_diagnostics/
"""

import os
import sys

# Allow running from any directory — find plot_config in the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import corner

# ── Configuration ──────────────────────────────────────────────────────────────

SXS_IDS    = ["0004"]
# SXS_IDS = [f"{i:04d}" for i in range(1, 14)]

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(_PROJECT_ROOT, "mcmc")
OUTPUT_DIR  = os.path.join(_PROJECT_ROOT, "diagnostic_figs", "mcmc_diagnostics")


# ── Load samples ───────────────────────────────────────────────────────────────

def load_data(path):
    return np.load(path, allow_pickle=True).item()


# ── Trace plots ────────────────────────────────────────────────────────────────

def plot_traces(data):
    meta    = data["metadata"]
    burn_in = int(meta["BURN_IN_FRAC"] * meta["NSTEPS"])
    omega_dw = meta["omega_dw_real"]
    kappa_dw = meta["kappa_dw"]

    if "full_chains" not in data or data["full_chains"] is None:
        print("  [skip] full_chains not in data — re-run 3_free_frequency.py to enable trace plots.")
        return

    for t0, full_chain, acc_frac in zip(
        data["t0_vals"],
        data["full_chains"],
        data["acceptance_fracs"],
    ):
        nsteps, nwalkers, _ = full_chain.shape
        steps = np.arange(nsteps)

        fig, axes = plt.subplots(4, 1, figsize=(6, 8), sharex=True, dpi=200)

        for ax, param_idx, ref, ylabel in [
            (axes[0], 0, None,     r"$A_\mathrm{re}$"),
            (axes[1], 1, None,     r"$A_\mathrm{im}$"),
            (axes[2], 2, omega_dw, r"$\mathrm{Re}(\omega)\;[M_f^{-1}]$"),
            (axes[3], 3, kappa_dw, r"$\kappa\;[M_f^{-1}]$"),
        ]:
            chains = full_chain[:, :, param_idx]
            if param_idx == 3:
                chains = -chains  # Im(omega) → kappa
            ax.plot(steps, chains, color="0.6", lw=0.3, alpha=0.5, rasterized=True)
            ax.plot(steps[burn_in:], np.median(chains[burn_in:], axis=1),
                    color="k", lw=1.0, zorder=5)
            ax.axvspan(0, burn_in, color="0.85", zorder=0, label=f"burn-in ({burn_in})")
            if ref is not None:
                ax.axhline(ref, color="0.5", lw=0.9, ls="--", zorder=6,
                           label=r"$\omega_H$")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=6, frameon=False, loc="upper right")

        axes[3].set_xlabel("Step")
        axes[0].set_title(
            f"SXS:BBH:{meta['SXS_ID']}  $t_0={t0:+.0f}\\,M$  "
            f"acc={np.mean(acc_frac):.2f}  nwalkers={nwalkers}  nsteps={nsteps}",
            fontsize=8,
        )
        plt.tight_layout()

        out = os.path.join(OUTPUT_DIR, f"trace_{meta['SXS_ID']}_t0{int(t0):+d}.pdf")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")


# ── Corner plots ───────────────────────────────────────────────────────────────

def plot_corners(data):
    meta     = data["metadata"]
    omega_dw = meta["omega_dw_real"]
    kappa_dw = meta["kappa_dw"]

    have_full = "full_chains" in data and data["full_chains"] is not None
    burn_in   = int(meta["BURN_IN_FRAC"] * meta["NSTEPS"]) if have_full else 0

    if have_full:
        entries = zip(data["t0_vals"], data["full_chains"])
    else:
        entries = zip(data["t0_vals"], data["omega_r_samples"], data["kappa_samples"])

    for entry in entries:
        if have_full:
            t0, full_chain = entry
            flat = full_chain[burn_in:].reshape(-1, 4).copy()
            flat[:, 3] = -flat[:, 3]
            truths = [None, None, omega_dw, kappa_dw]
            labels = [r"$A_\mathrm{re}$", r"$A_\mathrm{im}$",
                      r"$\mathrm{Re}(\omega)$", r"$\kappa$"]
        else:
            t0, omega_r_samp, kappa_samp = entry
            flat   = np.column_stack([omega_r_samp, kappa_samp])
            truths = [omega_dw, kappa_dw]
            labels = [r"$\mathrm{Re}(\omega)$", r"$\kappa$"]

        fig = corner.corner(
            flat,
            labels=labels,
            truths=truths,
            truth_color="0.5",
            show_titles=True,
            title_kwargs={"fontsize": 7},
            label_kwargs={"fontsize": 8},
            labelpad=0.05,
            quantiles=[0.16, 0.5, 0.84],
            smooth=1.0,
            color="steelblue",
        )
        fig.suptitle(f"SXS:BBH:{meta['SXS_ID']}  $t_0={t0:+.0f}\\,M$", fontsize=9, y=1.01)

        out = os.path.join(OUTPUT_DIR, f"corner_{meta['SXS_ID']}_t0{int(t0):+d}.pdf")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for sxs_id in SXS_IDS:
        samples_file = os.path.join(SAMPLES_DIR, f"free_freq_{sxs_id}_0_only.npy")
        if not os.path.exists(samples_file):
            print(f"SKIP {sxs_id} — {samples_file} not found (run 3_free_frequency.py first)")
            continue

        print(f"\nLoading {samples_file}...")
        data = load_data(samples_file)
        meta = data["metadata"]
        print(f"  SXS:BBH:{meta['SXS_ID']}  t0_vals={data['t0_vals'].tolist()}")
        print(f"  nsteps={meta['NSTEPS']}  nwalkers={meta['NWALKERS']}  "
              f"burn_in_frac={meta['BURN_IN_FRAC']}")

        print("  Trace plots...")
        plot_traces(data)

        print("  Corner plots...")
        plot_corners(data)

        print(f"  Done — diagnostics in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
