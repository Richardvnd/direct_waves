"""
MCMC free-frequency fits for SXS:BBH:0004.

At each selected start time, runs bgp.free_frequency_fit (emcee) with:
  - fixed linear-model modes taken from the 2_DW_content "with DW" output
    (same per-t0 nearest-neighbour lookup as 3a_free_frequency_plot.py)

Posterior samples (post burn-in, flat chain) are saved to
  mcmc/free_freq_<SXS_ID>.npy
"""

import os
import json
import numpy as np

import bgp_qnm_fits as bgp

# ── Configuration ──────────────────────────────────────────────────────────────

SXS_IDS     = ["0004"]
# SXS_IDS = [f"{i:04d}" for i in range(1, 14)]

DATA_TYPE   = "news"
NOISE_MODEL = "GP"       # "GP" or "WN"

TARGET_MODE     = (2, 2, "DW")
SPHERICAL_MODES = [(2, 2), (3, 2)]

T_MCMC  = 100.0
T0_MCMC = [0.0]

NSTEPS       = 10000
NWALKERS     = 32
BURN_IN_FRAC = 0.3

INCLUDE_CHIF = False
INCLUDE_MF   = False

# Search range — must match 3a_free_frequency_plot.py so the violin spans the
# same region as the Nelder-Mead scatter.
OMEGA_R_RANGE = (0.4, 0.85)
KAPPA_RANGE   = (0.01, 0.3)

# Multiplicative range for the amplitude prior (applied around the lstsq guess)
A_PRIOR_RANGE = (-100.0, 100.0)

OUTPUT_DIR  = "mcmc"


# ── Content loader ─────────────────────────────────────────────────────────────

def load_content_modes(path, target_mode):
    """
    Read per-t0 mode lists from 2_DW_content output. Returns only valid QNM
    4-tuples (l,m,n,p), dropping 2-tuple BGP entries, DW 3-tuples, and the
    target mode itself.
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
            if len(t) != 4:
                continue
            if t == target_tuple:
                continue
            fixed.append(t)
        content_modes.append(fixed)
    return content_t0s, content_modes


# ── MCMC fits ─────────────────────────────────────────────────────────────────

def run_fits(sxs_id):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading SXS:BBH:{sxs_id} ({DATA_TYPE}, Lev5, R2)...", flush=True)
    sim  = bgp.SXS_CCE(sxs_id, type=DATA_TYPE, lev="Lev5", radius="R2")
    Mf   = float(sim.Mf)
    chif = float(sim.chif_mag)
    print(f"  Mf={Mf:.6f}  chif={chif:.6f}")

    if NOISE_MODEL == "GP":
        tuned_params = bgp.get_tuned_param_dict("GP", data_type=DATA_TYPE)[sxs_id]
        kernel = bgp.kernel_GP
    else:
        tuned_params = bgp.get_tuned_param_dict("WN", data_type=DATA_TYPE)[sxs_id]
        kernel = bgp.kernel_WN

    content_file = f"mode_content_files/dw_{sxs_id}_with_dw.json"
    content_t0s, content_modes_list = load_content_modes(content_file, TARGET_MODE)
    print(f"Loaded content from {content_file} ({len(content_t0s)} entries)")

    omega_dw = bgp.qnm.omega_list([TARGET_MODE], chif, Mf)[0]
    kappa_dw = -omega_dw.imag   # positive surface gravity
    print(f"DW:  Re(omega) = {omega_dw.real:.4f}  kappa = {kappa_dw:.4f}")

    # Convert absolute prior bounds to multiplicative factors expected by
    # free_frequency_fit.  The class constructs:
    #   f_prior   = (omega.real * f_range[0],   omega.real * f_range[1])
    #   tau_prior = (omega.imag * tau_range[1],  omega.imag * tau_range[0])  # swapped because Im(omega)<0
    f_prior_range   = (OMEGA_R_RANGE[0] / omega_dw.real, OMEGA_R_RANGE[1] / omega_dw.real)
    tau_prior_range = (KAPPA_RANGE[0] / kappa_dw, KAPPA_RANGE[1] / kappa_dw)

    burn_in = int(BURN_IN_FRAC * NSTEPS)
    print(f"\nMCMC: nsteps={NSTEPS}  nwalkers={NWALKERS}  burn_in={burn_in}")
    print(f"Prior Re(omega): {OMEGA_R_RANGE}   kappa: {KAPPA_RANGE}")

    omega_r_samples_list   = []
    kappa_samples_list     = []
    A_re_samples_list      = []
    A_im_samples_list      = []
    amp_samples_list       = []
    full_chains_list       = []   # (nsteps, nwalkers, 4) per t0 — for diagnostics
    acceptance_fracs_list  = []   # (nwalkers,) per t0
    t0_success             = []

    for t0 in T0_MCMC:
        idx         = int(np.argmin(np.abs(content_t0s - t0)))
        fixed_modes = content_modes_list[idx]
        print(f"\n── t0 = {t0:+.1f} M  (content idx={idx}, n_fixed={len(fixed_modes)}) ──",
              flush=True)

        if len(fixed_modes) == 0:
            print("  No fixed modes — skipping.")
            continue

        fit = bgp.free_frequency_fit(
            sim.times,
            sim.h,
            fixed_modes,
            Mf,
            chif,
            tuned_params,
            kernel,
            t0=t0,
            target_mode=TARGET_MODE,
            A_real_prior_range=A_PRIOR_RANGE,
            A_imag_prior_range=A_PRIOR_RANGE,
            f_prior_range=f_prior_range,
            tau_prior_range=tau_prior_range,
            nsteps=NSTEPS,
            nwalkers=NWALKERS,
            T=T_MCMC,
            spherical_modes=SPHERICAL_MODES,
            include_chif=INCLUDE_CHIF,
            include_Mf=INCLUDE_MF,
            data_type=DATA_TYPE,
        )

        # Structured chain: (nsteps, nwalkers, 4) — param order [A_real, A_imag, f, tau]
        full_chain = fit.sampler.get_chain()
        acc_frac   = fit.sampler.acceptance_fraction   # shape (nwalkers,)

        # Post-burnin flat chain for downstream use
        flat_chain      = full_chain[burn_in:].reshape(-1, 4)
        omega_r_samples = flat_chain[:, 2]   # Re(omega)
        kappa_samples   = -flat_chain[:, 3]  # kappa = -Im(omega)
        A_re_samples    = flat_chain[:, 0]
        A_im_samples    = flat_chain[:, 1]
        amp_samples     = np.abs(flat_chain[:, 0] + 1j * flat_chain[:, 1])

        acc = float(np.mean(acc_frac))
        print(f"  Acceptance fraction: {acc:.3f}  |  posterior samples: {len(omega_r_samples)}")
        print(f"  Re(omega): median={np.median(omega_r_samples):.4f}  "
              f"68%=({np.percentile(omega_r_samples,16):.4f}, {np.percentile(omega_r_samples,84):.4f})")
        print(f"  kappa:     median={np.median(kappa_samples):.4f}  "
              f"68%=({np.percentile(kappa_samples,16):.4f}, {np.percentile(kappa_samples,84):.4f})")
        print(f"  |A|:       median={np.median(amp_samples):.4e}  "
              f"68%=({np.percentile(amp_samples,16):.4e}, {np.percentile(amp_samples,84):.4e})")

        omega_r_samples_list.append(omega_r_samples)
        kappa_samples_list.append(kappa_samples)
        A_re_samples_list.append(A_re_samples)
        A_im_samples_list.append(A_im_samples)
        amp_samples_list.append(amp_samples)
        full_chains_list.append(full_chain)
        acceptance_fracs_list.append(acc_frac)
        t0_success.append(t0)

    output = {
        "t0_vals":            np.array(t0_success),
        "omega_r_samples":    omega_r_samples_list,   # flat post-burnin, shape (n_samp,) per t0
        "kappa_samples":      kappa_samples_list,
        "A_re_samples":       A_re_samples_list,
        "A_im_samples":       A_im_samples_list,
        "amp_samples":        amp_samples_list,        # |A| = sqrt(A_re^2 + A_im^2)
        "full_chains":        full_chains_list,        # (nsteps, nwalkers, 4) per t0
        "acceptance_fracs":   acceptance_fracs_list,   # (nwalkers,) per t0
        "metadata": {
            "SXS_ID":          sxs_id,
            "DATA_TYPE":       DATA_TYPE,
            "TARGET_MODE":     list(TARGET_MODE),
            "SPHERICAL_MODES": [list(m) for m in SPHERICAL_MODES],
            "T_MCMC":          T_MCMC,
            "NSTEPS":          NSTEPS,
            "NWALKERS":        NWALKERS,
            "BURN_IN_FRAC":    BURN_IN_FRAC,
            "omega_dw_real":   float(omega_dw.real),
            "kappa_dw":        float(kappa_dw),
        },
    }
    output_file = f"{OUTPUT_DIR}/free_freq_{sxs_id}_0_only.npy"
    np.save(output_file, output, allow_pickle=True)
    print(f"\nSaved {output_file}")
    return output_file


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    for sxs_id in SXS_IDS:
        print(f"\n{'='*60}", flush=True)
        print(f"SXS:BBH:{sxs_id}", flush=True)
        print(f"{'='*60}", flush=True)
        run_fits(sxs_id)


if __name__ == "__main__":
    main()
