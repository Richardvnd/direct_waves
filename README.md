# Modeling Direct Waves in Binary Black Hole Ringdowns

Code accompanying the paper:

> **[Modeling Direct Waves in Binary Black Hole Ringdowns](https://arxiv.org/abs/2606.25021)**
> Richard Dyer, Adrian Ka-Wai Chung, and Christopher J. Moore
> arXiv:[2606.25021](https://arxiv.org/abs/2606.25021)

```bibtex
@article{Dyer:2026:DirectWaves,
    author        = {Dyer, Richard and Chung, Adrian Ka-Wai and Moore, Christopher J.},
    title         = {Modeling Direct Waves in Binary Black Hole Ringdowns},
    year          = {2026},
    eprint        = {2606.25021},
    archivePrefix = {arXiv},
    primaryClass  = {gr-qc},
    doi           = {10.48550/arXiv.2606.25021},
}
```

## Overview

This repository contains the code for identifying and characterising direct wave (DW) contributions in the gravitational wave ringdown signal from binary black hole mergers in numerical relativity simulations. 

## Dependencies

- [`bgp_qnm_fits`](https://github.com/BGP-QNM-FITS/bgp_qnm_fits) (v2.0)
- [`qnmfits`](https://github.com/sxs-collaboration/qnmfits)

## Data

Waveforms must be downloaded separately from the [SXS catalog](https://www.black-holes.org/waveforms). We use the Cauchy-characteristic extracted (CCE) simulations 0001-0013, Bondi news, Lev 5 resolution. The scripts expect `bgp_qnm_fits` to handle loading via `bgp.SXS_CCE(sxs_id, type="news", lev="Lev5", radius="R2")`. See https://arxiv.org/pdf/2510.11783 for more details on processing the waveforms.

Generated intermediate data (mode content JSON files) are included in `mode_content_files/`. MCMC posterior samples are not tracked by git and must be regenerated with `3_free_frequency.py`. For 10,000 steps this takes ~40 minutes. 

## Other simulations

The main paper focuses on simulation 0004. The mode content for the other simulations can be found in [`diagnostic_figs/mode_content/`](diagnostic_figs/mode_content/).

## License

This project is licensed under the terms of the MIT License. See [LICENSE](LICENSE) for details.
