# Modeling Direct Waves in Binary Black Hole Ringdown

Code accompanying the paper:

> **[Modeling Direct Waves in Binary Black Hole Ringdown]**
> Richard Dyer, Adrian Ka-Wai Chung, and Christopher J. Moore
> arXiv:[XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

## Overview

This repository contains the code for identifying and characterising *direct wave* (DW) contributions in the gravitational wave ringdown signal from binary black hole mergers in numerical relativity simulations. 

Waveforms are taken from the [SXS Gravitational Wave Catalog](https://www.black-holes.org/waveforms) (simulations 0001–0013), using Cauchy-characteristic extraction (CCE) news data at Lev5 resolution.

## Dependencies

- [`bgp_qnm_fits`](https://github.com/BGP-QNM-FITS/bgp_qnm_fits) (v2.0)

## Data

Waveforms must be downloaded separately from the [SXS catalog](https://www.black-holes.org/waveforms). The scripts expect `bgp_qnm_fits` to handle loading via `bgp.SXS_CCE(sxs_id, type="news", lev="Lev5", radius="R2")`. See https://arxiv.org/pdf/2510.11783 for more details on processing the waveforms.

Generated intermediate data (mode content JSON files) are included in `mode_content_files/`. MCMC posterior samples are not tracked by git and must be regenerated with `3_free_frequency.py`.

## Other simulations

The main paper focuses on simulation 0004. The mode content for the other simulations can be found in [`diagnostic_figs/mode_content/`](diagnostic_figs/mode_content/).

## License

This project is licensed under the terms of the MIT License. See [LICENSE](LICENSE) for details.
