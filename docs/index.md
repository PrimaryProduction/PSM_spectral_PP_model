# PSM spectral primary production model

A compact Python implementation of the **Platt–Sathyendranath model (PSM)** of
ocean primary production. It is spectrally resolved (400–700 nm) and resolves
depth and time of day. From two quantities that satellites measure,
**surface chlorophyll‑a** and **daily photosynthetically available radiation
(PAR)**, it computes:

- a vertical chlorophyll profile with a deep chlorophyll maximum,
- the partition of chlorophyll into pico-, nano- and microphytoplankton and the
  resulting phytoplankton absorption spectrum,
- the clear-sky (and cloud-corrected) spectral irradiance at the sea surface,
  split into direct and diffuse components,
- the spectral light field $E_d(z,\lambda,t)$ in the water column, and
- primary production $P(z,t)$, which can be integrated over depth and the day.

The model follows the formulation used for the global primary production
estimates of [Kulk et al. (2020)](references.md#kulk2020) and the long line of
work by Platt, Sathyendranath and co-workers that it builds on
([Platt & Sathyendranath 1988](references.md#platt1988);
[Sathyendranath & Platt 1988](references.md#sathyendranath1988);
[Sathyendranath et al. 1989](references.md#sathyendranath1989)). It was put
together as a teaching and research tool for the **CBIOMES primary production
workshop**, and is the spectral companion to the non-spectral practical in
[`PrimaryProduction/Non_spectral_PP_model`](https://github.com/PrimaryProduction/Non_spectral_PP_model).

![Schematic of the computation of primary production](img/psm_schematic.png){ .figure width="620" }

*Structure of the model. Surface biomass is extended to a profile; surface
irradiance is propagated to depth with a light-penetration model; a
photosynthesis–irradiance function converts light and biomass into primary
production at each depth and time.*

## Quick example

```python
import numpy as np
import pandas as pd

import absorption, chlorophyll, daylight, light_profile, scatter, surface_light

lat, par = 45.0, 25.0                      # degrees N, einstein m-2 d-1
chlz = chlorophyll.sinusodal_profile(1.5)  # mg m-3 on config.depths
ac, _ = absorption.calc_ac(chlz)           # phytoplankton absorption (m-1)
d_lambda = scatter.all()["d_lambda"]

Pb_m, alpha_b = 6.0, 0.02                  # photosynthesis parameters
alpha_spec = alpha_b * ac * 6022.0 / (2.77 * 36.0 * ac.mean(axis=1)[:, None])

pp = []
times = daylight.time_range("2020-05-06", lat, zenith_deg=80, steps=11)
for t in times:
    irs = surface_light.instant(t, lat, par)
    zen = np.rad2deg(daylight.zenith(t, lat))
    irz = light_profile.compute_profile(chlz, zen, irs["direct"], irs["diffuse"])
    i_alpha = np.sum(d_lambda * alpha_spec * irz["Iz"] / irz["mu_d"], axis=1)
    pp.append(chlz * Pb_m * (1 - np.exp(-i_alpha / Pb_m)))   # mg C m-3 h-1
```

The [tutorial](tutorial.md) goes through each step with figures, and the
[model description](theory/index.md) gives the equations.

## Where to go next

| If you want to… | Read |
|---|---|
| install the model and run the notebook | [Getting started](getting-started.md) |
| follow a worked example step by step | [Tutorial](tutorial.md) |
| understand the physics and biology | [Model description](theory/index.md) |
| look up a function's arguments and units | [API reference](api/index.md) |
| know the limitations of the current code | [Known issues](known-issues.md) |
| find the original papers | [References](references.md) |

## Citing

If you use this code, please cite the repository together with the papers
that describe the model, in particular [Kulk et al. (2020)](references.md#kulk2020)
and [Sathyendranath et al. (1989)](references.md#sathyendranath1989).
