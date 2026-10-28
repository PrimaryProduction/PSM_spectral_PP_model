# Tutorial: production at one location

This tutorial follows `light_field.ipynb` and computes daily primary
production for a single satellite pixel at **45° N on 6 May**, with surface
chlorophyll of **1.5 mg m⁻³** and daily PAR of **25 einstein m⁻² d⁻¹**. Run the
code blocks in order from the repository root, in a Python session or a
notebook. Figures for every step are on the
[model description](theory/index.md) pages.

## 1. Imports and grids

```python
import numpy as np
import pandas as pd

import absorption
import chlorophyll
import config
import daylight
import light_profile
import scatter
import size_fraction
import surface_light

wavelengths = config.wavelengths   # 400-700 nm, 5 nm steps (61 bands)
depths = config.depths             # 0-249.5 m, 0.5 m steps (500 levels)
```

The default grids are set in [`config`](api/config.md). Every function also
accepts its own `wavelengths=` / `depths=` arguments.

## 2. Chlorophyll profile

```python
chl_surface = 1.5
chlz = chlorophyll.sinusodal_profile(chl_surface, rho=0.1)  # weak DCM at 60 m
print(chlz.shape, chlz[0], chlz.max())
```

`rho` controls how pronounced the deep maximum is (0 gives a uniform
profile) and `z_m` and `sigma` set its depth and width. See
[Chlorophyll profile](theory/chlorophyll.md).

## 3. Size classes and phytoplankton absorption

```python
pico, nano, micro = size_fraction.brewin(chlz)
ac, ac_mean = absorption.calc_ac(chlz, wavelengths=wavelengths)
print(ac.shape)        # (depth, wavelength) = (500, 61), in m-1
```

## 4. Water and CDOM properties

```python
sc = scatter.all(wavelengths=wavelengths)
print(sorted(sc))      # ['aw', 'ay', 'bbr', 'bw', 'd_lambda']
```

## 5. Sun and surface light

```python
lat, par = 45.0, 25.0
day = pd.Timestamp("2020-05-06")

print("Sunrise:   ", daylight.sunrise(day, lat))
print("Day length:", daylight.daylength(day, lat), "h")

noon = day + pd.Timedelta("12:00:00")
zen_noon = np.rad2deg(daylight.zenith(noon, lat))
irs = surface_light.instant(noon, lat, par, wavelengths=wavelengths)
print("Noon zenith angle: %.1f deg" % zen_noon)
print("Below-surface PAR at noon: %.2f einstein m-2 h-1" % irs["i_zero"])
```

`irs["direct"]` and `irs["diffuse"]` are spectra just below the surface in
einstein m⁻² h⁻¹ nm⁻¹. Add `cloud=50` to see how cloud cover shifts light
from the direct to the diffuse component. The total stays constrained by PAR.

## 6. Underwater light field

```python
irz = light_profile.compute_profile(chlz, zen_noon, irs["direct"], irs["diffuse"],
                                    wavelengths=wavelengths)
print("Euphotic depth: %.1f m" % irz["euphotic_depth"])
```

`irz["Iz"]` holds $E_d(z,\lambda)$ with shape (500, 61), `irz["K"]` the
attenuation coefficients and `irz["mu_d"]` the mean cosine.

!!! note
    Pass the actual solar zenith angle at each time step. It enters through
    the mean cosine of the refracted direct beam.

## 7. Production at noon

```python
Pb_m = 6.0      # assimilation number, mg C (mg Chl)-1 h-1
Al_b = 0.02     # initial slope, mg C (mg Chl)-1 h-1 (W m-2)-1

alpha_spec = Al_b * ac * 6022.0 / (2.77 * 36.0 * ac.mean(axis=1)[:, None])
I_alpha = np.sum(sc["d_lambda"] * alpha_spec * irz["Iz"] / irz["mu_d"], axis=1)
pp_noon = chlz * Pb_m * (1 - np.exp(-I_alpha / Pb_m))     # mg C m-3 h-1
print("Surface production at noon: %.2f mg C m-3 h-1" % pp_noon[0])
```

See [Primary production](theory/primary-production.md) for the derivation of
the spectral initial slope and the unit-conversion factor.

## 8. Over the day

```python
times = daylight.time_range(day, lat, zenith_deg=80, steps=11)
pp = []
for t in times:
    irs = surface_light.instant(t, lat, par, wavelengths=wavelengths)
    zen = np.rad2deg(daylight.zenith(t, lat))
    irz = light_profile.compute_profile(chlz, zen, irs["direct"], irs["diffuse"],
                                        wavelengths=wavelengths)
    I_alpha = np.sum(sc["d_lambda"] * alpha_spec * irz["Iz"] / irz["mu_d"], axis=1)
    pp.append(chlz * Pb_m * (1 - np.exp(-I_alpha / Pb_m)))
pp = np.array(pp)                                  # (time, depth), mg C m-3 h-1

hours = (times - times[0]).total_seconds().to_numpy() / 3600
daily_profile = np.trapezoid(pp, hours, axis=0)    # mg C m-3 d-1
daily_total = np.trapezoid(daily_profile, depths)  # mg C m-2 d-1
print("Daily water-column production: %.0f mg C m-2 d-1" % daily_total)
```

The time steps run from the moment the sun reaches 80° zenith in the morning
to the matching time in the afternoon, so the low-light periods near sunrise
and sunset are left out. Use more `steps` for a more accurate integral.

!!! tip "Things to try"
    - Move the chlorophyll maximum: `chlorophyll.sinusodal_profile(0.3, rho=0.8, z_m=40)`.
    - Change latitude and date and watch day length and zenith angle change.
    - Vary `Pb_m` and `Al_b` within their natural ranges.
    - Add cloud cover with `surface_light.instant(..., cloud=70)`.
    - Use a coarser spectral grid, e.g. `wavelengths = np.arange(400, 701, 25)`
      (pass it to every function).
