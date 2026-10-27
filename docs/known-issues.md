# Known issues and implementation notes

This page lists behaviour in the current code that may surprise users. Items
marked **bug** are probably unintended. The documentation does not change any
of them, and the code reproduces the original implementation exactly.

## Possible bugs

### `daylight.thekaekara` extrapolates after day 23 — *bug, no effect on output*

The table lookup compares the loop index `i`, not `day_points[i]`, with the
day of year:

```python
for i in range(len(day_points)):
    idx = i
    if i >= jday:      # probably meant: day_points[i] >= jday
        break
```

For day of year ≥ 24 the last table segment (days 355–365) is extrapolated,
which gives for example 1410 W m⁻² on 20 January and 1380 W m⁻² on 21 June
(expected ~1310 W m⁻²). `surface_light.instant` uses the value only as a
uniform scale factor, and the spectrum is later rescaled to match satellite
PAR, so the factor **cancels exactly**: replacing it with a constant changes
the output by less than 10⁻¹⁵ (relative). It matters only if `thekaekara` is
used on its own.

### `daylight.sunrise(..., return_all=True)` returns a function

The second returned value is the module function `delta`, not the
declination value, because the local variable is never assigned. Only
`daylight.all` uses `return_all`, and it ignores that value.

### Aerosol transmittance uses an index-based band split

`Irradiance.aerosol_transmittance` applies the short-wavelength Ångström
parameters to the *first ten bands* (`[..., 0:10]`). On the original 24-band
grid this is the 500 nm split of Bird & Riordan (1986). On the default 5 nm
grid it falls between 445 and 450 nm, and on any other grid it moves again.
The affected bands get slightly different aerosol transmittance. Most of the
effect on the absolute level is removed by the scaling to PAR, but the
spectral shape is affected.

### `surface_light.instant` before sunrise or after sunset

The fraction of the day is computed with `(dtm - sunrise).seconds`. For
times before sunrise the timedelta is negative and `.seconds` wraps around
(e.g. −1 h gives 82 800 s), which gives meaningless values. Only call it for
times between sunrise and sunset, as `daylight.time_range` produces.

## Behaviour worth knowing

- **Two declination formulas.** `daylight.daylength` uses the Cooper (1969)
  declination of the Brock model, whereas `sunrise` and `zenith` use the
  Spencer (1971) series. Day length and `2 × (12 − sunrise)` therefore differ,
  by up to about 17 minutes at latitudes within ±60°, so the sinusoidal PAR cycle in
  `surface_light.instant` may not reach exactly zero at the computed sunset.
- **`chlorophyll.sinusodal_profile` is Gaussian.** The name is historical.
- **`absorption.calc_ac(size_class=...)`** is honoured only for scalar
  chlorophyll. Array input always returns the total. Selecting the
  nanoplankton prints `"nano"`.
- **Fractional gas weights.** `irradiance.coefficients` linearly interpolates
  the 0/1 weight arrays that select the bands affected by mixed-gas absorption.
  On the 5 nm grid, `dif_weights` is non-zero from 670 to 700 nm and
  `ozone_weights` at 695–700 nm.
- **Wavelength range.** `irradiance.interpolate_lut` raises `ValueError`
  outside 400–710 nm. The absorption and water tables are held constant
  outside 400–700 nm.
- **Top layer.** `light_profile.compute_profile` attenuates the first depth
  level by its full layer thickness (`numpy.gradient(depths)[0]`), and the
  euphotic depth is relative to PAR at that first level, not at 0⁻.
- **`config.yellow_substance`** is used only when `yellow_substance=None` is
  passed explicitly, since the default argument is `0.3`.
- **Unreachable code.** The ends of `light_profile.compute_profile` and
  `daylight.daylength` contain code after `return`, left over from the port
  of the Fortran code. It refers to undefined names and never runs.

## Notes on `light_field.ipynb`

- `light_profile.compute_profile` is called with a fixed zenith angle of 80°
  rather than the solar zenith angle at each time step. The
  [tutorial](tutorial.md) uses the actual angle.
- The diurnal loop computes surface light at 60° N (`surface_light.instant(dtm, 60, 25)`)
  while the rest of the notebook uses 45° N.
- The daily profile is formed with `PP.sum(axis=1)`, a sum over time steps
  without the time-step length. Multiply by the step in hours, or use
  `numpy.trapezoid` as in the tutorial, to get mg C m⁻³ d⁻¹.
