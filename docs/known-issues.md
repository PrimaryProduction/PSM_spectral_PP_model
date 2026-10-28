# Known issues and implementation notes

## Fixed

The following problems in the original code have been fixed. The *Effect*
column says how the fix changes results for typical daytime use on the
default grids.

| Problem | Fix | Effect on output |
|---|---|---|
| `daylight.thekaekara` compared the loop index instead of the tabulated day with the day of year, so after day 23 it extrapolated the last table segment (≈1380 instead of ≈1310 W m⁻² in June). | Compare `day_points[i]` with the day of year. | None on the model: the value only scales the clear-sky spectrum, which is later rescaled to PAR. `thekaekara` itself now matches the table. |
| `Irradiance.aerosol_transmittance` chose the Ångström regime by band *index* (first ten bands), which is 500 nm only on the original 24-band grid (450 nm on the 5 nm grid). | Choose by wavelength (< 500 nm), following Bird & Riordan (1986). | Clear-sky spectra change by ≤ 1.2 % at 450–495 nm. Underwater PAR changes by ≤ 0.1 %. The original 24-band grid gives identical results. |
| `surface_light.instant` returned **NaN** when the sun was exactly overhead (Fresnel reflectance 0/0). | Use the normal-incidence limit $((n-1)/(n+1))^2$. | Fixes NaNs at the subsolar point. No other change. |
| `surface_light.instant` returned NaN (or wrapped-around values) before sunrise, after sunset and in polar night. | Return zero irradiance whenever the sun is below the horizon or outside the sinusoidal PAR cycle; use `total_seconds()` instead of `.seconds`. | Daytime values change by ~10⁻⁵ (sub-second timing). Loops over a full 24 h day now work. |
| `daylight.time_range` crashed in polar night when `zenith_deg` was given. | Return an empty `DatetimeIndex` when the sun never rises or never reaches `zenith_deg`. `zenith_time` returns `NaT` in that case. | None in daylight. |
| `light_profile.compute_profile` crashed with a dark surface, and silently truncated output when the profile and depth grid had different lengths (e.g. `mld=`). | Handle zero surface light (zero light field, NaN euphotic depth) and raise a clear `ValueError` on mismatched lengths. | None for valid input. |
| `absorption.calc_ac(size_class=...)` ignored `size_class` for array input, matched on any letter (so `"microplankton"` selected the nano class) and printed `"nano"`. | Select by first letter for scalar and array input; no printing. | None for the default `"all"`. |
| `size_fraction.brewin` failed on Python lists. | Convert array-like input to an array. | None. |
| The cloud correction used plain sums over wavelength. | Weight by band width $\Delta\lambda$. | None on uniform grids; correct on non-uniform grids. |
| `daylight.sunrise(..., return_all=True)` returned the `delta` *function*. | Return the declination value. | None (`daylight.all` ignores it). |
| `config.yellow_substance` had no effect because `compute_profile` defaulted to `0.3`. | Default to `None` → `config.yellow_substance`. | None (same value). |
| Expected `arccos` warnings during polar day/night. | Silenced where the result is masked. | None. |
| Unreachable code after `return` in `compute_profile` and `daylength`. | Removed. | None. |

### Notebook (`light_field.ipynb`)

- `dtm` was reused as the loop variable, so after the irradiance plots the
  "noon" production cell actually used the last time step (≈18:10). Loops now
  use their own variable.
- `compute_profile` received a fixed 80° zenith angle. It now uses the solar
  zenith angle at each time.
- The diurnal loop used 60° N and a PAR of 25 instead of the `lat`/`par`
  variables.
- The daily profile summed time steps without their length. It is now
  integrated with the trapezoidal rule over hours (mg C m⁻³ d⁻¹), and the
  water-column total is printed.

## Behaviour worth knowing

- **Two declination formulas.** `daylight.daylength` uses the Cooper (1969)
  declination of the Brock model, whereas `sunrise` and `zenith` use the
  Spencer (1971) series. The day length and `2 × (12 − sunrise)` differ by up
  to about 17 minutes within ±60° latitude, so the sinusoidal PAR cycle in
  `surface_light.instant` does not end exactly at the geometric sunset. Using
  a consistent day length changes daily water-column production by at most
  0.12 % in tests at 60° S–60° N, so it is left as in the original code.
- **Day angle.** `daylight.delta` uses $\Gamma = 2\pi d/365$ where Spencer
  (1971) uses $d-1$, a one-day shift.
- **Polar day.** With 24 h daylight the sinusoidal cycle still drops to zero
  at midnight. This is a limitation of the sinusoidal assumption.
- **Spectral quadrature.** `scatter.all()["d_lambda"]` gives every band the
  width to the next band and repeats the last width, so 400–700 nm sums to
  305 nm. Because the surface spectrum is scaled to PAR with the same weights,
  the total light is consistent.
- **`chlorophyll.sinusodal_profile` is Gaussian.** The name is historical.
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
- **`zenith_time` with array latitudes** always returns decimal hours, using
  the sentinel values 0 and −1 h.
