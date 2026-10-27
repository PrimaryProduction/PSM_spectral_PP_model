# API reference

These pages are generated from the docstrings in the source code
(numpydoc convention), so they always match the code. Modules are listed in
the order in which they enter the calculation.

| Module | Main entry points | Role |
|---|---|---|
| [`config`](config.md) | `wavelengths`, `depths`, `yellow_substance` | Default grids and constants |
| [`chlorophyll`](chlorophyll.md) | `sinusodal_profile` | Chlorophyll profile from surface value |
| [`size_fraction`](size_fraction.md) | `brewin` | Pico/nano/micro chlorophyll |
| [`absorption`](absorption.md) | `calc_ac` | Phytoplankton absorption spectrum |
| [`scatter`](scatter.md) | `all`, `calc_aw`, `calc_bw`, `calc_bbr`, `calc_ay` | Water, Raman and CDOM optical properties |
| [`daylight`](daylight.md) | `zenith`, `sunrise`, `daylength`, `time_range` | Solar geometry |
| [`irradiance`](irradiance.md) | `Irradiance`, `all` | Clear-sky spectral irradiance (Bird 1984) |
| [`surface_light`](surface_light.md) | `instant` | Below-surface spectral irradiance scaled to PAR |
| [`light_profile`](light_profile.md) | `compute_profile` | Attenuation and underwater light field |

## Conventions

- **Units.** Wavelengths are in nm, depths in m (positive downward), angles
  in degrees unless the name ends in `_rad`, chlorophyll in mg m⁻³, and
  irradiance in einstein m⁻² h⁻¹ nm⁻¹ after `surface_light` (W m⁻² µm⁻¹
  inside `irradiance`).
- **Array shapes.** Spectral quantities have wavelength as the *last* axis.
  Depth-resolved spectral quantities have shape `(n_depths, n_wavelengths)`.
- **Time.** Timestamps are interpreted as local solar time. Anything accepted
  by `pandas.to_datetime` works where a date is expected.
- **Grids.** Functions that take `wavelengths=None` or `depths=None` fall back
  to the grids in `config`. When you change the grid, pass the same grid to
  every function in the chain.
