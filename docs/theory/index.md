# Model description

Primary production, the fixation of inorganic carbon into organic matter by
photosynthesis, depends on how much phytoplankton there is, how much light
reaches it, and how well it uses that light. The Platt–Sathyendranath approach
writes this as

$$
P(z,t) = B(z)\,P^B\big(I(z,t)\big),
$$

where $B$ is the chlorophyll-a concentration (a proxy for phytoplankton
biomass), $P^B$ the chlorophyll-normalised production rate and $I$ the light
available for photosynthesis
([Platt & Sathyendranath 1988](../references.md#platt1988)). The model on these
pages resolves every term spectrally and in depth:

``` mermaid
flowchart LR
    CHL["Satellite chlorophyll B₀"] --> PROF["Chlorophyll profile B(z)"]
    PROF --> PSC["Size classes<br/>pico / nano / micro"]
    PSC --> APH["Phytoplankton absorption a_ph(z,λ)"]
    PAR["Satellite daily PAR"] --> SURF["Surface spectral irradiance<br/>direct + diffuse"]
    SUN["Solar geometry<br/>lat, date, time"] --> SURF
    APH --> K["Attenuation K_d(z,λ)"]
    IOP["Water, CDOM, scattering"] --> K
    SURF --> LF["Light field E_d(z,λ,t)"]
    K --> LF
    LF --> PP["Production P(z,t)"]
    APH --> PP
    PI["P–I parameters P^B_m, α^B"] --> PP
    PP --> INT["Daily, depth-integrated production"]
```

The calculation has four parts, each described on its own page:

1. **[Chlorophyll profile and size classes](chlorophyll.md):** surface
   chlorophyll is extended to a depth profile with a Gaussian deep chlorophyll
   maximum, then split into three size classes that set the shape of the
   phytoplankton absorption spectrum.
2. **[Sunlight at the sea surface](sunlight.md):** solar geometry and a
   clear-sky spectral model give the spectral shape and the direct/diffuse
   split of surface irradiance. Satellite PAR fixes its magnitude.
3. **[Underwater light field](optics.md):** inherent optical properties of
   water, phytoplankton and CDOM give the diffuse attenuation coefficient, and
   light is propagated downward layer by layer.
4. **[Primary production](primary-production.md):** a spectral
   photosynthesis–irradiance function turns light and biomass into production,
   which is integrated over depth and the day.

All symbols and units are collected on the [Symbols & units](symbols.md) page.

## Assumptions in brief

- **Case-1 waters.** Optical properties other than those of pure water
  co-vary with chlorophyll. CDOM absorption is a fixed fraction of
  phytoplankton absorption at 440 nm, and particle scattering is a function of
  chlorophyll ([Morel 1988](../references.md#morel1988)).
- **Plane-parallel, horizontally homogeneous ocean and atmosphere.** No
  horizontal light transport.
- **Constant photosynthetic parameters** over depth and the day (they can be
  set per region or season, as in
  [Longhurst et al. 1995](../references.md#longhurst1995) and
  [Kulk et al. 2020](../references.md#kulk2020)).
- **Sinusoidal diurnal cycle.** Total surface irradiance follows a sine curve
  between sunrise and sunset whose integral matches the satellite daily PAR,
  so clouds are assumed constant over the day.
- **No photoinhibition** in the photosynthesis–irradiance function.
