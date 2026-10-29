"""Generate the figures used in the documentation.

Run from the repository root:

    pixi run -e docs python docs/scripts/make_figures.py

The figures are written to ``docs/img`` and are produced by the model code
itself, so they stay in sync with the implementation.
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import absorption  # noqa: E402
import chlorophyll  # noqa: E402
import config  # noqa: E402
import daylight  # noqa: E402
import light_profile  # noqa: E402
import scatter  # noqa: E402
import size_fraction  # noqa: E402
import surface_light  # noqa: E402

OUT = ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

# Categorical slots in fixed order, and a single-hue sequential ramp.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUES)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c9c8c3",
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "text.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e8e7e3",
    "grid.linewidth": 0.8,
    "lines.linewidth": 2,
    "font.size": 11,
    "legend.frameon": False,
    "figure.dpi": 110,
})

wavelengths = config.wavelengths
depths = config.depths


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("wrote", OUT / name)


# 1. Chlorophyll profiles -----------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 5))
cases = [(1.5, dict(rho=0.1), "B(0)=1.5, ρ=0.1"),
         (0.5, dict(z_m=80), "B(0)=0.5, zₘ=80 m"),
         (0.2, dict(), "B(0)=0.2, defaults")]
for (chl, kw, lab), col, ls in zip(cases, SERIES, ["-", "--", "-."]):
    ax.plot(chlorophyll.sinusodal_profile(chl, **kw), depths, color=col, ls=ls, label=lab)
ax.set_ylim(200, 0)
ax.set_xlabel("Chlorophyll-a (mg m⁻³)")
ax.set_ylabel("Depth (m)")
ax.set_title("Shifted-Gaussian chlorophyll profiles", loc="left", fontsize=12)
ax.legend(loc="lower right")
save(fig, "chl_profiles.png")

# 2. Size fractions ---------------------------------------------------------------
chl = np.logspace(-2, 1.3, 200)
p, n, m = size_fraction.brewin(chl)
fig, ax = plt.subplots(figsize=(6, 4.2))
for frac, lab, col, ls in zip([p, n, m], ["Pico", "Nano", "Micro"], SERIES, ["-", "--", "-."]):
    ax.plot(chl, frac / chl, color=col, ls=ls, label=lab)
ax.set_xscale("log")
ax.set_xlim(chl[0], chl[-1])
ax.set_ylim(0, 1)
ax.set_xlabel("Total chlorophyll-a (mg m⁻³)")
ax.set_ylabel("Fraction of total chlorophyll")
ax.set_title("Three-component size-class model (Brewin et al. 2010)", loc="left", fontsize=12)
ax.legend(loc="center left")
save(fig, "size_fractions.png")

# 3. Specific absorption spectra ----------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4.2))
for arr, lab, col, ls in zip([absorption.PICO, absorption.NANO, absorption.MICRO],
                             ["Pico", "Nano", "Micro"], SERIES, ["-", "--", "-."]):
    ax.plot(config.phyto_wl, arr, color=col, ls=ls, label=lab)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("a*ᵢ (m² mg Chl⁻¹)")
ax.set_title("Chlorophyll-specific absorption by size class", loc="left", fontsize=12)
ax.legend()
save(fig, "specific_absorption.png")

# 4. Absorption heat map -------------------------------------------------------------
chlz = chlorophyll.sinusodal_profile(1.5, rho=0.1)
ac, _ = absorption.calc_ac(chlz, wavelengths=wavelengths)
fig, ax = plt.subplots(figsize=(6, 4.5))
cs = ax.contourf(wavelengths, depths, ac, np.linspace(0, ac.max(), 25), cmap=SEQ)
ax.set_ylim(200, 0)
ax.grid(False)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Depth (m)")
ax.set_title("Phytoplankton absorption aₚₕ(z, λ)", loc="left", fontsize=12)
fig.colorbar(cs, ax=ax, label="m⁻¹")
save(fig, "absorption_depth.png")

# 5. Water and CDOM optical properties ----------------------------------------------
sc = scatter.all(wavelengths=wavelengths)
fig, ax = plt.subplots(figsize=(6.5, 5))
for key, lab, col, ls in zip(["aw", "bw", "bbr", "ay"],
                             ["a_w  pure water", "b_w  seawater scattering",
                              "b_br Raman", "a_y/a_y(440)  CDOM shape"],
                             SERIES, ["-", "--", "-.", ":"]):
    ax.plot(wavelengths, sc[key], color=col, ls=ls, label=lab)
ax.set_yscale("log")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("m⁻¹ (CDOM dimensionless)")
ax.set_title("Optical properties of water and CDOM", loc="left", fontsize=12)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=9)
save(fig, "water_iops.png")

# 6. Surface spectra through the day -------------------------------------------------
day = pd.Timestamp("2020-05-06")
lat, par = 45, 25
times = daylight.time_range(day, lat, zenith_deg=80, steps=11)[:6]
cols = [SEQ(x) for x in np.linspace(0.25, 1, len(times))]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for t, col in zip(times, cols):
    irs = surface_light.instant(t, lat, par, wavelengths=wavelengths)
    for ax, key in zip(axes, ["direct", "diffuse"]):
        ax.plot(wavelengths, irs[key], color=col, label=t.strftime("%H:%M"))
for ax, key in zip(axes, ["Direct", "Diffuse"]):
    ax.set_title(f"{key}, just below surface", loc="left", fontsize=12)
    ax.set_xlabel("Wavelength (nm)")
axes[0].set_ylabel("Einstein m⁻² h⁻¹ nm⁻¹")
axes[1].legend(title="Solar time", fontsize=9)
save(fig, "surface_spectra.png")

# 7. Underwater light field --------------------------------------------------------------
t = pd.Timestamp("2020-05-06 12:00")
irs = surface_light.instant(t, lat, par)
zen = np.rad2deg(daylight.zenith(t, lat))
irz = light_profile.compute_profile(chlz, zen, irs["direct"], irs["diffuse"])
fig, ax = plt.subplots(figsize=(6, 4.5))
logI = np.log10(np.maximum(irz["Iz"], 1e-12))
cs = ax.contourf(wavelengths, depths, logI, np.arange(-7, -1.75, 0.25), cmap=SEQ, extend="min")
ax.axhline(irz["euphotic_depth"], color=INK, ls=":", lw=1.5)
ax.annotate(f"1 % PAR depth ({irz['euphotic_depth']:.0f} m)", (695, irz["euphotic_depth"]),
            xytext=(0, 5), textcoords="offset points", color=INK, fontsize=10, ha="right",
            bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.85))
ax.set_ylim(100, 0)
ax.grid(False)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Depth (m)")
ax.set_title("Downwelling irradiance E_d(z, λ) at noon", loc="left", fontsize=12)
fig.colorbar(cs, ax=ax, label="log₁₀ E_d (Einstein m⁻² h⁻¹ nm⁻¹)")
save(fig, "light_field.png")

# 8. Primary production profiles ------------------------------------------------------------
Pb_m, Al_b = 6.0, 0.02
ac, _ = absorption.calc_ac(chlz)
Al_zl = Al_b * ac * 6022.0 / (2.77 * 36.0 * ac.mean(axis=1)[:, None])
times = daylight.time_range(day, lat, steps=13)
pp = []
for t in times:
    irs = surface_light.instant(t, lat, par)
    zen = np.rad2deg(daylight.zenith(t, lat))
    irz = light_profile.compute_profile(chlz, zen, irs["direct"], irs["diffuse"])
    I_alpha = np.sum(sc["d_lambda"] * Al_zl * irz["Iz"] / irz["mu_d"], axis=1)
    pp.append(chlz * Pb_m * (1 - np.exp(-I_alpha / Pb_m)))
pp = np.array(pp)
hours = (times - times[0]).total_seconds().values / 3600
daily = np.trapezoid(pp, hours, axis=0)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
cols = [SEQ(x) for x in np.linspace(0.25, 1, 6)]
for i, col in zip(range(1, 7), cols):          # skip sunrise (zero light)
    axes[0].plot(pp[i], depths, color=col, label=times[i].strftime("%H:%M"))
axes[0].set_title("Instantaneous P(z, t), morning to noon", loc="left", fontsize=12)
axes[0].set_xlabel("mg C m⁻³ h⁻¹")
axes[0].set_ylabel("Depth (m)")
axes[0].legend(title="Solar time", fontsize=9)
axes[1].plot(daily, depths, color=SERIES[0])
axes[1].set_title("Daily production P(z)", loc="left", fontsize=12)
axes[1].set_xlabel("mg C m⁻³ d⁻¹")
total = np.trapezoid(daily, depths)
axes[1].annotate(f"∫P dz = {total:.0f} mg C m⁻² d⁻¹", (0.35, 0.9),
                 xycoords="axes fraction", color=INK)
axes[0].set_ylim(120, 0)
save(fig, "pp_profiles.png")

# 9. P-I curve ---------------------------------------------------------------------------------
I = np.linspace(0, 1500, 300)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(I, Pb_m * (1 - np.exp(-0.02 * I / Pb_m)), color=SERIES[0])
ax.axhline(Pb_m, color=INK2, ls=":", lw=1)
ax.plot(I[:80], 0.02 * I[:80], color=SERIES[1], ls="--")
ax.annotate(r"$P^B_m$", (1400, Pb_m), xytext=(0, 5), textcoords="offset points", color=INK2)
ax.annotate(r"slope $\alpha^B$", (I[60], 0.02 * I[60]), xytext=(6, 0), textcoords="offset points", color=INK2)
ax.axvline(Pb_m / 0.02, color=INK2, ls=":", lw=1)
ax.annotate(r"$I_k = P^B_m/\alpha^B$", (Pb_m / 0.02, 0.5), xytext=(5, 0), textcoords="offset points", color=INK2)
ax.set_ylim(0, Pb_m * 1.15)
ax.set_xlabel("Irradiance I")
ax.set_ylabel(r"$P^B$ (mg C mg Chl⁻¹ h⁻¹)")
ax.set_title("Photosynthesis-irradiance curve (Platt et al. 1980)", loc="left", fontsize=12)
save(fig, "pi_curve.png")
