import os
import numpy as np
import matplotlib.pyplot as plt
import miepython as mp
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import matplotlib.patheffects as path_effects

# ============================================================
# SETTINGS
# ============================================================

WAVELENGTH_MIN_NM = 300
WAVELENGTH_MAX_NM = 800

wavelengths_nm = np.arange(
    WAVELENGTH_MIN_NM,
    WAVELENGTH_MAX_NM + 1,
    1
)

radii_nm = [20, 40, 60, 80, 100]
host_indices = [1.00, 1.33, 1.45, 1.52, 1.60]

# Updated to look inside the Dataset subfolder
materials = {
    "Au": os.path.join("Dataset", "au_nk.csv"),
    "Ag": os.path.join("Dataset", "ag_nk.csv")
}


# ============================================================
# LOAD JOHNSON & CHRISTY DATA
# ============================================================

def load_material_data(filename):
    df = pd.read_csv(filename)
    wl_nm = df["wl"].values * 1000
    n = df["n"].values
    k = df["k"].values
    m_complex = n + 1j * k

    interpolation = interp1d(
        wl_nm,
        m_complex,
        kind="linear",
        bounds_error=True
    )
    return interpolation, wl_nm


# ============================================================
# MIE CALCULATION
# ============================================================

def calculate_mie(radius_nm, n_medium, m_interpolated):
    qabs_values, qsca_values, qext_values = [], [], []

    for wavelength_nm in wavelengths_nm:
        m_particle = m_interpolated(wavelength_nm)
        m_relative = m_particle / n_medium
        x = 2 * np.pi * n_medium * radius_nm / wavelength_nm

        qext, qsca, qback, g = mp.single_sphere(m_relative, x, 0, 1)
        qabs = max(qext - qsca, 0)

        qabs_values.append(qabs)
        qsca_values.append(qsca)
        qext_values.append(qext)

    return np.array(qabs_values), np.array(qsca_values), np.array(qext_values)


# ============================================================
# FIND RESONANCE
# ============================================================

def find_resonance(material, qext):
    search_min = 450 if material == "Au" else 330
    search_max = 750 if material == "Au" else 600

    mask = (wavelengths_nm >= search_min) & (wavelengths_nm <= search_max)
    wl_region = wavelengths_nm[mask]
    qext_region = qext[mask]

    peaks, _ = find_peaks(
        qext_region,
        prominence=np.max(qext_region) * 0.01,
        distance=10
    )

    if len(peaks) > 0:
        strongest = peaks[np.argmax(qext_region[peaks])]
    else:
        strongest = np.argmax(qext_region)

    return int(wl_region[strongest]), qext_region[strongest]


# ============================================================
# LOAD DATA
# ============================================================

material_data = {}

for material, filename in materials.items():
    print(f"\nLoading {material} data from {filename}...")
    m_interpolated, available_wavelengths = load_material_data(filename)

    if wavelengths_nm.min() < available_wavelengths.min() or wavelengths_nm.max() > available_wavelengths.max():
        raise ValueError(f"{filename} wavelength coverage error.")

    material_data[material] = m_interpolated


# ============================================================
# HELPER FOR LEGIBLE ANNOTATIONS
# ============================================================

def annotate_peak(wl, val, offset=(5, 8)):
    txt = plt.annotate(
        f"{wl} nm",
        (wl, val),
        xytext=offset,
        textcoords="offset points",
        fontsize=9,
        color="darkred",
        fontweight="bold",
        zorder=6
    )
    # Add white outline so text stays readable over lines
    txt.set_path_effects([
        path_effects.withStroke(linewidth=3, foreground='white')
    ])


# ============================================================
# STUDY 1: EFFECT OF RADIUS
# ============================================================

radius_results = {}
radius_summary = []

for material in materials:
    print(f"\nCalculating radius dependence for {material}...")
    radius_results[material] = {}
    n_medium = 1.00

    for radius_nm in radii_nm:
        print(f"  Radius = {radius_nm} nm")
        qabs, qsca, qext = calculate_mie(radius_nm, n_medium, material_data[material])
        peak_wavelength, peak_qext = find_resonance(material, qext)

        radius_results[material][radius_nm] = {
            "qext": qext,
            "peak_wavelength": peak_wavelength,
            "peak_qext": peak_qext
        }

        radius_summary.append({
            "Material": material,
            "Radius (nm)": radius_nm,
            "Host n": n_medium,
            "Resonance (nm)": peak_wavelength,
            "Qext at resonance": round(peak_qext, 4)
        })


# PLOT 1: Au RADIUS (Original 10x6 Size)
plt.figure(figsize=(10, 6))
for radius_nm in radii_nm:
    data = radius_results["Au"][radius_nm]
    plt.plot(wavelengths_nm, data["qext"], label=f"{radius_nm} nm")
    plt.scatter(data["peak_wavelength"], data["peak_qext"], s=50, zorder=5)
    
    # Custom slight offset for tight clusters
    offset = (-25, 8) if radius_nm == 60 else (5, 8)
    annotate_peak(data["peak_wavelength"], data["peak_qext"], offset)

plt.title("Effect of Nanoparticle Radius on Au Optical Response")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Extinction Efficiency, $Q_{ext}$")
plt.xlim(WAVELENGTH_MIN_NM, WAVELENGTH_MAX_NM)
plt.grid(True, alpha=0.3)
plt.legend(title="Radius")
plt.tight_layout()
plt.show()


# PLOT 2: Ag RADIUS (Original 10x6 Size)
plt.figure(figsize=(10, 6))
for radius_nm in radii_nm:
    data = radius_results["Ag"][radius_nm]
    plt.plot(wavelengths_nm, data["qext"], label=f"{radius_nm} nm")
    plt.scatter(data["peak_wavelength"], data["peak_qext"], s=50, zorder=5)
    
    # Custom offset to separate overlapping Ag labels
    if radius_nm == 80:
        offset = (-35, 8)
    elif radius_nm == 100:
        offset = (-35, -12)
    else:
        offset = (5, 8)
        
    annotate_peak(data["peak_wavelength"], data["peak_qext"], offset)

plt.title("Effect of Nanoparticle Radius on Ag Optical Response")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Extinction Efficiency, $Q_{ext}$")
plt.xlim(WAVELENGTH_MIN_NM, WAVELENGTH_MAX_NM)
plt.grid(True, alpha=0.3)
plt.legend(title="Radius")
plt.tight_layout()
plt.show()


# ============================================================
# STUDY 2: EFFECT OF HOST MEDIUM
# ============================================================

medium_results = {}
medium_summary = []
FIXED_RADIUS_NM = 40

for material in materials:
    print(f"\nCalculating host-medium dependence for {material}...")
    medium_results[material] = {}

    for n_medium in host_indices:
        print(f"  Host refractive index = {n_medium}")
        qabs, qsca, qext = calculate_mie(FIXED_RADIUS_NM, n_medium, material_data[material])
        peak_wavelength, peak_qext = find_resonance(material, qext)

        medium_results[material][n_medium] = {
            "qext": qext,
            "peak_wavelength": peak_wavelength,
            "peak_qext": peak_qext
        }

        medium_summary.append({
            "Material": material,
            "Radius (nm)": FIXED_RADIUS_NM,
            "Host n": n_medium,
            "Resonance (nm)": peak_wavelength,
            "Qext at resonance": round(peak_qext, 4)
        })


# PLOT 3: Au HOST MEDIUM (Original 10x6 Size)
plt.figure(figsize=(10, 6))
for n_medium in host_indices:
    data = medium_results["Au"][n_medium]
    plt.plot(wavelengths_nm, data["qext"], label=f"n = {n_medium}")
    plt.scatter(data["peak_wavelength"], data["peak_qext"], s=50, zorder=5)
    annotate_peak(data["peak_wavelength"], data["peak_qext"], (5, 8))

plt.title(f"Effect of Host Medium on Au Optical Response (r = {FIXED_RADIUS_NM} nm)")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Extinction Efficiency, $Q_{ext}$")
plt.xlim(WAVELENGTH_MIN_NM, WAVELENGTH_MAX_NM)
plt.grid(True, alpha=0.3)
plt.legend(title="Host refractive index")
plt.tight_layout()
plt.show()


# PLOT 4: Ag HOST MEDIUM (Original 10x6 Size)
plt.figure(figsize=(10, 6))
for n_medium in host_indices:
    data = medium_results["Ag"][n_medium]
    plt.plot(wavelengths_nm, data["qext"], label=f"n = {n_medium}")
    plt.scatter(data["peak_wavelength"], data["peak_qext"], s=50, zorder=5)
    
    offset = (-25, 10) if n_medium in [1.33, 1.45] else (5, 8)
    annotate_peak(data["peak_wavelength"], data["peak_qext"], offset)

plt.title(f"Effect of Host Medium on Ag Optical Response (r = {FIXED_RADIUS_NM} nm)")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Extinction Efficiency, $Q_{ext}$")
plt.xlim(WAVELENGTH_MIN_NM, WAVELENGTH_MAX_NM)
plt.grid(True, alpha=0.3)
plt.legend(title="Host refractive index")
plt.tight_layout()
plt.show()


# ============================================================
# PRINT & EXPORT SUMMARY TABLES
# ============================================================

radius_summary_df = pd.DataFrame(radius_summary)
medium_summary_df = pd.DataFrame(medium_summary)

print("\n" + "=" * 65)
print("STUDY 1: EFFECT OF NANOPARTICLE RADIUS")
print("=" * 65)
print(radius_summary_df.to_string(index=False))

print("\n" + "=" * 65)
print("STUDY 2: EFFECT OF HOST-MEDIUM REFRACTIVE INDEX")
print("=" * 65)
print(medium_summary_df.to_string(index=False))

radius_summary_df.to_csv("Study1_Radius_Effect.csv", index=False)
medium_summary_df.to_csv("Study2_Host_Medium_Effect.csv", index=False)