"""
Title: Silver Volume Fraction Permittivity Visualization Suite
Description: Computes and displays clean, publication-ready figure for Silver permittivity vs volume fraction.
Project: Tiny Tech: Tuning Optical Properties with Nanoparticles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_optical_data(filepath):
    """Loads CSV data containing wavelength (wl), refractive index (n), and extinction coefficient (k)."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df

def main():
    print("Loading optical datasets for Silver...")
    try:
        ag_df = load_optical_data('ag_nk.csv')
    except FileNotFoundError as e:
        print(f"Error loading CSV files: {e}. Please ensure ag_nk.csv is in the directory.")
        return

    wl_ag, n_ag, k_ag = ag_df['wl'].values, ag_df['n'].values, ag_df['k'].values

    n_m = 1.33
    eps_m = n_m**2
    eps1_ag = n_ag**2 - k_ag**2
    eps2_ag = 2 * n_ag * k_ag
    eps_s_ag = eps1_ag + 1j * eps2_ag

    fig, ax = plt.subplots(figsize=(9, 6))
    ag_styles = [
        {'f': 0.02, 'color': '#7f8c8d', 'offset': (18, -18)},
        {'f': 0.05, 'color': '#95a5a6', 'offset': (18, 12)},
        {'f': 0.10, 'color': '#2c3e50', 'offset': (18, 42)}
    ]

    for style in ag_styles:
        f_val = style['f']
        eps_eff_ag_f = eps_m * (eps_s_ag + 2*eps_m + 2*f_val*(eps_s_ag - eps_m)) / (eps_s_ag + 2*eps_m - f_val*(eps_s_ag - eps_m))
        ax.plot(wl_ag, np.real(eps_eff_ag_f), label=f'Silver ($f = {f_val}$)', color=style['color'], linewidth=2.2)
        
        max_idx_ag_f = np.argmax(np.real(eps_eff_ag_f))
        peak_x = wl_ag[max_idx_ag_f]
        peak_y = np.real(eps_eff_ag_f)[max_idx_ag_f]

        ax.scatter(peak_x, peak_y, color='#e74c3c', s=35, zorder=5)
        ax.annotate(f'$f = {f_val}$ | Peak: {peak_y:.1f}', (peak_x, peak_y), 
                    textcoords="offset points", xytext=style['offset'], ha='left', color='#c0392b', fontsize=8.5, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bdc3c7", alpha=0.85))

    ax.set_title(r'Silver Permittivity vs Volume Fraction ($f$)', fontsize=12, fontweight='bold')
    ax.set_xlabel(r'Wavelength ($\mu$m)', fontsize=10)
    ax.set_ylabel(r'Re($\varepsilon_{\text{eff}}$)', fontsize=10)
    ax.set_xlim(0.15, 0.75)
    ax.set_ylim(-4.0, 15.0)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('silver_volume_fraction_variation.png', dpi=300)
    plt.show()
    print("Silver Volume Fraction figure saved successfully!")

if __name__ == '__main__':
    main()