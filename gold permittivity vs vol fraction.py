"""
Title: Gold Volume Fraction Permittivity Visualization Suite
Description: Computes and displays clean, publication-ready figure for Gold permittivity vs volume fraction.
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
    print("Loading optical datasets for Gold...")
    try:
        au_df = load_optical_data('au_nk.csv')
    except FileNotFoundError as e:
        print(f"Error loading CSV files: {e}. Please ensure au_nk.csv is in the directory.")
        return

    wl_au, n_au, k_au = au_df['wl'].values, au_df['n'].values, au_df['k'].values

    n_m = 1.33
    eps_m = n_m**2
    eps1_au = n_au**2 - k_au**2
    eps2_au = 2 * n_au * k_au
    eps_s_au = eps1_au + 1j * eps2_au

    fig, ax = plt.subplots(figsize=(9, 6))
    au_styles = [
        {'f': 0.02, 'color': '#2980b9', 'offset': (18, -20)},
        {'f': 0.05, 'color': '#e67e22', 'offset': (18, 5)},
        {'f': 0.10, 'color': '#27ae60', 'offset': (18, 30)}
    ]

    for style in au_styles:
        f_val = style['f']
        eps_eff_au_f = eps_m * (eps_s_au + 2*eps_m + 2*f_val*(eps_s_au - eps_m)) / (eps_s_au + 2*eps_m - f_val*(eps_s_au - eps_m))
        ax.plot(wl_au, np.real(eps_eff_au_f), label=f'Gold ($f = {f_val}$)', color=style['color'], linewidth=2.2)
        
        max_idx = np.argmax(np.real(eps_eff_au_f))
        peak_x = wl_au[max_idx]
        peak_y = np.real(eps_eff_au_f)[max_idx]

        ax.scatter(peak_x, peak_y, color='#e74c3c', s=35, zorder=5)
        ax.annotate(f'$f = {f_val}$ | Peak: {peak_y:.2f}', (peak_x, peak_y), 
                    textcoords="offset points", xytext=style['offset'], ha='left', color='#c0392b', fontsize=8.5, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bdc3c7", alpha=0.85))

    ax.set_title(r'Gold Permittivity vs Volume Fraction ($f$)', fontsize=12, fontweight='bold')
    ax.set_xlabel(r'Wavelength ($\mu$m)', fontsize=10)
    ax.set_ylabel(r'Re($\varepsilon_{\text{eff}}$)', fontsize=10)
    ax.set_ylim(1.3, 3.9)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('gold_volume_fraction_variation.png', dpi=300)
    plt.show()
    print("Gold Volume Fraction figure saved successfully!")

if __name__ == '__main__':
    main()