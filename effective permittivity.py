"""
Title: Effective Permittivity Visualization Suite
Description: Computes and displays clean, publication-ready figure for Effective Permittivity at f = 0.05.
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
    print("Loading optical datasets for Gold and Silver...")
    try:
        ag_df = load_optical_data('ag_nk.csv')
        au_df = load_optical_data('au_nk.csv')
    except FileNotFoundError as e:
        print(f"Error loading CSV files: {e}. Please ensure ag_nk.csv and au_nk.csv are in the directory.")
        return

    wl_ag, n_ag, k_ag = ag_df['wl'].values, ag_df['n'].values, ag_df['k'].values
    wl_au, n_au, k_au = au_df['wl'].values, au_df['n'].values, au_df['k'].values

    n_m = 1.33
    eps_m = n_m**2
    f_fixed = 0.05

    eps1_ag = n_ag**2 - k_ag**2
    eps1_au = n_au**2 - k_au**2
    eps2_ag = 2 * n_ag * k_ag
    eps2_au = 2 * n_au * k_au

    eps_s_ag = eps1_ag + 1j * eps2_ag
    eps_s_au = eps1_au + 1j * eps2_au

    eps_eff_ag = eps_m * (eps_s_ag + 2*eps_m + 2*f_fixed*(eps_s_ag - eps_m)) / (eps_s_ag + 2*eps_m - f_fixed*(eps_s_ag - eps_m))
    eps_eff_au = eps_m * (eps_s_au + 2*eps_m + 2*f_fixed*(eps_s_au - eps_m)) / (eps_s_au + 2*eps_m - f_fixed*(eps_s_au - eps_m))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(wl_ag, np.real(eps_eff_ag), label='Silver (Ag)', color='silver', linewidth=2.5)
    ax.plot(wl_au, np.real(eps_eff_au), label='Gold (Au)', color='goldenrod', linewidth=2.5)
    
    peak_idx_ag = np.argmax(np.real(eps_eff_ag))
    trough_idx_ag = np.argmin(np.real(eps_eff_ag))
    
    ax.scatter([wl_ag[peak_idx_ag], wl_ag[trough_idx_ag]], [np.real(eps_eff_ag)[peak_idx_ag], np.real(eps_eff_ag)[trough_idx_ag]], color='#e74c3c', s=35, zorder=5)
    ax.annotate(f'Ag Peak: {np.real(eps_eff_ag)[peak_idx_ag]:.1f}', (wl_ag[peak_idx_ag], np.real(eps_eff_ag)[peak_idx_ag]), 
                textcoords="offset points", xytext=(20, 20), ha='left', color='#c0392b', fontsize=8.5, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bdc3c7", alpha=0.85))
    ax.annotate(f'Ag Trough: {np.real(eps_eff_ag)[trough_idx_ag]:.1f}', (wl_ag[trough_idx_ag], np.real(eps_eff_ag)[trough_idx_ag]), 
                textcoords="offset points", xytext=(20, -30), ha='left', color='#c0392b', fontsize=8.5, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bdc3c7", alpha=0.85))

    peak_idx_au = np.argmax(np.real(eps_eff_au))
    ax.scatter(wl_au[peak_idx_au], np.real(eps_eff_au)[peak_idx_au], color='#e74c3c', s=35, zorder=5)
    ax.annotate(f'Au Peak: {np.real(eps_eff_au)[peak_idx_au]:.2f}', (wl_au[peak_idx_au], np.real(eps_eff_au)[peak_idx_au]), 
                textcoords="offset points", xytext=(20, 10), ha='left', color='#c0392b', fontsize=8.5, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bdc3c7", alpha=0.85))

    ax.set_title(r'Effective Permittivity Real Part ($\varepsilon_{\text{eff}}$, $f = 0.05$)', fontsize=12, fontweight='bold')
    ax.set_xlabel(r'Wavelength ($\mu$m)', fontsize=10)
    ax.set_ylabel(r'Re($\varepsilon_{\text{eff}}$)', fontsize=10)
    ax.set_ylim(-4.5, 7.5)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('effective_permittivity.png', dpi=300)
    plt.show()
    print("Effective Permittivity figure saved successfully!")

if __name__ == '__main__':
    main()