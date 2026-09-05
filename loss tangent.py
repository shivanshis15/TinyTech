"""
Title: Loss Tangent Visualization Suite
Description: Computes and displays clean, publication-ready figure for Loss Tangent.
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

    eps1_ag = n_ag**2 - k_ag**2
    eps1_au = n_au**2 - k_au**2
    eps2_ag = 2 * n_ag * k_ag
    eps2_au = 2 * n_au * k_au

    tan_ag = eps2_ag / np.abs(eps1_ag)
    tan_au = eps2_au / np.abs(eps1_au)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(wl_ag, tan_ag, label='Silver (Ag)', color='silver', linewidth=2.5)
    ax.plot(wl_au, tan_au, label='Gold (Au)', color='goldenrod', linewidth=2.5)
    
    max_idx_au = np.argmax(tan_au)
    ax.scatter(wl_au[max_idx_au], tan_au[max_idx_au], color='#e74c3c', s=35, zorder=5)
    ax.annotate(f'Au Peak: {tan_au[max_idx_au]:.1f}', (wl_au[max_idx_au], tan_au[max_idx_au]), 
                textcoords="offset points", xytext=(20, 15), ha='left', color='#c0392b', fontsize=8.5, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bdc3c7", alpha=0.85))

    max_idx_ag = np.argmax(tan_ag)
    ax.scatter(wl_ag[max_idx_ag], tan_ag[max_idx_ag], color='#e74c3c', s=35, zorder=5)
    ax.annotate(f'Ag Peak: {tan_ag[max_idx_ag]:.1f}', (wl_ag[max_idx_ag], tan_ag[max_idx_ag]), 
                textcoords="offset points", xytext=(20, 40), ha='left', color='#c0392b', fontsize=8.5, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bdc3c7", alpha=0.85))

    ax.set_title(r'Loss Tangent ($\tan\delta = \varepsilon_2 / \varepsilon_1$)', fontsize=12, fontweight='bold')
    ax.set_xlabel(r'Wavelength ($\mu$m)', fontsize=10)
    ax.set_ylabel(r'Loss Tangent ($\tan\delta$)', fontsize=10)
    ax.set_ylim(-20, 400)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('loss_tangent.png', dpi=300)
    plt.show()
    print("Loss Tangent figure saved successfully!")

if __name__ == '__main__':
    main()