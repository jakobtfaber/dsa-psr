import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.lines import Line2D
from matplotlib.legend import Legend # Correct import

print("--- Phase 3: Analysis and Plotting ---")

# --- Load Data ---
output_dir = "cadence_simulation_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

config_filepath = os.path.join("simulation_data", "cadence_config_fine.npz")
results_filepath = os.path.join("simulation_data", "cadence_results_fine.npz")

if not (os.path.exists(config_filepath) and os.path.exists(results_filepath)):
    print("Error: Data files not found. Please run Phase 1 and 2 scripts first.")
else:
    print("Loading configuration and results data...")
    config = np.load(config_filepath)
    results = np.load(results_filepath, allow_pickle=True)

    # Extract data
    frequencies_ghz = config['frequencies_ghz']
    search_radii_multiples = config['search_radii_multiples']
    summary_results_array = results['summary_results']
    heatmap_data = results['heatmap_data'].item()
    max_extent = results['max_extent']
    
    # Correctly create the DataFrame
    summary_df = pd.DataFrame(list(summary_results_array))

    # --- Generate Detailed Heatmap Plots ---
    # This part remains the same
    print("Generating detailed heatmap plots for each scenario...")
    for freq_ghz in frequencies_ghz:
        for radius_multiple in search_radii_multiples:
            scenario_key = f"{freq_ghz:.2f}GHz_{radius_multiple:.2f}xFWHM"
            if f"visits_{scenario_key}" in heatmap_data:
                num_visits = heatmap_data[f"visits_{scenario_key}"]
                cumulative_sn = heatmap_data[f"sn_{scenario_key}"]
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
                fig.suptitle(f'Results for {freq_ghz:.2f} GHz with Search Radius = {radius_multiple:.2f}x FWHM', fontsize=16)

                im1 = ax1.imshow(num_visits, extent=[-max_extent, max_extent, -max_extent, max_extent], origin='lower', cmap='plasma', interpolation='nearest')
                fig.colorbar(im1, ax=ax1, label='Number of Visits')
                ax1.set_title('Overlapping Observations')
                ax1.set_xlabel('RA Offset (degrees)'); ax1.set_ylabel('Dec Offset (degrees)'); ax1.set_aspect('equal')
                
                im2 = ax2.imshow(cumulative_sn, extent=[-max_extent, max_extent, -max_extent, max_extent], origin='lower', cmap='viridis', interpolation='nearest')
                fig.colorbar(im2, ax=ax2, label='Relative Cumulative S/N')
                ax2.set_title('Cumulative Sensitivity (S/N)')
                ax2.set_xlabel('RA Offset (degrees)'); ax2.set_ylabel('Dec Offset (degrees)'); ax2.set_aspect('equal')

                filename = f"results_{scenario_key}.png"
                plt.savefig(os.path.join(output_dir, filename))
                plt.close(fig)
    print("Detailed plots saved.")


    # --- Generate Alternative Summary Plot (Panel Plot) - Corrected ---
    print("Generating alternative summary panel plot...")
    
    # Create a 2xN grid of subplots.
    # FIX: Set sharey=False to allow independent y-axis scaling for each subplot.
    fig, axes = plt.subplots(2, len(frequencies_ghz), figsize=(20, 10), sharex=True, sharey=False)
    fig.suptitle('Survey Performance vs. Search Radius for Different Frequencies', fontsize=20)

    for i, freq in enumerate(frequencies_ghz):
        # Filter data for the current frequency
        freq_data = summary_df[summary_df['frequency'] == freq].sort_values(by='radius_multiple')

        # --- Top Row: Peak S/N (Depth) ---
        ax_top = axes[0, i]
        ax_top.plot(freq_data['radius_multiple'], freq_data['peak_sn'], 'o-', color='C0')
        ax_top.set_title(f'{freq:.2f} GHz', fontsize=14)
        ax_top.grid(True, linestyle='--', alpha=0.6)
        if i == 0:
            ax_top.set_ylabel('Peak S/N (Depth)', fontsize=12)

        # --- Bottom Row: Effective Area (Width) ---
        ax_bottom = axes[1, i]
        ax_bottom.plot(freq_data['radius_multiple'], freq_data['effective_area'], 'o-', color='C1')
        ax_bottom.set_xlabel('Search Radius (x FWHM)', fontsize=12)
        ax_bottom.grid(True, linestyle='--', alpha=0.6)
        if i == 0:
            ax_bottom.set_ylabel('Effective Area (sq. deg)', fontsize=12)

    # Set common labels
    fig.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle

    # Save the new summary figure
    summary_filename_alt = "summary_panel_plot.png"
    plt.savefig(os.path.join(output_dir, summary_filename_alt))
    print(f"Alternative summary panel plot saved as '{os.path.join(output_dir, summary_filename_alt)}'")
    plt.close(fig)

    print("\nPhase 3 complete. All plots have been generated successfully.")
