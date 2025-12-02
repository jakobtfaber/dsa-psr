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
    
    # --- FIX 1: Correctly create the DataFrame ---
    # Explicitly convert the numpy array of objects back to a list of dicts
    # This ensures pandas interprets the dictionary keys as column names.
    summary_df = pd.DataFrame(list(summary_results_array))

    # --- Generate Detailed Heatmap Plots ---
    print("Generating detailed heatmap plots for each scenario...")
    for freq_ghz in frequencies_ghz:
        for radius_multiple in search_radii_multiples:
            scenario_key = f"{freq_ghz:.2f}GHz_{radius_multiple:.2f}xFWHM"
            num_visits = heatmap_data[f"visits_{scenario_key}"]
            cumulative_sn = heatmap_data[f"sn_{scenario_key}"]
            
            #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            #fig.suptitle(f'Results for {freq_ghz:.2f} GHz with Search Radius = {radius_multiple:.2f}x FWHM', fontsize=16)

            #im1 = ax1.imshow(num_visits, extent=[-max_extent, max_extent, -max_extent, max_extent], origin='lower', cmap='plasma', interpolation='nearest')
            #fig.colorbar(im1, ax=ax1, label='Number of Visits')
            #ax1.set_title('Overlapping Observations')
            #ax1.set_xlabel('RA Offset (degrees)'); ax1.set_ylabel('Dec Offset (degrees)'); ax1.set_aspect('equal')
            #
            #im2 = ax2.imshow(cumulative_sn, extent=[-max_extent, max_extent, -max_extent, max_extent], origin='lower', cmap='viridis', interpolation='nearest')
            #fig.colorbar(im2, ax=ax2, label='Relative Cumulative S/N')
            #ax2.set_title('Cumulative Sensitivity (S/N)')
            #ax2.set_xlabel('RA Offset (degrees)'); ax2.set_ylabel('Dec Offset (degrees)'); ax2.set_aspect('equal')

            #filename = f"results_{scenario_key}.png"
            #plt.savefig(os.path.join(output_dir, filename))
            #plt.close(fig)

    print("Detailed plots saved.")

    # --- Generate Final Corrected Summary Plot ---
    print("Generating final corrected summary plot...")
    plt.figure(figsize=(12, 9))
    colors = plt.cm.viridis(np.linspace(0, 1, len(frequencies_ghz)))
    markers = ['o', 's', 'D', '^', 'v', 'P', 'X', 'H', '8', 'p', 'o', 's', 'D', '^', 'v'] 

    # --- FIX 2: More robust plotting loop ---
    for i, freq in enumerate(frequencies_ghz):
        # Filter data for the current frequency and sort by the radius multiple
        freq_data = summary_df[summary_df['frequency'] == freq].sort_values(by='radius_multiple')
        
        # Plot the connecting line
        plt.plot(freq_data['effective_area'], freq_data['peak_sn'], 
                 linestyle='-', color=colors[i], 
                 label=f'{freq:.2f} GHz', zorder=2)
        
        # Plot the markers by iterating through the sorted rows
        for j, row in enumerate(freq_data.itertuples()):
            plt.scatter(row.effective_area, row.peak_sn,
                        marker=markers[j], # j corresponds to the sorted radius multiple
                        color=colors[i],
                        s=120, 
                        edgecolors='black', 
                        zorder=3)

    legend_elements = [Line2D([0], [0], color='grey', marker=m, linestyle='None', markersize=10, label=f'{mult}x FWHM') for m, mult in zip(markers, search_radii_multiples)]
    leg = Legend(plt.gca(), legend_elements, 'Radius Multiple', loc='lower right', frameon=True)
    plt.gca().add_artist(leg)
    plt.title('Pulsar Survey Strategy: Depth vs. Width Trade-off', fontsize=18)
    plt.xlabel('Effective Survey Area (sq. degrees)', fontsize=14)
    plt.ylabel('Peak Cumulative S/N (Relative to single pointing)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Frequency', loc='upper left')

    summary_filename = "summary_tradeoff_plot_fine.png"
    plt.savefig(os.path.join(output_dir, summary_filename))
    print(f"Summary plot saved as '{os.path.join(output_dir, summary_filename)}'")
    plt.close()

    print("\nPhase 3 complete. All plots have been generated successfully.")
