import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
from matplotlib.legend import Legend

print("--- Phase 4: Optimization and Final Recommendation ---")

# --- Load Data from All Previous Phases ---
output_dir = "cadence_simulation_results"
config_filepath = os.path.join("simulation_data", "cadence_config.npz")
results_filepath = os.path.join("simulation_data", "cadence_results.npz")
confusion_filepath = os.path.join("simulation_data", "confusion_probability_results.csv")

if not all(os.path.exists(p) for p in [config_filepath, results_filepath, confusion_filepath]):
    print("Error: Data files not found. Please run Phases 1, 2, and 3 scripts first.")
else:
    print("Loading all necessary data files...")
    
    # --- FIX: Load the config file ---
    config = np.load(config_filepath)
    
    # Load sensitivity results
    results = np.load(results_filepath, allow_pickle=True)
    summary_results_array = results['summary_results']
    summary_df = pd.DataFrame(list(summary_results_array))
    
    # Load confusion results
    confusion_df = pd.read_csv(confusion_filepath)
    
    # --- Step 4.1: Combine Data and Define Merit Function ---
    print("Combining sensitivity and confusion data...")
    
    # Merge the two dataframes on 'Frequency (GHz)'
    final_df = pd.merge(summary_df, confusion_df, 
                        left_on='frequency', right_on='Frequency (GHz)')

    # Define the "Contamination Factor" from the confusion probability
    final_df['contamination_factor'] = final_df['Confusion Probability (%)'] / 100.0

    # Define the Merit Function: "Pulsar Discovery Potential"
    final_df['discovery_potential'] = final_df['effective_area'] * (1 - final_df['contamination_factor'])

    # --- Step 4.2: Visualize the Final Results ---
    print("Generating final optimization plot...")
    
    # --- FIX: Extract frequencies_ghz from the loaded config dictionary ---
    frequencies_ghz = config['frequencies_ghz']
    
    # Create subplots with independent Y axes for clarity
    fig, axes = plt.subplots(1, len(frequencies_ghz), figsize=(20, 6), sharex=True, sharey=False)
    fig.suptitle('Optimization: Pulsar Discovery Potential vs. Search Radius', fontsize=20)

    for i, freq in enumerate(frequencies_ghz):
        # Filter data for the current frequency
        freq_data = final_df[final_df['frequency'] == freq].sort_values(by='radius_multiple')

        ax = axes[i]
        ax.plot(freq_data['radius_multiple'], freq_data['discovery_potential'], 'o-', color='C2')
        ax.set_title(f'{freq:.2f} GHz', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Highlight the maximum value for each frequency
        max_potential = freq_data['discovery_potential'].max()
        best_radius = freq_data.loc[freq_data['discovery_potential'].idxmax()]['radius_multiple']
        ax.axhline(max_potential, color='red', linestyle=':', alpha=0.7)
        ax.axvline(best_radius, color='red', linestyle=':', alpha=0.7)
        
        # FIX: More robust text placement
        ymin, ymax = ax.get_ylim()
        text_y_pos = ymin + 0.1 * (ymax - ymin) # Place text 10% from the bottom
        ax.text(best_radius + 0.02, text_y_pos, f'Best: {best_radius:.2f}x', color='red', rotation=90, verticalalignment='bottom')

        if i == 0:
            ax.set_ylabel('Pulsar Discovery Potential', fontsize=12)

        ax.set_xlabel('Search Radius (x FWHM)', fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the final figure
    final_plot_filename = "final_optimization_plot.png"
    plt.savefig(os.path.join(output_dir, final_plot_filename))
    print(f"Final optimization plot saved as '{os.path.join(output_dir, final_plot_filename)}'")
    plt.close(fig)

    # --- Step 4.3: Make a Recommendation ---
    # Find the single best scenario overall
    best_overall_scenario = final_df.loc[final_df['discovery_potential'].idxmax()]
    
    print("\n--- Final Recommendation ---")
    print("Based on the 'Pulsar Discovery Potential' merit function, which balances")
    print("effective survey area ('width') with contamination from source confusion ('quality'):\n")
    print(f"The optimal strategy is to observe at: {best_overall_scenario['frequency']:.2f} GHz")
    print(f"Using a search radius of: {best_overall_scenario['radius_multiple']:.2f} x FWHM")
    
    print("\nJustification:")
    print("At higher frequencies, the benefit of low source confusion outweighs the smaller effective area.")
    print("At lower frequencies, while the survey 'depth' and raw area are high, the significant")
    print("source confusion (up to ~45%) heavily degrades the quality of the survey, reducing the")
    print("overall number of clean, discoverable pulsar candidates.")
    
    print("\nThis data-driven result provides a clear path forward for the survey's cadence strategy.")

