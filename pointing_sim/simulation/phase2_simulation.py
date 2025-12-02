import numpy as np
import os

print("--- Phase 2: Simulation ---")

# Load configuration from Phase 1
config_filepath = os.path.join("simulation_data", "cadence_config_fine.npz")
if not os.path.exists(config_filepath):
    print(f"Error: Configuration file not found at '{config_filepath}'")
    print("Please run the Phase 1 setup script first.")
else:
    print(f"Loading configuration from '{config_filepath}'...")
    config = np.load(config_filepath)
    pointing_centers = config['pointing_centers']
    frequencies_ghz = config['frequencies_ghz']
    search_radii_multiples = config['search_radii_multiples']
    f_ref_ghz = config['f_ref_ghz']
    fwhm_ref_deg = config['fwhm_ref_deg']

    # Define the Gaussian gain model
    def gaussian_gain(delta_theta_deg, fwhm_deg):
        sigma_deg = fwhm_deg / (2 * np.sqrt(2 * np.log(2)))
        return np.exp(-0.5 * (delta_theta_deg / sigma_deg)**2)

    # Set up a 2D grid of sky coordinates
    max_extent = np.max(np.abs(pointing_centers)) + 2 * (fwhm_ref_deg * (f_ref_ghz / np.min(frequencies_ghz)))
    grid_resolution = 400
    x = np.linspace(-max_extent, max_extent, grid_resolution)
    y = np.linspace(-max_extent, max_extent, grid_resolution)
    xx, yy = np.meshgrid(x, y)
    sky_coords = np.vstack([xx.ravel(), yy.ravel()]).T

    # --- Run the Simulation ---
    print("Starting cumulative sensitivity simulation...")
    summary_results = []
    heatmap_data = {}

    for freq_ghz in frequencies_ghz:
        for radius_multiple in search_radii_multiples:
            
            scenario_key = f"{freq_ghz:.2f}GHz_{radius_multiple:.2f}xFWHM"
            print(f"  Simulating: {scenario_key}")

            current_fwhm_deg = fwhm_ref_deg * (f_ref_ghz / freq_ghz)
            current_search_radius_deg = radius_multiple * current_fwhm_deg
            
            num_visits = np.zeros(sky_coords.shape[0])
            total_sensitivity_squared = np.zeros(sky_coords.shape[0])

            # Loop through each of the 25 pointings
            for center in pointing_centers:
                # Calculate distance from each sky point to the current pointing center
                delta_theta = np.sqrt(np.sum((sky_coords - center)**2, axis=1))

                # Calculate the sensitivity gain from this single pointing across all sky points
                sensitivity_from_this_pointing = gaussian_gain(delta_theta, current_fwhm_deg)
                
                # *** THE FIX IS HERE ***
                # We only accumulate sensitivity for pixels that are *within* the search radius
                # for this specific pointing. We set the sensitivity to 0 for all pixels outside this radius.
                sensitivity_from_this_pointing[delta_theta > current_search_radius_deg] = 0.0

                # Add the squared sensitivity to the total accumulator.
                # Since sensitivity is 0 outside the radius, this correctly models the search area.
                total_sensitivity_squared += sensitivity_from_this_pointing**2

                # This calculation remains for creating the 'visits' heatmap
                num_visits[delta_theta <= current_search_radius_deg] += 1

            cumulative_sn = np.sqrt(total_sensitivity_squared)
            
            # Extract metrics
            peak_sn = np.max(cumulative_sn)
            effective_area_mask = cumulative_sn >= 1.0
            num_pixels_in_area = np.sum(effective_area_mask)
            pixel_area_sq_deg = (x[1] - x[0]) * (y[1] - y[0])
            effective_survey_area_sq_deg = num_pixels_in_area * pixel_area_sq_deg
            
            summary_results.append({
                'frequency': freq_ghz,
                'radius_multiple': radius_multiple,
                'peak_sn': peak_sn,
                'effective_area': effective_survey_area_sq_deg
            })
            
            # Store heatmap data
            heatmap_data[f"visits_{scenario_key}"] = num_visits.reshape(grid_resolution, grid_resolution)
            heatmap_data[f"sn_{scenario_key}"] = cumulative_sn.reshape(grid_resolution, grid_resolution)

    # Save results to a file
    results_filepath = os.path.join("simulation_data", "cadence_results_fine.npz")
    np.savez(
        results_filepath,
        summary_results=np.array(summary_results),
        heatmap_data=heatmap_data,
        max_extent=max_extent
    )
    print(f"\nSimulation results saved successfully to '{results_filepath}'")
    print("Phase 2 Complete. You can now run the analysis and plotting script.")
