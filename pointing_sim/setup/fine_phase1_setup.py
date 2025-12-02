import numpy as np
import os

print("--- Phase 1: Setup and Configuration ---")

# Ensure a directory exists for the output data
output_dir = "simulation_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f"Output data will be saved in the '{output_dir}/' directory.")

# Step 1.1: Recreate the CASS Pointing Grid
print("Setting up the CASS pointing grid...")
spacing = 1.68
hex_coords_int = []
for i in range(-3, 4):
    for j in range(-3, 4):
        if abs(i + j) <= 2:
            hex_coords_int.append((i, j))
v1 = np.array([spacing, 0])
v2 = np.array([spacing * np.cos(np.pi / 3), spacing * np.sin(np.pi / 3)])
pointing_centers = np.array([i * v1 + j * v2 for i, j in hex_coords_int])
pointing_centers -= np.mean(pointing_centers, axis=0)

# Step 1.2: Define the Parameter Space
print("Defining the simulation parameter space...")
frequencies_ghz = np.array([0.7, 1.0, 1.35, 1.67, 2.0])

# MODIFICATION: Sample the search radius range more finely using np.linspace
# This creates 11 points from 0.5 to 1.5 inclusive.
search_radii_multiples = np.linspace(0.5, 1.5, 11)
print(f"Finely sampled search radii: {np.round(search_radii_multiples, 2)}")

f_ref_ghz = 2.0
fwhm_ref_deg = 1.68

# Save configuration to a file
config_filepath = os.path.join(output_dir, "cadence_config_fine.npz")
np.savez(
    config_filepath,
    pointing_centers=pointing_centers,
    frequencies_ghz=frequencies_ghz,
    search_radii_multiples=search_radii_multiples,
    f_ref_ghz=f_ref_ghz,
    fwhm_ref_deg=fwhm_ref_deg
)

print(f"\nConfiguration saved successfully to '{config_filepath}'")
print("Phase 1 Complete. You can now run the simulation script.")
