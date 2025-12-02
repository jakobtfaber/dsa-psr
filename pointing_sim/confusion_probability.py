import numpy as np
import pandas as pd
import os

print("--- Phase 3.2: Confusion Probability Calculation ---")

# --- Load Data from Previous Steps ---

# Load rho_conf from the file saved in Phase 3.1
confusion_data_path = "simulation_data/confusion_data.npz"
if not os.path.exists(confusion_data_path):
    print(f"Error: Confusion data file not found at '{confusion_data_path}'")
    print("Please run the Phase 3.1 script first.")
else:
    print(f"Loading confusion data from '{confusion_data_path}'...")
    confusion_data = np.load(confusion_data_path)
    rho_conf_per_sq_deg = confusion_data['rho_conf']

# Load frequencies from the Phase 1 config file
config_filepath = os.path.join("simulation_data", "cadence_config.npz")
if not os.path.exists(config_filepath):
     print(f"Error: Configuration file not found at '{config_filepath}'")
     print("Please run the Phase 1 setup script first.")
else:
    print(f"Loading configuration from '{config_filepath}'...")
    config = np.load(config_filepath)
    frequencies_ghz = config['frequencies_ghz']
    f_ref_ghz = config['f_ref_ghz']


    # --- Step 3.2a: Calculate Synthesized Beam Area ---
    print("\nCalculating synthesized beam area for each frequency...")

    # Synthesized beam FWHM is 2.2 arcsec at 2.0 GHz, scales inversely with frequency
    synth_fwhm_ref_arcsec = 2.2
    synth_fwhm_arcsec = synth_fwhm_ref_arcsec * (f_ref_ghz / frequencies_ghz)
    
    # Convert FWHM from arcseconds to degrees
    synth_fwhm_deg = synth_fwhm_arcsec / 3600.0
    
    # The solid angle (area) of a 2D Gaussian beam is Omega = (pi / (4 * ln(2))) * FWHM^2
    # This is the area we will use for confusion calculation.
    beam_area_sq_deg = (np.pi / (4 * np.log(2))) * (synth_fwhm_deg**2)


    # --- Step 3.2b: Calculate Confusion Probability ---
    print("Calculating confusion probability...")

    # The expected number of confusing sources per beam (lambda) is rho * area
    lambda_per_beam = rho_conf_per_sq_deg * beam_area_sq_deg
    
    # The probability of confusion is P(N>=1) = 1 - P(N=0).
    # For a Poisson process, P(N=0) = exp(-lambda).
    # So, P_conf = 1 - exp(-lambda_per_beam)
    p_confusion = 1 - np.exp(-lambda_per_beam)
    

    # --- Display Results ---
    print("\n--- Results ---")
    
    results_df = pd.DataFrame({
        "Frequency (GHz)": frequencies_ghz,
        "Synth. Beam FWHM (arcsec)": synth_fwhm_arcsec,
        "Synth. Beam Area (sq. deg)": beam_area_sq_deg,
        "Expected Sources per Beam (lambda)": lambda_per_beam,
        "Confusion Probability (%)": p_confusion * 100
    })
    
    # Set display format for better readability
    pd.set_option('display.float_format', '{:.2e}'.format)
    print(results_df)

    # Save the results for the next phase
    results_df.to_csv("simulation_data/confusion_probability_results.csv", index=False)
    print("\nResults saved to 'simulation_data/confusion_probability_results.csv'")
    print("Phase 3.2 Complete.")

