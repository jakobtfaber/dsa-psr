import numpy as np
from scipy.integrate import quad

print("--- Phase 3.1: Source Density Estimation ---")

def condon_1984_source_counts(S_jy):
    """
    Calculates the differential source count n(S) in units of sr^-1 Jy^-1
    based on the polynomial fit from Condon (1984), as shown in Condon (2002).

    The model is a polynomial fit to log10(S^2.5 * n(S)).
    S_jy: Flux density in Janskys.
    """
    # The polynomial is valid for S > 10 microJy.
    # We will extrapolate below that, but our integration limit is 7 microJy, so this is fine.
    if S_jy <= 0:
        return 0

    x = np.log10(S_jy * 1e3)  # x is log10(S in mJy)

    # Polynomial coefficients for log10(S^2.5 * n(S)) from Condon 1984
    # where n(S) is in sr^-1 Jy^-1 and S is in Jy.
    # The formula is log10(y) = a0 + a1*x + a2*x^2 + ...
    a = [1.71, 0.177, -0.116, -0.015, 0.009]
    
    log_y = a[0] + a[1]*x + a[2]*x**2 + a[3]*x**3 + a[4]*x**4

    y = 10**log_y  # y = S^2.5 * n(S)
    
    # n(S) = y / S^2.5
    n_S = y / (S_jy**2.5)
    
    return n_S

def integrate_source_counts(S_limit_jy):
    """
    Integrates the differential source counts n(S) from a lower
    flux limit to infinity to get the total number of sources per steradian.
    """
    # We integrate from S_limit_jy up to a very high flux (e.g., 100 Jy),
    # which is effectively infinity for this calculation.
    integral, error = quad(condon_1984_source_counts, S_limit_jy, 100.0)
    return integral

# --- Main Calculation ---

# The DSA-2000 CASS point source detection threshold is 7 microJy.
flux_limit_microJy = 7.0
flux_limit_Jy = flux_limit_microJy * 1e-6

print(f"Calculating total source density above {flux_limit_microJy} uJy...")

# Get the total number of sources per steradian
total_sources_per_sr = integrate_source_counts(flux_limit_Jy)

# Convert steradians to square degrees for easier interpretation
# 1 steradian = (180/pi)^2 square degrees
sr_to_sq_deg = (180 / np.pi)**2
total_sources_per_sq_deg = total_sources_per_sr / sr_to_sq_deg

# This is our key result for Phase 3.1
rho_conf = total_sources_per_sq_deg

print("\n--- Results ---")
print(f"Integrated source density (rho_conf): {rho_conf:.2f} sources per square degree")
print("\nThis value represents the number of potentially confusing background sources")
print("brighter than the CASS detection limit of 7 uJy.")
print("\nPhase 3.1 Complete. We can now use this value to calculate confusion probability.")

# We can save this result for the next script if needed
np.savez("simulation_data/confusion_data.npz", rho_conf=rho_conf)
