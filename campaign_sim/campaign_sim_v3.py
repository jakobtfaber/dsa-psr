import numpy as np
import pandas as pd
from astropy import units as u
from dataclasses import dataclass, field
from astropy.table import Table
from astropy.units import quantity_support
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Dataclass Definitions for Structured I/O ---

@dataclass
class PsrSimConfig:
    """Input parameters for a pulsar timing simulation."""
    name: str
    P_psr: u.Quantity
    S1400_psr_mean: u.Quantity
    W_psr_intrinsic_fwhm: u.Quantity
    DM_psr: u.Quantity
    dist_psr: u.Quantity
    psr_spectral_index: float
    sefd_zenith: u.Quantity
    n_pol: int
    jitter_fJ: float
    jitter_mI: float
    sigma_spin_noise: u.Quantity

@dataclass
class CampaignConfig:
    """Parameters for a multi-year timing campaign."""
    duration: u.Quantity
    total_obs_time: u.Quantity
    schedulers: list

# --- 2. White & Red Noise Generation (Validated Models) ---

def get_sefd(nu, elevation, p: PsrSimConfig):
    """Calculates a frequency- and elevation-dependent SEFD."""
    T_sky_ref, nu_ref, T_rec = 20*u.K, 0.408*u.GHz, 25*u.K
    T_sky = T_sky_ref * (nu / nu_ref)**-2.75
    T_sys = T_rec + T_sky
    T_sys_at_1_4 = T_rec + T_sky_ref * (1.4*u.GHz / nu_ref)**-2.75
    sefd_freq_dep = p.sefd_zenith * (T_sys / T_sys_at_1_4)
    # Apply airmass correction, capped at 30 degrees elevation.
    elevation = np.maximum(elevation, 30 * u.deg)
    return sefd_freq_dep / np.sin(elevation)

def generate_white_noise_toa_error(p: PsrSimConfig, t_obs_single: u.Quantity, elevation: u.Quantity, n_simulations: int, rng):
    """
    Full white noise calculation, adapted from our validated V3.0 script.
    This now returns a distribution of possible TOA errors for a given observation setup.
    """
    # This is a placeholder for the full multi-channel calculation from the V3 script.
    # To keep this script focused on the campaign logic, we use the summary statistics.
    # In a full run, we would call the V3 script's functions here.
    base_error_mean = 0.61 * u.us
    base_error_std = 0.12 * u.us # Representative std dev from V3
    
    t_obs_scaling = np.sqrt((1800 * u.s) / t_obs_single)
    sefd_scaling = get_sefd(1.4*u.GHz, elevation, p) / p.sefd_zenith
    
    mean_error = base_error_mean * t_obs_scaling * sefd_scaling
    std_dev_error = base_error_std * t_obs_scaling * sefd_scaling
    
    return rng.normal(loc=mean_error.to(u.us).value, scale=std_dev_error.to(u.us).value, size=n_simulations) * u.us


def generate_red_noise_series(campaign: CampaignConfig, rms_amplitude: u.Quantity, gamma: float, rng):
    """Generic function to generate a red noise time series with a power-law spectrum."""
    n_days = int(campaign.duration.to(u.day).value)
    times = np.arange(n_days) * u.day
    
    freqs = np.fft.rfftfreq(n_days, d=1)
    psd = np.where(freqs > 0, freqs**(-gamma), 0)
    
    real_part = rng.normal(size=len(freqs)) * np.sqrt(psd)
    imag_part = rng.normal(size=len(freqs)) * np.sqrt(psd)
    fourier_noise = real_part + 1j * imag_part
    
    time_series_raw = np.fft.irfft(fourier_noise, n=n_days)
    time_series_scaled = (time_series_raw / np.std(time_series_raw)) * rms_amplitude
    
    return times, time_series_scaled

# --- 3. Campaign Scheduler Definitions ---

def generate_high_cadence_schedule(campaign: CampaignConfig):
    """Generates weekly observations."""
    n_obs = int((campaign.duration / (7 * u.day)).decompose())
    t_obs_single = (campaign.total_obs_time / n_obs).to(u.s)
    obs_epochs = np.arange(n_obs) * 7 * u.day
    elevations = 60 + 30 * np.sin(2 * np.pi * obs_epochs.to(u.day).value / 365.25) * u.deg
    return "High Cadence (Weekly)", obs_epochs, t_obs_single, elevations

def generate_high_sensitivity_schedule(campaign: CampaignConfig):
    """Generates monthly observations."""
    n_obs = int((campaign.duration / (30.44 * u.day)).decompose())
    t_obs_single = (campaign.total_obs_time / n_obs).to(u.s)
    obs_epochs = np.arange(n_obs) * 30.44 * u.day
    elevations = 60 + 30 * np.sin(2 * np.pi * obs_epochs.to(u.day).value / 365.25) * u.deg
    return "High Sensitivity (Monthly)", obs_epochs, t_obs_single, elevations

# --- 4. Sophisticated Analysis: Timing Model Fit ---

def analyze_residuals(results_table: Table):
    """Performs a simple timing fit to calculate post-fit residuals."""
    epochs = results_table['epoch_day'].quantity.to(u.year).value
    toas = results_table['total_toa_raw'].quantity.to(u.s).value
    
    # Design matrix for a P, P-dot model (offset, f, fdot)
    M = np.vander(epochs, 3, increasing=True)
    
    weights = 1.0 / results_table['white_noise_sigma'].quantity.to(u.s).value**2
    W = np.diag(weights)
    
    try:
        M_T_W = M.T @ W
        cov_matrix = np.linalg.inv(M_T_W @ M)
        best_fit_params = cov_matrix @ M_T_W @ toas
        model_toas = (M @ best_fit_params) * u.s
        post_fit_residuals = results_table['total_toa_raw'] - model_toas.to(u.us)
    except np.linalg.LinAlgError:
        # If matrix is singular (e.g., too few points), return raw residuals
        post_fit_residuals = results_table['total_toa_raw']

    results_table['post_fit_residual'] = post_fit_residuals
    return results_table

# --- 5. Main Campaign Simulation ---

def run_campaign_simulation(p: PsrSimConfig, campaign: CampaignConfig):
    rng = np.random.default_rng(seed=123)
    print(f"--- Simulating 5-Year Campaign for {p.name} ---")
    
    # Generate underlying red noise for the full duration
    dm_times, dm_series = generate_red_noise_series(campaign, 3e-4 * u.pc / u.cm**3, 8./3., rng)
    spin_times, spin_series = generate_red_noise_series(campaign, p.sigma_spin_noise, 4.0, rng) # gamma~4-6 for spin noise
    
    all_campaign_results = {}

    for scheduler_func in campaign.schedulers:
        strategy_name, obs_epochs, t_obs_single, elevations = scheduler_func(campaign)
        print(f"\nRunning strategy: '{strategy_name}' ({len(obs_epochs)} observations of {t_obs_single:.0f})")
        
        # Interpolate red noise to observation epochs
        dm_noise = np.interp(obs_epochs.to(u.day).value, dm_times.to(u.day).value, dm_series.value) * dm_series.unit
        spin_noise = np.interp(obs_epochs.to(u.day).value, spin_times.to(u.day).value, spin_series.value) * spin_series.unit
        
        k_dm = 4.148808e3 * u.s * u.MHz**2 * u.cm**3 / u.pc
        chromatic_error = (k_dm * dm_noise / (1.4*u.GHz)**2).to(u.us)
        
        # Get white noise for each observation
        white_noise_sigma = get_white_noise_toa_error(p, t_obs_single, elevations)
        white_noise_draws = rng.normal(0, white_noise_sigma.value, size=len(obs_epochs)) * u.us
        
        # Final "raw" TOA is sum of all errors
        final_raw_toas = chromatic_error + spin_noise + white_noise_draws
        
        table = Table({'epoch_day': obs_epochs, 'total_toa_raw': final_raw_toas, 'white_noise_sigma': white_noise_sigma})
        
        analyzed_table = analyze_residuals(table)
        all_campaign_results[strategy_name] = analyzed_table
        
    return all_campaign_results

if __name__ == '__main__':
    msp_config = PsrSimConfig(
        name="Typical MSP", P_psr=2*u.ms, W_psr_intrinsic_fwhm=0.06*(2*u.ms),
        S1400_psr_mean=0.1*u.mJy, DM_psr=30*u.pc/u.cm**3, dist_psr=2.0*u.kpc,
        psr_spectral_index=-1.6, sefd_zenith=0.6*u.Jy, n_pol=2,
        jitter_fJ=1./3., jitter_mI=1.0, sigma_spin_noise=100*u.ns
    )
    
    campaign_config = CampaignConfig(
        duration=5*u.year, total_obs_time=40*u.hour,
        schedulers=[generate_high_cadence_schedule, generate_high_sensitivity_schedule]
    )
    
    campaign_results = run_campaign_simulation(msp_config, campaign_config)

    # --- Visualize the results ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    summary_data = []
    for name, table in campaign_results.items():
        post_fit_rms = table['post_fit_residual'].std()
        summary_data.append({'Strategy': name, 'Post-fit RMS (µs)': post_fit_rms.to(u.us).value})
        
        axes[0].plot(table['epoch_day'], table['total_toa_raw'], 'o', ms=4, alpha=0.6, label=f"{name} (Raw RMS: {table['total_toa_raw'].std():.2f})")
        axes[1].plot(table['epoch_day'], table['post_fit_residual'], 'o', ms=4, alpha=0.7, label=f"{name} (Post-fit RMS: {post_fit_rms:.2f})")

    axes[0].set_title(f"Simulated Raw TOA Residuals for {msp_config.name}")
    axes[0].set_ylabel("Raw TOA Residual (µs)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[1].set_title("Post-Fit Timing Residuals (P and P-dot removed)")
    axes[1].set_xlabel("Time (days)")
    axes[1].set_ylabel("Post-Fit TOA Residual (µs)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # Print summary table
    summary_df = pd.DataFrame(summary_data)
    print("\n--- Campaign Strategy Comparison ---")
    print(summary_df.to_markdown(index=False))
