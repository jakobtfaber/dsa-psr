import numpy as np
import pandas as pd
from astropy import units as u
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
    sigma_spin_noise: u.Quantity # ✨ NEW: Amplitude of intrinsic spin noise

@dataclass
class CampaignConfig:
    """Parameters for a multi-year timing campaign."""
    duration: u.Quantity
    total_obs_time: u.Quantity
    schedulers: list

# --- 2. White Noise Models (Simplified Core) ---

def get_elevation_dependent_sefd(p: PsrSimConfig, elevation: u.Quantity):
    """ ✨ NEW: Applies an airmass correction to the zenith SEFD. """
    # A simple 1/sin(el) airmass model. Capped at 30 degrees.
    elevation = np.maximum(elevation, 30 * u.deg)
    return p.sefd_zenith / np.sin(elevation)

def get_white_noise_toa_error(p: PsrSimConfig, t_obs_single: u.Quantity, elevation: u.Quantity):
    """
    Simplified placeholder for our full V3.0 white noise calculation.
    ✨ NEW: Now includes elevation-dependent SEFD.
    """
    base_error = 0.61 * u.us
    # Scale error by sqrt(t_obs)
    t_obs_scaling = np.sqrt((1800 * u.s) / t_obs_single)
    # Scale error by SEFD change
    sefd_scaling = get_elevation_dependent_sefd(p, elevation) / p.sefd_zenith
    return base_error * t_obs_scaling * sefd_scaling

# --- 3. Red Noise Generation ---

def generate_red_noise_series(campaign: CampaignConfig, rms_amplitude: u.Quantity, gamma: float, rng):
    """
    Generic function to generate a red noise time series with a power-law spectrum.
    """
    n_days = int(campaign.duration.to(u.day).value)
    times = np.arange(n_days) * u.day
    
    freqs = np.fft.rfftfreq(n_days, d=1)
    psd = np.where(freqs > 0, freqs**(-gamma), 0)
    
    real_part = rng.normal(size=len(freqs)) * np.sqrt(psd)
    imag_part = rng.normal(size=len(freqs)) * np.sqrt(psd)
    fourier_noise = real_part + 1j * imag_part
    
    time_series_raw = np.fft.irfft(fourier_noise, n=n_days)
    
    # Scale to the correct physical amplitude
    time_series_scaled = (time_series_raw / np.std(time_series_raw)) * rms_amplitude
    return times, time_series_scaled

# --- 4. Campaign Scheduler Definitions ---

def generate_high_cadence_schedule(campaign: CampaignConfig):
    """Generates weekly observations."""
    n_obs = int((campaign.duration / (7 * u.day)).decompose())
    t_obs_single = (campaign.total_obs_time / n_obs).to(u.s)
    obs_epochs = np.arange(n_obs) * 7 * u.day
    # Assume a simple repeating elevation track for each observation
    elevations = 60 + 30 * np.sin(2 * np.pi * obs_epochs.to(u.day).value / 365.25) * u.deg
    return "High Cadence (Weekly)", obs_epochs, t_obs_single, elevations

def generate_high_sensitivity_schedule(campaign: CampaignConfig):
    """Generates monthly observations."""
    n_obs = int((campaign.duration / (30.44 * u.day)).decompose())
    t_obs_single = (campaign.total_obs_time / n_obs).to(u.s)
    obs_epochs = np.arange(n_obs) * 30.44 * u.day
    elevations = 60 + 30 * np.sin(2 * np.pi * obs_epochs.to(u.day).value / 365.25) * u.deg
    return "High Sensitivity (Monthly)", obs_epochs, t_obs_single, elevations

# --- 5. Sophisticated Analysis: Timing Model Fit ---

def analyze_residuals(results_table: Table):
    """
    ✨ NEW: Performs a simple timing fit to calculate post-fit residuals.
    This simulates solving for P and P-dot and seeing what noise remains.
    """
    epochs = results_table['epoch_day'].quantity.to(u.year).value
    toas = results_table['total_toa_raw'].quantity.to(u.s).value
    
    # Design matrix for a simple P and P-dot model (offset, f, fdot)
    M = np.vander(epochs, 3)
    
    # White noise weighting matrix (inverse of variance)
    weights = 1.0 / results_table['white_noise_sigma'].quantity.to(u.s).value**2
    W = np.diag(weights)
    
    # Solve for the best-fit parameters using weighted least squares
    M_T_W = M.T @ W
    cov_matrix = np.linalg.inv(M_T_W @ M)
    best_fit_params = cov_matrix @ M_T_W @ toas
    
    # Calculate the model predicted TOAs and the post-fit residuals
    model_toas = (M @ best_fit_params) * u.s
    post_fit_residuals = results_table['total_toa_raw'] - model_toas.to(u.us)
    
    results_table['post_fit_residual'] = post_fit_residuals
    return results_table

# --- 6. Main Campaign Simulation ---

def run_campaign_simulation(p: PsrSimConfig, campaign: CampaignConfig):
    """Simulates a full timing campaign for a given pulsar and multiple cadences."""
    rng = np.random.default_rng(seed=123)
    
    print(f"--- Simulating 5-Year Campaign for {p.name} ---")
    
    # 1. Generate underlying red noise components
    dm_times, dm_series = generate_red_noise_series(campaign, 3e-4 * u.pc / u.cm**3, 8./3., rng)
    spin_times, spin_series = generate_red_noise_series(campaign, p.sigma_spin_noise, 5.0, rng)
    
    all_campaign_results = {}

    for scheduler_func in campaign.schedulers:
        strategy_name, obs_epochs, t_obs_single, elevations = scheduler_func(campaign)
        print(f"\nRunning strategy: '{strategy_name}'")
        print(f" -> {len(obs_epochs)} observations, {t_obs_single:.0f} each")
        
        # Interpolate red noise to observation epochs
        dm_noise = np.interp(obs_epochs, dm_times, dm_series) * u.pc/u.cm**3
        spin_noise = np.interp(obs_epochs, spin_times, spin_series).to(u.us)
        
        k_dm = 4.148808e3 * u.s * u.MHz**2 * u.cm**3 / u.pc
        chromatic_error = (k_dm * dm_noise / (1.4*u.GHz)**2).to(u.us)
        
        # Calculate white noise sigma for each observation
        white_noise_sigma = get_white_noise_toa_error(p, t_obs_single, elevations)
        white_noise_draws = rng.normal(0, white_noise_sigma.value, size=len(obs_epochs)) * u.us
        
        # Final "raw" TOA residual is the sum of all noise terms
        final_raw_toas = chromatic_error + spin_noise + white_noise_draws
        
        results_table = Table({
            'epoch_day': obs_epochs,
            'total_toa_raw': final_raw_toas,
            'white_noise_sigma': white_noise_sigma
        })
        
        # ✨ NEW: Analyze the results to get post-fit residuals
        analyzed_table = analyze_residuals(results_table)
        all_campaign_results[strategy_name] = analyzed_table
        
    return all_campaign_results

if __name__ == '__main__':
    msp_config = PsrSimConfig(
        name="Typical MSP", P_psr=2*u.ms, W_psr_intrinsic_fwhm=0.06*(2*u.ms),
        S1400_psr_mean=0.1*u.mJy, DM_psr=30*u.pc/u.cm**3, dist_psr=2.0*u.kpc,
        psr_spectral_index=-1.6, sefd_zenith=0.6*u.Jy, n_pol=2,
        jitter_fJ=1./3., jitter_mI=1.0,
        sigma_spin_noise=100*u.ns # Assume 100 ns intrinsic spin noise
    )
    
    campaign_config = CampaignConfig(
        duration=5*u.year, total_obs_time=40*u.hour,
        schedulers=[generate_high_cadence_schedule, generate_high_sensitivity_schedule]
    )
    
    campaign_results = run_campaign_simulation(msp_config, campaign_config)

    # --- Visualize the results ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, table in campaign_results.items():
        # ✨ NEW: Plotting post-fit residuals
        post_fit_rms = table['post_fit_residual'].std()
        ax.plot(table['epoch_day'], table['post_fit_residual'], 'o', ms=4, alpha=0.7,
                label=f"{name} (Post-fit RMS: {post_fit_rms:.2f})")

    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Post-Fit TOA Residual (µs)")
    ax.set_title(f"Simulated Post-Fit Timing Residuals for {msp_config.name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
