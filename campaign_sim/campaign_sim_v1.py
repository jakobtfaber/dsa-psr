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
    """Input parameters for a pulsar timing simulation. Version 3.0"""
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
    
@dataclass
class CampaignConfig:
    """Parameters for a multi-year timing campaign."""
    duration: u.Quantity
    total_obs_time: u.Quantity
    schedulers: list # List of scheduler functions to run
    
# --- 2. White Noise Models (Validated V3.0 Core) ---
# NOTE: These functions are simplified for this example. A full implementation
# would use the multi-channel approach from our previous script.
def get_white_noise_toa_error(p: PsrSimConfig, t_obs_single: u.Quantity):
    """
    A simplified placeholder for our full V3.0 white noise calculation.
    For this example, it returns a fixed value representative of the MSP case.
    """
    # In a real run, this would call the full V3.0 simulation.
    # We use a representative value from our previous MSP sanity check.
    base_error = 0.61 * u.us
    # Scale error by sqrt(t_obs)
    scaling_factor = np.sqrt((1800 * u.s) / t_obs_single)
    return base_error * scaling_factor


# --- 3. Red Noise Generation ---

def generate_dm_wander(p: PsrSimConfig, campaign: CampaignConfig, rng):
    """
    Generates a realistic time series of DM variations (DM wander).
    
    This creates colored noise with a power-law spectrum S(f) ~ f^-gamma,
    consistent with a Kolmogorov turbulence model.
    
    Ref: Cordes & Shannon (2010), Eq. 30 and surrounding discussion.
    """
    # The structure function for DM variations from Kolmogorov turbulence
    # D_DM(tau) ~ tau^(5/3), implies a power spectrum P(f) ~ f^(-8/3).
    gamma = 8. / 3.
    
    # Estimate amplitude from C&S Eq. 13.
    # sigma_DM(1yr) ~ 1e-4 pc/cm^3 for a typical mid-latitude pulsar.
    # This is the RMS of DM *differences*. The RMS of the series is ~ a few times this.
    dm_rms_1yr = 3e-4 * u.pc / u.cm**3
    
    # Generate the time series
    n_days = int(campaign.duration.to(u.day).value)
    times = np.arange(n_days) * u.day
    
    # Create noise in the Fourier domain
    freqs = np.fft.rfftfreq(n_days, d=1)
    # Avoid division by zero at f=0
    psd = np.where(freqs > 0, freqs**(-gamma), 0)
    
    # Generate random phases and amplitudes
    real_part = rng.normal(size=len(freqs)) * np.sqrt(psd)
    imag_part = rng.normal(size=len(freqs)) * np.sqrt(psd)
    fourier_noise = real_part + 1j * imag_part
    
    # Inverse transform to get the time series
    dm_series_raw = np.fft.irfft(fourier_noise, n=n_days)
    
    # Scale to the correct physical amplitude
    dm_series_scaled = (dm_series_raw / np.std(dm_series_raw)) * dm_rms_1yr
    
    return times, dm_series_scaled


# --- 4. Campaign Scheduler Definitions ---

def generate_high_cadence_schedule(campaign: CampaignConfig):
    """Generates weekly observations, adjusting t_obs to meet total time."""
    n_obs = int((campaign.duration / (7 * u.day)).decompose())
    t_obs_single = (campaign.total_obs_time / n_obs).to(u.s)
    obs_epochs = np.arange(n_obs) * 7 * u.day
    return "High Cadence (Weekly)", obs_epochs, t_obs_single

def generate_high_sensitivity_schedule(campaign: CampaignConfig):
    """Generates monthly observations, adjusting t_obs to meet total time."""
    n_obs = int((campaign.duration / (30.44 * u.day)).decompose())
    t_obs_single = (campaign.total_obs_time / n_obs).to(u.s)
    obs_epochs = np.arange(n_obs) * 30.44 * u.day
    return "High Sensitivity (Monthly)", obs_epochs, t_obs_single


# --- 5. Main Campaign Simulation ---

def run_campaign_simulation(p: PsrSimConfig, campaign: CampaignConfig):
    """
    Simulates a full timing campaign for a given pulsar and multiple cadences.
    """
    rng = np.random.default_rng(seed=123)
    
    print(f"--- Simulating 5-Year Campaign for {p.name} ---")
    
    # 1. Generate the underlying red noise for the entire campaign duration
    dm_times, dm_series = generate_dm_wander(p, campaign, rng)
    
    all_campaign_results = {}

    # 2. Loop through each observing strategy
    for scheduler_func in campaign.schedulers:
        
        strategy_name, obs_epochs, t_obs_single = scheduler_func(campaign)
        print(f"\nRunning strategy: '{strategy_name}'")
        print(f" -> {len(obs_epochs)} observations, {t_obs_single:.0f} each")
        
        # 3. Simulate TOAs for this specific cadence
        
        # Interpolate the DM wander to the observation epochs
        dm_red_noise_at_epochs = np.interp(obs_epochs, dm_times, dm_series)
        
        # Calculate the chromatic timing error from this DM wander
        # delta_t = k_DM * delta_DM / nu^2
        k_dm = 4.148808e3 * u.s * u.MHz**2 * u.cm**3 / u.pc
        chromatic_error = (k_dm * dm_red_noise_at_epochs / (1.4*u.GHz)**2).to(u.us)
        
        # Get the white noise uncertainty for this observation length
        white_noise_sigma = get_white_noise_toa_error(p, t_obs_single)
        
        # Generate random white noise draws for each observation
        white_noise_draws = rng.normal(0, white_noise_sigma.value, size=len(obs_epochs)) * u.us
        
        # Final observed TOA is the sum of errors (ideal TOA is 0)
        final_toas = chromatic_error + white_noise_draws
        
        # Store results in a table
        results_table = Table({
            'epoch_day': obs_epochs,
            'dm_wander_toa_error': chromatic_error,
            'white_noise_toa_error': white_noise_draws,
            'total_toa_residual': final_toas
        }, meta={'name': strategy_name, 't_obs': t_obs_single})
        
        all_campaign_results[strategy_name] = results_table
        
    return all_campaign_results


if __name__ == '__main__':
    
    # --- Define the Pulsar and Campaign ---
    msp_config = PsrSimConfig(
        name="Typical MSP",
        P_psr=2 * u.ms, W_psr_intrinsic_fwhm=0.06 * (2*u.ms),
        S1400_psr_mean=0.1 * u.mJy, DM_psr=30 * u.pc / u.cm**3,
        dist_psr=2.0 * u.kpc, psr_spectral_index=-1.6,
        sefd_zenith=0.6 * u.Jy, t_obs=1800 * u.s, n_pol=2,
        nu_low=0.7 * u.GHz, nu_high=2.0 * u.GHz, n_channels=32,
        jitter_fJ=1./3., jitter_mI=1.0
    )
    
    campaign_config = CampaignConfig(
        duration = 5 * u.year,
        total_obs_time = 40 * u.hour,
        schedulers = [generate_high_cadence_schedule, generate_high_sensitivity_schedule]
    )

    # --- Run the Simulation and Print Summary ---
    campaign_results = run_campaign_simulation(msp_config, campaign_config)

    # --- Visualize the results ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, table in campaign_results.items():
        ax.plot(table['epoch_day'], table['total_toa_residual'], 'o', ms=4, alpha=0.7, label=f"{name} (RMS: {table['total_toa_residual'].std():.2f})")

    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("TOA Residual (µs)")
    ax.set_title(f"Simulated Timing Residuals for {msp_config.name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

