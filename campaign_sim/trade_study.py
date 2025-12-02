import numpy as np
import pandas as pd
from astropy import units as u
from dataclasses import dataclass, field
from astropy.table import Table
from astropy.units import quantity_support

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
    t_obs: u.Quantity
    n_pol: int
    nu_low: u.Quantity
    nu_high: u.Quantity
    n_channels: int
    jitter_fJ: float
    jitter_mI: float
    channel_centers: u.Quantity = field(init=False)
    channel_bw: u.Quantity = field(init=False)

    def __post_init__(self):
        """Calculate derived parameters after initialization."""
        self.channel_centers = np.linspace(self.nu_low.value, self.nu_high.value, self.n_channels) * self.nu_low.unit
        self.channel_bw = (self.nu_high - self.nu_low) / int(self.n_channels)

@dataclass
class TimingNoiseBudget:
    """Holds the results of the timing noise simulation as a table."""
    results_table: Table

    def print_summary(self):
        """Prints a formatted summary of the noise budget."""
        name = self.results_table.meta.get('name', '(unnamed)')
        print(f"--- Noise Contribution Analysis ({name}) ---")
        with quantity_support():
            for col in self.results_table.colnames:
                if 'total' in col: continue
                mean_val = self.results_table[col].quantity.mean()
                print(f"Mean {col.replace('_', ' ').title()}: {mean_val:.2f}")

            print("\n--- Final Timing Precision ---")
            total_noise = self.results_table['total_white_noise'].quantity
            mean_toa_unc = total_noise.mean()
            std_toa_unc = total_noise.std()
            
            print(f"Mean Total TOA Uncertainty: {mean_toa_unc:.2f} +/- {std_toa_unc:.2f}")
            p16, p50, p84 = np.percentile(total_noise.value, [16, 50, 84])
            print(f"Median: {p50*u.us:.2f}")
            print(f"68% Confidence Interval: [{p16*u.us:.2f}, {p84*u.us:.2f}]")


# --- 2. Corrected Physics Helper Functions ---

def get_sefd(nu, p: PsrSimConfig):
    """Calculates a frequency-dependent SEFD."""
    T_sky_ref, nu_ref, T_rec = 20*u.K, 0.408*u.GHz, 25*u.K
    T_sky = T_sky_ref * (nu / nu_ref)**-2.75
    T_sys = T_rec + T_sky
    T_sys_at_1_4 = T_rec + T_sky_ref * (1.4*u.GHz / nu_ref)**-2.75
    return p.sefd_zenith * (T_sys / T_sys_at_1_4)

def get_scattering_timescale(p: PsrSimConfig, nu, rng, n_simulations):
    """Calculates a distribution of scattering timescales (tau_d)."""
    scatter_dex = 0.65
    log_tau_d_ms_mean = -6.46 + 0.154*np.log10(p.DM_psr.value) + 1.07*(np.log10(p.DM_psr.value))**2 - 3.86*np.log10(nu.to(u.GHz).value)
    # Ensure mean is array-like for broadcasting
    log_tau_d_ms_mean_arr = np.asarray(log_tau_d_ms_mean)
    size = (len(nu), n_simulations) if hasattr(nu, "__len__") else n_simulations
    stochastic_log_tau_ms = rng.normal(log_tau_d_ms_mean_arr[..., np.newaxis], scatter_dex, size=size)
    return (10**stochastic_log_tau_ms * u.ms).to(u.s)

def get_radiometer_noise(S_mean, W_obs, P_psr, sefd, t_obs, bw, n_pol):
    """
    Calculates TOA uncertainty from radiometer noise.
    Ref: Cordes & Shannon (2010), Appendix A.
    """
    duty_cycle = (W_obs / P_psr).decompose()
    S_peak = np.where(duty_cycle > 0, S_mean / duty_cycle, 0 * u.Jy)
    
    snr_profile = np.where(S_peak > 0*u.Jy, S_peak / sefd * np.sqrt(n_pol * bw * t_obs), np.inf*u.dimensionless_unscaled)
    
    # Note: This approximation mixes a Gaussian FWHM (W_intrinsic) with an
    # exponential tail (tau_d). A full treatment requires Fourier-domain matched-filtering.
    W_eff = W_obs / np.sqrt(8 * np.log(2)) 
    sigma_rad = W_eff / (snr_profile * np.sqrt(N_pulses))
    return sigma_rad.to(u.us)
    
def get_jitter_noise(p: PsrSimConfig):
    """Calculates TOA uncertainty from pulse phase jitter. Ref: C&S Eq. A6."""
    N_pulses = (p.t_obs / p.P_psr).decompose().value
    sigma_jitter = (p.jitter_fJ * p.W_psr_intrinsic_fwhm * np.sqrt(1 + p.jitter_mI**2)) / (2 * np.sqrt(2 * N_pulses * np.log(2)))
    return sigma_jitter.to(u.us)

def get_finite_scintle_noise(tau_d, p: PsrSimConfig, nu):
    """Calculates TOA uncertainty from finite-scintle noise. Ref: C&S Eqs. 23-24."""
    dnu_d = (1 / (2 * np.pi * tau_d))
    dt_iss = 220 * u.s * np.sqrt(p.dist_psr.to(u.kpc).value) * (nu.to(u.GHz).value)**(-11./5.)
    eta_t, eta_nu = 0.2, 0.2
    
    bw_tot = p.nu_high - p.nu_low
    N_t = 1 + eta_t * (p.t_obs / dt_iss)
    N_nu = 1 + eta_nu * (bw_tot / dnu_d)
    N_diss = (N_t * N_nu).decompose()
    
    return (tau_d / np.sqrt(N_diss)).to(u.us)

def run_timing_simulation_v3(p: PsrSimConfig, n_simulations=5000):
    """Main function for the V3.0 simulation."""
    rng = np.random.default_rng(seed=42)
    print(f"--- Running {p.name} ({n_simulations} iterations) ---")
    
    gains = rng.exponential(scale=1.0, size=(p.n_channels, n_simulations))
    
    # --- Radiometer Noise ---
    S_chan_mean = p.S1400_psr_mean * (p.channel_centers[:, np.newaxis] / (1.4 * u.GHz))**p.psr_spectral_index
    S_chan_inst = S_chan_mean * gains
    
    tau_d_chan = get_scattering_timescale(p, p.channel_centers, rng, n_simulations)
    W_obs_chan = np.sqrt(p.W_psr_intrinsic_fwhm**2 + tau_d_chan**2)
    sefd_chan = get_sefd(p.channel_centers, p)

    # Pass all required arguments, including n_pol
    sigma_rad_chan = get_radiometer_noise(S_chan_inst, W_obs_chan, p.P_psr, sefd_chan[:, np.newaxis], p.t_obs, p.channel_bw, p.n_pol)
    
    # Use correct units for zero in where clause
    inv_var_sum = np.sum(np.where(np.isfinite(sigma_rad_chan), 1/sigma_rad_chan**2, 0*u.us**-2), axis=0)
    sigma_rad_total = np.where(inv_var_sum > 0, 1 / np.sqrt(inv_var_sum), np.inf * u.us)

    # --- Jitter & Scintle Noise ---
    sigma_jitter_total = get_jitter_noise(p)
    
    nu_center_band = (p.nu_low + p.nu_high) / 2
    # Pass the full distribution of tau_d, not the mean
    tau_d_center_dist = get_scattering_timescale(p, nu_center_band, rng, n_simulations)
    sigma_scatt_total = get_finite_scintle_noise(tau_d_center_dist, p, nu_center_band)
    
    # --- Combine and Return ---
    results_table = Table({
        'radiometer_error': sigma_rad_total,
        'jitter_error': np.repeat(sigma_jitter_total, n_simulations),
        'scintillation_error': sigma_scatt_total,
        'total_white_noise': np.sqrt(sigma_rad_total**2 + sigma_jitter_total**2 + sigma_scatt_total**2)
    }, meta={'name': p.name})
    return TimingNoiseBudget(results_table)

def perform_trade_study():
    """Runs the timing simulation across a grid of DM and center frequency values."""
    # Use np.array() for unit broadcasting
    dm_values = np.array([10, 30, 100, 300]) * u.pc / u.cm**3
    center_freq_values = np.array([800, 1400, 2000]) * u.MHz
    
    results_grid = {}
    base_config_dict = {
        "P_psr": 2 * u.ms, "W_psr_intrinsic_fwhm": 0.06 * (2*u.ms),
        "S1400_psr_mean": 0.1 * u.mJy, "dist_psr": 2.0 * u.kpc,
        "psr_spectral_index": -1.6, "sefd_zenith": 0.6 * u.Jy,
        "t_obs": 1800 * u.s, "n_pol": 2, "n_channels": 32,
        "jitter_fJ": 1./3., "jitter_mI": 1.0,
    }
    
    dsa_total_bw = 312.5 * u.MHz
    print("--- Starting DSA-2000 Timing Precision Trade Study ---")
    
    for dm in dm_values:
        results_grid[dm.value] = {}
        for nu_center in center_freq_values:
            config = PsrSimConfig(
                name=f"MSP (DM={dm.value}, nu_c={nu_center.to(u.MHz).value:.0f} MHz)",
                DM_psr=dm, nu_low=nu_center - dsa_total_bw/2, nu_high=nu_center + dsa_total_bw/2,
                **base_config_dict
            )
            
            # Call correct function name
            budget = run_timing_simulation_v3(config, n_simulations=1000)
            
            with quantity_support():
                median_toa_error = np.median(budget.results_table['total_white_noise'])
            results_grid[dm.value][nu_center.to(u.MHz).value] = median_toa_error

    # --- Format and Print Results Table ---
    df = pd.DataFrame(results_grid).T
    df.index.name = "DM (pc/cm^3)"
    df.columns = [f"{int(col)} MHz" for col in df.columns]
    
    # More robust formatter for mixed type columns
    def formatter(x):
        if isinstance(x, u.Quantity) and np.isfinite(x.value):
            return f"{x.to(u.us).value:.2f} µs"
        return "Scattering Dominated"
    
    df = df.applymap(formatter)

    print("\n\n--- Trade Study Results: Median TOA Uncertainty ---")
    print(df.to_markdown())

if __name__ == '__main__':
    perform_trade_study()