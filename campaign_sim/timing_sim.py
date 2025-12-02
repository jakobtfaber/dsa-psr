import numpy as np
from astropy import units as u
from astropy.constants import G
from dataclasses import dataclass
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

@dataclass
class TimingNoiseBudget:
    """Holds the results of the timing noise simulation as a table."""
    results_table: Table

    def print_summary(self):
        """Prints a formatted summary of the noise budget."""
        print(f"--- V3.0 Noise Contribution Analysis ({self.results_table.meta['name']}) ---")
        for col in self.results_table.colnames:
            if 'total' in col: continue
            # 🚨 FIX: Use .quantity.mean() for robust unit handling with Astropy tables.
            mean_val = self.results_table[col].quantity.mean()
            print(f"Mean {col.replace('_', ' ').title()}: {mean_val:.2f}")
        
        print("\n--- Final V3.0 Timing Precision ---")
        total_noise = self.results_table['total_white_noise'].quantity
        mean_toa_unc = total_noise.mean()
        std_toa_unc = total_noise.std()
        
        print(f"Mean Total TOA Uncertainty: {mean_toa_unc:.2f} +/- {std_toa_unc:.2f}")
        # 🚨 FIX: Re-attach units after numpy operation.
        with quantity_support():
            p16, p84 = np.percentile(total_noise.value, [16, 84])
        print(f"68% Confidence Interval: [{p16*u.us:.2f}, {p84*u.us:.2f}]")


# --- 2. Corrected Physics Helper Functions ---

def get_sefd(nu, p: PsrSimConfig):
    T_sky_ref, nu_ref, T_rec = 20*u.K, 0.408*u.GHz, 25*u.K
    T_sky = T_sky_ref * (nu / nu_ref)**-2.75
    T_sys = T_rec + T_sky
    T_sys_at_1_4 = T_rec + T_sky_ref * (1.4*u.GHz / nu_ref)**-2.75
    return p.sefd_zenith * (T_sys / T_sys_at_1_4)

def get_scattering_timescale(p: PsrSimConfig, nu, rng, n_simulations):
    scatter_dex = 0.65
    log_tau_d_ms_mean = -6.46 + 0.154*np.log10(p.DM_psr.value) + 1.07*(np.log10(p.DM_psr.value))**2 - 3.86*np.log10(nu.to(u.GHz).value)
    size = (len(nu), n_simulations) if hasattr(nu, "__len__") else n_simulations
    stochastic_log_tau_ms = rng.normal(log_tau_d_ms_mean[..., np.newaxis], scatter_dex, size=size)
    return (10**stochastic_log_tau_ms * u.ms).to(u.s)

def get_radiometer_noise(S_mean, W_obs, P_psr, sefd, t_obs, bw, n_pol):
    # Calculating W_eff from a quadrature sum of a Gaussian (intrinsic) and
    # a one-sided exponential (scattering) is an approximation. For this work,
    # we treat the resulting observed pulse as roughly Gaussian. A full
    # treatement requires Fourier-domain matched-filtering (Downs & Reichley 1983).
    W_eff = W_obs / np.sqrt(8 * np.log(2))
    duty_cycle = (W_eff / P_psr).decompose()
    
    S_peak = np.where(duty_cycle > 0, S_mean / duty_cycle, 0 * u.Jy)
    snr = np.where(S_peak > 0*u.Jy, S_peak / sefd * np.sqrt(n_pol * bw * t_obs), np.inf*u.dimensionless_unscaled)
    
    sigma_rad = W_eff / snr
    return sigma_rad.to(u.us)
    
def get_jitter_noise(p: PsrSimConfig):
    N_pulses = (p.t_obs / p.P_psr).decompose().value
    sigma_jitter = (p.jitter_fJ * p.W_psr_intrinsic_fwhm * np.sqrt(1 + p.jitter_mI**2)) / (2 * np.sqrt(2 * N_pulses * np.log(2)))
    return sigma_jitter.to(u.us)

def get_finite_scintle_noise(tau_d, p: PsrSimConfig, nu):
    dnu_d = (1 / (2 * np.pi * tau_d))
    # Use more physical scaling for DISS timescale, now exposed as an argument.
    dt_iss = 220 * u.s * np.sqrt(p.dist_psr.to(u.kpc).value) * (nu.to(u.GHz).value)**(-11./5.)
    eta_t, eta_nu = 0.2, 0.2
    
    bw_tot = p.nu_high - p.nu_low
    N_t = 1 + eta_t * (p.t_obs / dt_iss)
    N_nu = 1 + eta_nu * (bw_tot / dnu_d)
    N_diss = (N_t * N_nu).decompose()
    
    return (tau_d / np.sqrt(N_diss)).to(u.us)

def run_timing_simulation(p: PsrSimConfig, n_simulations=5000):
    rng = np.random.default_rng(seed=42)
    print(f"--- Running V3.0 Simulation for {p.name} ({n_simulations} iterations) ---")
    
    gains = rng.exponential(scale=1.0, size=(p.n_channels, n_simulations))
    
    # --- Radiometer Noise ---
    channel_centers = np.linspace(p.nu_low.value, p.nu_high.value, p.n_channels) * p.nu_low.unit
    channel_bw = (p.nu_high - p.nu_low) / p.n_channels
    
    sefd_chan = get_sefd(channel_centers, p)
    S_chan_mean = p.S1400_psr_mean * (channel_centers[:, np.newaxis] / (1.4 * u.GHz))**p.psr_spectral_index
    S_chan_inst = S_chan_mean * gains
    
    tau_d_chan = get_scattering_timescale(p, channel_centers, rng, n_simulations)
    W_obs_chan = np.sqrt(p.W_psr_intrinsic_fwhm**2 + tau_d_chan**2)
    
    # Pass all required arguments, including n_pol
    sigma_rad_chan = get_radiometer_noise(S_chan_inst, W_obs_chan, p.P_psr, sefd_chan[:, np.newaxis], p.t_obs, channel_bw, p.n_pol)
    
    # Safer way to handle units in inverse variance sum
    inv_var_sum = np.sum(np.where(np.isfinite(sigma_rad_chan), 1/sigma_rad_chan**2, 0*u.us**-2), axis=0)
    sigma_rad_total = np.where(inv_var_sum > 0, 1 / np.sqrt(inv_var_sum), np.inf * u.us)

    # --- Jitter & Scintle Noise ---
    sigma_jitter_total = get_jitter_noise(p)
    
    nu_center_band = (p.nu_low + p.nu_high) / 2
    tau_d_center = get_scattering_timescale(p, nu_center_band, rng, n_simulations)
    sigma_scatt_total = get_finite_scintle_noise(tau_d_center, p, nu_center_band)
    
    # --- Combine and Return ---
    results_table = Table({
        'radiometer_error': sigma_rad_total,
        'jitter_error': np.repeat(sigma_jitter_total, n_simulations),
        'scintillation_error': sigma_scatt_total,
        'total_white_noise': np.sqrt(sigma_rad_total**2 + sigma_jitter_total**2 + sigma_scatt_total**2)
    }, meta={'name': p.name})
    return TimingNoiseBudget(results_table)

if __name__ == '__main__':
    msp_config = PsrSimConfig(
        name="MSP Sanity Check (P=2ms, S=0.1mJy)",
        P_psr=2 * u.ms,
        W_psr_intrinsic_fwhm=0.06 * (2*u.ms),
        S1400_psr_mean=0.1 * u.mJy,
        DM_psr=30 * u.pc / u.cm**3,
        dist_psr=2.0 * u.kpc,
        psr_spectral_index=-1.6,
        sefd_zenith=0.6 * u.Jy,
        t_obs=1800 * u.s,
        n_pol=2,
        nu_low=0.7 * u.GHz,
        nu_high=2.0 * u.GHz,
        n_channels=32,
        jitter_fJ=1./3.,
        jitter_mI=1.0
    )
    
    noise_budget = run_timing_simulation(msp_config)
    noise_budget.print_summary()