import os
import sys
import yaml  
import numpy as np
import pandas as pd
from astropy import units as u
from dataclasses import dataclass, field, replace
from astroplan import FixedTarget, Observer
from astropy.time import Time
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
    sigma_spin_noise: u.Quantity # Amplitude of intrinsic spin noise

@dataclass
class CampaignConfig:
    """Parameters for a multi-year timing campaign."""
    duration: u.Quantity
    total_obs_time: u.Quantity
    schedulers: list
    start_time: Time = Time("2025-01-01T00:00:00", scale="utc")

# ---------- Factory to build PsrSimConfig from a metadata dict ----------
def config_from_dict(d: dict) -> PsrSimConfig:
    """
    Convert one catalogue row (keys listed below) into PsrSimConfig.
    Required keys: class_name, P_ms, DM, S_1400_mJy, sigma_spin_noise_ns
    Optional keys can be added with d.get().
    """
    return PsrSimConfig(
        name=d['class_name'],
        P_psr=d['P_ms'] * u.ms,
        W_psr_intrinsic_fwhm=0.06 * d['P_ms'] * u.ms,
        S1400_psr_mean=d['S_1400_mJy'] * u.mJy,
        DM_psr=d['DM'] * u.pc / u.cm**3,
        dist_psr=2 * u.kpc,                 # default
        psr_spectral_index=-1.6,
        sefd_zenith=0.6 * u.Jy,
        n_pol=2,
        jitter_fJ=1/3,
        jitter_mI=1,
        sigma_spin_noise=d['sigma_spin_noise_ns'] * u.ns
    )

@dataclass
class Observatory:
    name: str                 # e.g. "ovro"
    min_elev: u.Quantity = 30 * u.deg

    def __post_init__(self):
        self.site = Observer.at_site(self.name)

    def sefd(self, nu, elev, sefd_zenith):
        """Equation from earlier helper."""
        T_sky_ref, nu_ref, T_rec = 20*u.K, 0.408*u.GHz, 25*u.K
        T_sky = T_sky_ref * (nu/nu_ref)**-2.75
        T_sys = T_rec + T_sky
        T_sys_14 = T_rec + T_sky_ref*(1.4*u.GHz/nu_ref)**-2.75
        sefd_freq = sefd_zenith * (T_sys/T_sys_14)
        elev_clipped = np.maximum(elev, self.min_elev)
        return sefd_freq/np.sin(elev_clipped)

    def elevation(self, target: FixedTarget,
                  times: u.Quantity,
                  start_time: Time) -> u.Quantity:
        t = start_time + times
        return self.site.altaz(t, target).alt
    
@dataclass
class SchedulerSpec:
    cadence_days: float      # e.g. 7.0
    dwell_frac: float        # fraction of total_obs_time per visit

    def build(self, campaign: CampaignConfig):
        n_visits = int((campaign.duration / (self.cadence_days*u.day)).decompose())
        t_visit  = (campaign.total_obs_time * self.dwell_frac / n_visits).to(u.s)
        epochs   = np.arange(n_visits) * self.cadence_days * u.day
        return epochs, t_visit

@dataclass
class WhiteNoiseModel:
    def __init__(self, obs: Observatory):
        self.obs = obs
        self.rng = np.random.default_rng(42)

    def __call__(self, p, t_obs, nu_center, elevation):
        return get_full_white_noise_toa_error(p, t_obs, elevation, self.rng)

class CampaignRunner:
    def __init__(self, pulsar: PsrSimConfig, campaign: CampaignConfig,
                 observer: Observatory):
        self.p = pulsar
        self.camp = campaign
        self.obs  = observer
        self.white_model = WhiteNoiseModel(observer)
        self.rng = np.random.default_rng(123)
        
    def run(self):
        tgt = FixedTarget.from_name(self.p.name)  # or use coord
        dm_sigma = dm_rms_kolmogorov(self.camp.duration)
        dm_t, dm_series = generate_red_noise_series(self.camp, dm_sigma, 8/3, self.rng)
        spin_t, spin_series = generate_red_noise_series(self.camp,
                                   self.p.sigma_spin_noise, 4.0, self.rng)

        results = {}
        for spec in self.camp.schedulers:        # list of SchedulerSpec
            epochs, t_visit = spec.build(self.camp)
            elevs = self.obs.elevation(tgt, epochs, self.camp.start_time)
            good = np.isfinite(elevs)            # mask invisible epochs
            epochs, elevs = epochs[good], elevs[good]

            # red-noise interpolation
            dm_noise   = np.interp(epochs.to(u.day).value,
                                   dm_t.to(u.day).value, dm_series.value)*dm_series.unit
            spin_noise = np.interp(epochs.to(u.day).value,
                                   spin_t.to(u.day).value, spin_series.value)*spin_series.unit
            k_dm = 4.148808e3*u.s*u.MHz**2*u.cm**3/u.pc
            chrom  = (k_dm*dm_noise/(1.4*u.GHz)**2).to(u.us)

            # white noise
            white_sigmas = [ self.white_model(self.p, t_visit, 1.4*u.GHz, el)
                 for el in elevs ]
            white_draws  = self.rng.normal(0, [s.value for s in white_sigmas]) * u.us

            raw_toas = chrom + spin_noise + white_draws
            tab = Table({'epoch_day': epochs, 'total_toa_raw': raw_toas,
                         'white_noise_sigma': white_sigmas})
            results[ f"cad{spec.cadence_days}d" ] = analyze_residuals(tab)
        return results

# --- 2. Physics Helper Functions (from Validated White Noise Model) ---

def dm_rms_kolmogorov(T: u.Quantity, C_DM=3e-6*u.pc/u.cm**3):
    """
    Kolmogorov DM rms over span T (Lam+16).
    Default C_DM ~ 3e-6 pc/cm^3 for DM~30 pc/cm^3 lines of sight.
    """
    return 0.5 * np.sqrt(C_DM) * T.to(u.yr)**(5/6)

def get_scattering_timescale(p: PsrSimConfig, nu, rng, n_simulations=1):
    scatter_dex = 0.65
    log_tau_d_ms_mean = -6.46 + 0.154*np.log10(p.DM_psr.value) + 1.07*(np.log10(p.DM_psr.value))**2 - 3.86*np.log10(nu.to(u.GHz).value)
    log_tau_d_ms_mean_arr = np.asarray(log_tau_d_ms_mean)
    size = (len(nu), n_simulations) if hasattr(nu, "__len__") else n_simulations
    stochastic_log_tau_ms = rng.normal(log_tau_d_ms_mean_arr[..., np.newaxis], scatter_dex, size=size)
    return (10**stochastic_log_tau_ms * u.ms).to(u.s)

def get_full_white_noise_toa_error(p: PsrSimConfig, t_obs_single: u.Quantity, elevation: u.Quantity, rng):
    """
    Calculates the total white noise TOA error for a single observation.
    This function encapsulates the full multi-channel, multi-noise component model.
    """
    n_mc_internal = 100 # Internal MC loop to average over scintillation states
    
    # Define observing band based on a typical 1.4 GHz DSA-2000 setup
    bw_total = 312.5 * u.MHz
    nu_center = 1.4 * u.GHz
    nu_low, nu_high = nu_center - bw_total/2, nu_center + bw_total/2
    n_channels = 32
    channel_centers = np.linspace(nu_low, nu_high, n_channels)
    channel_bw = (nu_high - nu_low) / n_channels

    # Number of pulses in a single observation
    N_pulses = (t_obs_single / p.P_psr).decompose().value

    # --- Radiometer Noise ---
    gains = rng.exponential(scale=1.0, size=(n_channels, n_mc_internal))
    S_chan_mean = p.S1400_psr_mean * (channel_centers[:, np.newaxis] / (1.4 * u.GHz))**p.psr_spectral_index
    S_chan_inst = S_chan_mean * gains
    tau_d_chan = get_scattering_timescale(p, channel_centers, rng, n_mc_internal)
    W_obs_chan = np.sqrt(p.W_psr_intrinsic_fwhm**2 + tau_d_chan**2)
    sefd_chan = obs.sefd(channel_centers, elevation, p.sefd_zenith)
    
    duty_cycle = (W_obs_chan / p.P_psr).decompose()
    S_peak = np.where(duty_cycle > 0, S_chan_inst / duty_cycle, 0 * u.Jy)
    snr_profile = np.where(S_peak > 0*u.Jy, S_peak / sefd_chan[:, np.newaxis] * np.sqrt(p.n_pol * channel_bw * t_obs_single), np.inf*u.dimensionless_unscaled)
    sigma_rad_chan_sq = (W_obs_chan / (snr_profile * np.sqrt(N_pulses)))**2
    
    inv_var_sum = np.sum(np.where(np.isfinite(sigma_rad_chan_sq), 1/sigma_rad_chan_sq, 0*u.us**-2), axis=0)
    sigma_rad_total = np.where(inv_var_sum > 0, 1 / np.sqrt(inv_var_sum), np.inf * u.us)

    # --- Jitter Noise ---
    sigma_jitter_total = (p.jitter_fJ * p.W_psr_intrinsic_fwhm * np.sqrt(1 + p.jitter_mI**2)) / (2 * np.sqrt(2 * N_pulses * np.log(2)))

    # --- Scintillation Noise ---
    tau_d_center = get_scattering_timescale(p, nu_center, rng, n_mc_internal)
    dt_iss = 220 * u.s * np.sqrt(p.dist_psr.to(u.kpc).value) * (nu_center.to(u.GHz).value)**(-11./5.)
    N_t = 1 + 0.2 * (t_obs_single / dt_iss)
    N_nu = 1 + 0.2 * (bw_total / (1 / (2 * np.pi * tau_d_center)))
    sigma_scatt_total = tau_d_center / np.sqrt((N_t * N_nu).decompose())
    
    # --- Total White Noise ---
    sigma_white_total = np.sqrt(sigma_rad_total**2 + sigma_jitter_total**2 + sigma_scatt_total.mean()**2)
    
    # Return the mean of the distribution as the representative error for this observation
    return sigma_white_total.mean()


def generate_red_noise_series(campaign: CampaignConfig, rms_amplitude: u.Quantity, gamma: float, rng):
    """Generic function to generate a red noise time series with a power-law spectrum."""
    n_days = int(campaign.duration.to(u.day).value)
    times = np.arange(n_days) * u.day
    freqs = np.fft.rfftfreq(n_days, d=1)
    # The absolute PSD normalisation is arbitrary because we rescale the
    # time series to the requested rms_amplitude further below.
    psd = np.where(freqs > 0, freqs**(-gamma), 0)
    real_part = rng.normal(size=len(freqs)) * np.sqrt(psd)
    imag_part = rng.normal(size=len(freqs)) * np.sqrt(psd)
    fourier_noise = real_part + 1j * imag_part
    time_series_raw = np.fft.irfft(fourier_noise, n=n_days)
    time_series_scaled = (time_series_raw / np.std(time_series_raw)) * rms_amplitude
    return times, time_series_scaled

# --- 4. Campaign Scheduler Definitions ---

def generate_high_cadence_schedule(campaign: CampaignConfig):
    n_obs = int((campaign.duration / (7 * u.day)).decompose())
    t_obs_single = (campaign.total_obs_time / n_obs).to(u.s)
    obs_epochs = np.arange(n_obs) * 7 * u.day
    elevations = 60 + 30 * np.sin(2 * np.pi * obs_epochs.to(u.day).value / 365.25) * u.deg
    # Site-based visibility mask (optional)
    site_lat = 37.2 * u.deg          # e.g. Owens Valley
    target_dec = 20.0 * u.deg        # <-- replace with each pulsar’s decl.
    max_hour_angle = np.arccos(
            (np.sin(30*u.deg) - np.sin(site_lat)*np.sin(target_dec)) /
            (np.cos(site_lat)*np.cos(target_dec))) * u.rad
    # mask epochs that fall outside ±HA_max
    ha = (obs_epochs.to(u.day) * 2*np.pi/0.9972696) % (2*np.pi) * u.rad   # crude sidereal HA
    elevations[ np.abs(ha) > max_hour_angle ] = np.nan*u.deg
    return "High Cadence (Weekly)", obs_epochs, t_obs_single, elevations

def generate_high_sensitivity_schedule(campaign: CampaignConfig):
    n_obs = int((campaign.duration / (30.44 * u.day)).decompose())
    t_obs_single = (campaign.total_obs_time / n_obs).to(u.s)
    obs_epochs = np.arange(n_obs) * 30.44 * u.day
    elevations = 60 + 30 * np.sin(2 * np.pi * obs_epochs.to(u.day).value / 365.25) * u.deg
    # Site-based visibility mask (optional)
    site_lat = 37.2 * u.deg          # e.g. Owens Valley
    target_dec = 20.0 * u.deg        # <-- replace with each pulsar’s decl.
    max_hour_angle = np.arccos(
            (np.sin(30*u.deg) - np.sin(site_lat)*np.sin(target_dec)) /
            (np.cos(site_lat)*np.cos(target_dec))) * u.rad
    # mask epochs that fall outside ±HA_max
    ha = (obs_epochs.to(u.day) * 2*np.pi/0.9972696) % (2*np.pi) * u.rad   # crude sidereal HA
    elevations[ np.abs(ha) > max_hour_angle ] = np.nan*u.deg
    return "High Sensitivity (Monthly)", obs_epochs, t_obs_single, elevations

def generate_hybrid_schedule(campaign: CampaignConfig):
    """
    Weekly observations while the pulsar is above 60° elevation,
    monthly otherwise.
    """
    n_weeks = int((campaign.duration / (7*u.day)).decompose())
    n_months = int((campaign.duration / (30.44*u.day)).decompose())
    
    weekly_epochs  = np.arange(n_weeks)  * 7*u.day
    monthly_epochs = np.arange(n_months) * 30.44*u.day
    
    # visibility test
    elev_weekly = 60 + 30*np.sin(2*np.pi*weekly_epochs.to(u.day).value/365.25)*u.deg
    keep = elev_weekly > 60*u.deg
    obs_epochs = np.concatenate([weekly_epochs[keep], monthly_epochs])
    obs_epochs.sort()
    
    n_obs = len(obs_epochs)
    t_obs_single = (campaign.total_obs_time / n_obs).to(u.s)
    elevations = 60 + 30*np.sin(2*np.pi*obs_epochs.to(u.day).value/365.25)*u.deg
    return "Hybrid (Zenith-weighted)", obs_epochs, t_obs_single, elevations

# --- 5. Sophisticated Analysis: Timing Model Fit ---

def analyze_residuals(results_table: Table):
    epochs = results_table['epoch_day'].quantity.to(u.year).value
    toas = results_table['total_toa_raw'].quantity.to(u.s).value
    weights = 1.0 / results_table['white_noise_sigma'].quantity.to(u.s).value**2
    
    # Design matrix for a P, P-dot model (offset, f, fdot)
    M = np.vander(epochs, 3, increasing=True)
    W = np.diag(weights)
    
    try:
        M_T_W = M.T @ W
        cov_matrix = np.linalg.inv(M_T_W @ M)
        best_fit_params = cov_matrix @ M_T_W @ toas
        model_toas = (M @ best_fit_params) * u.s
        post_fit_residuals = results_table['total_toa_raw'] - model_toas.to(u.us)
    except np.linalg.LinAlgError:
        post_fit_residuals = results_table['total_toa_raw']

    results_table['post_fit_residual'] = post_fit_residuals
    return results_table

# --- 6. Main Campaign Simulation ---

def run_campaign_simulation(p: PsrSimConfig, campaign: CampaignConfig):
    rng = np.random.default_rng(seed=123)
    print(f"--- Simulating 5-Year Campaign for {p.name} ---")
    
    dm_sigma = dm_rms_kolmogorov(campaign.duration)
    print(f"DM sigma (want to be close to 5e-5 pc/cm^3): {dm_sigma}")
    dm_times, dm_series = generate_red_noise_series(campaign, dm_sigma, 8/3, rng)
    spin_times, spin_series = generate_red_noise_series(campaign, p.sigma_spin_noise, 4.0, rng)
    
    all_campaign_results = {}

    for scheduler_func in campaign.schedulers:
        strategy_name, obs_epochs, t_obs_single, elevations = scheduler_func(campaign)
        print(f"\nRunning strategy: '{strategy_name}' ({len(obs_epochs)} observations of {t_obs_single:.0f})")
        
        dm_noise = np.interp(obs_epochs.to(u.day).value, dm_times.to(u.day).value, dm_series.value) * dm_series.unit
        spin_noise = np.interp(obs_epochs.to(u.day).value, spin_times.to(u.day).value, spin_series.value) * spin_series.unit
        
        k_dm = 4.148808e3 * u.s * u.MHz**2 * u.cm**3 / u.pc
        chromatic_error = (k_dm * dm_noise / (1.4*u.GHz)**2).to(u.us)
        
        white_noise_sigma = np.array(
            [get_full_white_noise_toa_error(p, t_obs_single, el, rng) for el in elevations]) #white noise is in us here
        white_noise_draws = rng.normal(0, white_noise_sigma.value) * u.us
        
        final_raw_toas = chromatic_error + spin_noise + white_noise_draws
        
        table = Table({'epoch_day': obs_epochs, 'total_toa_raw': final_raw_toas, 'white_noise_sigma': white_noise_sigma})
        all_campaign_results[strategy_name] = analyze_residuals(table)
        
    return all_campaign_results

if __name__ == '__main__':

    # --- OLD METHOD FOR RUNNING A SINGLE PULSAR ---
    #msp_config = PsrSimConfig(
    #    name="Typical MSP", P_psr=2*u.ms, W_psr_intrinsic_fwhm=0.06*(2*u.ms),
    #    S1400_psr_mean=0.1*u.mJy, DM_psr=30*u.pc/u.cm**3, dist_psr=2.0*u.kpc,
    #    psr_spectral_index=-1.6, sefd_zenith=0.6*u.Jy, n_pol=2,
    #    jitter_fJ=1./3., jitter_mI=1.0, sigma_spin_noise=100*u.ns
    #)

    #campaign_config = CampaignConfig(
    #    duration=5*u.year, total_obs_time=40*u.hour,
    #    schedulers=[generate_high_cadence_schedule, generate_high_sensitivity_schedule, generate_hybrid_schedule]
    #)

    #campaign_results = run_campaign_simulation(msp_config, campaign_config)

    #fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    #summary_data = []
    #for name, table in campaign_results.items():
    #    post_fit_rms = table['post_fit_residual'].std()
    #    summary_data.append({'Strategy': name, 'Post-fit RMS (µs)': post_fit_rms.to(u.us).value})
    #    axes[0].plot(table['epoch_day'], table['total_toa_raw'], 'o', ms=4, alpha=0.6, label=f"{name} (Raw RMS: {table['total_toa_raw'].std():.2f})")
    #    axes[1].plot(table['epoch_day'], table['post_fit_residual'], 'o', ms=4, alpha=0.7, label=f"{name} (Post-fit RMS: {post_fit_rms:.2f})")

    #axes[0].set_title(f"Simulated Raw TOA Residuals for {msp_config.name}")
    #axes[0].set_ylabel("Raw TOA Residual (µs)")
    #axes[0].legend(); axes[0].grid(True, alpha=0.3)
    #
    #axes[1].axhline(0, color='k', linestyle='--', alpha=0.5)
    #axes[1].set_title("Post-Fit Timing Residuals (P and P-dot removed)")
    #axes[1].set_xlabel("Time (days)"); axes[1].set_ylabel("Post-Fit TOA Residual (µs)")
    #axes[1].legend(); axes[1].grid(True, alpha=0.3)
    #
    #plt.tight_layout(); plt.show()
    #
    #summary_df = pd.DataFrame(summary_data)
    #print("\n--- Campaign Strategy Comparison ---")
    #print(summary_df.to_markdown(index=False))

    #plt.figure(figsize=(6,3))
    #sns.heatmap(summary_df.set_index('Strategy')[['Post-fit RMS (µs)']],
    #            annot=True, fmt=".2f", cmap="viridis")
    #plt.title("Post-fit RMS by Strategy")
    #plt.ylabel("")
    #plt.show()

    catalogue = yaml.safe_load_all(open("pulsar_classes.yaml"))
    obs  = Observatory("ovro")
    camp = CampaignConfig(duration=5*u.year, total_obs_time=40*u.hour,
                        schedulers=[SchedulerSpec(7, 1.0), SchedulerSpec(30.44, 1.0)])

    for src in catalogue:
        p_cfg = config_from_dict(src)
        runner = CampaignRunner(p_cfg, camp, obs)
        res = runner.run()

        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        summary_data = []
        for name, table in res.items():
            post_fit_rms = table['post_fit_residual'].std()
            summary_data.append({'Strategy': name, 'Post-fit RMS (µs)': post_fit_rms.to(u.us).value})
            axes[0].plot(table['epoch_day'], table['total_toa_raw'], 'o', ms=4, alpha=0.6, label=f"{name} (Raw RMS: {table['total_toa_raw'].std():.2f})")
            axes[1].plot(table['epoch_day'], table['post_fit_residual'], 'o', ms=4, alpha=0.7, label=f"{name} (Post-fit RMS: {post_fit_rms:.2f})")

        axes[0].set_title(f"Simulated Raw TOA Residuals for {p_cfg.name}")
        axes[0].set_ylabel("Raw TOA Residual (µs)")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        
        axes[1].axhline(0, color='k', linestyle='--', alpha=0.5)
        axes[1].set_title("Post-Fit Timing Residuals (P and P-dot removed)")
        axes[1].set_xlabel("Time (days)"); axes[1].set_ylabel("Post-Fit TOA Residual (µs)")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout(); plt.show()
        
        summary_df = pd.DataFrame(summary_data)
        print("\n--- Campaign Strategy Comparison ---")
        print(summary_df.to_markdown(index=False))

        summary_rows = []
        for strat, tab in res.items():
            summary_rows.append({
            "Source": p_cfg.name,
            "Strategy": strat,
            "RMS_postfit": tab['post_fit_residual'].std().to(u.us).value
            })
        df_sum = pd.DataFrame(summary_rows)
        sns.heatmap(df_sum.pivot("Source","Strategy","RMS_postfit"),
                    annot=True, fmt=".2f", cmap="viridis")
        plt.show()


    
    
    
    
    
    