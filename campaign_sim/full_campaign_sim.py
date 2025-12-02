import yaml  
import numpy as np
import pandas as pd
from astropy import units as u
from dataclasses import dataclass
from astroplan import FixedTarget, Observer
from astropy.time import Time
from astropy.table import Table
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
    nu_center:  u.Quantity = 1.4 * u.GHz
    bw_total: u.Quantity = 312.5 * u.MHz
    n_channels: int = 32

    @property
    def nu_low(self):  return self.nu_center - self.bw_total/2
    @property
    def nu_high(self): return self.nu_center + self.bw_total/2
    @property
    def channel_centers(self):
        return np.linspace(self.nu_low, self.nu_high, self.n_channels)
    @property
    def channel_bw(self):
        return self.bw_total / self.n_channels

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
    
# --- SchedulerSpec ----------------------------------------------------
@dataclass
class SchedulerSpec:
    cadence_days: float            # e.g. 7.0
    dwell_min:   float             # minutes per visit

    def build(self, campaign: CampaignConfig):
        n_visits = int((campaign.duration / (self.cadence_days * u.day)).decompose())
        t_visit  = (self.dwell_min * u.min).to(u.s)          # ← absolute minutes
        # sanity guard: total ≤ campaign.total_obs_time
        if n_visits * t_visit > campaign.total_obs_time:
            raise ValueError("Cadence × dwell exceeds campaign total_obs_time")
        epochs   = np.arange(n_visits) * self.cadence_days * u.day
        return epochs, t_visit

@dataclass
class WhiteNoiseModel:
    def __init__(self, obs: Observatory):
        self.obs = obs
        self.rng = np.random.default_rng(42)

    def __call__(self, p, t_obs, elevation):
        return get_full_white_noise_toa_error(self.obs, p, t_obs, elevation, self.rng)

class CampaignRunner:
    def __init__(self, pulsar: PsrSimConfig, campaign: CampaignConfig,
                 observer: Observatory, white_model: WhiteNoiseModel | None = None):
        self.p = pulsar
        self.camp = campaign
        self.obs  = observer
        self.white_model = white_model or WhiteNoiseModel(observer)
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

            # ---- diagnostic print (optional) -----------------------------
            if self.camp.verbose:
                print(f"→ cadence = {spec.cadence_days:4.1f} d, "
                    f"dwell = {spec.dwell_min:4.0f} min "
                    f"({len(epochs)} visits, {t_visit.to(u.min):.0f} each)")

            elevs = self.obs.elevation(tgt, epochs, self.camp.start_time)
            mask  = np.isfinite(elevs)               # drop visits below min elevation
            epochs, elevs = epochs[mask], elevs[mask]

            # red-noise interpolation
            dm_noise   = np.interp(epochs.to(u.day).value,
                                   dm_t.to(u.day).value, dm_series.value)*dm_series.unit
            spin_noise = np.interp(epochs.to(u.day).value,
                                   spin_t.to(u.day).value, spin_series.value)*spin_series.unit
            k_dm = 4.148808e3*u.s*u.MHz**2*u.cm**3/u.pc
            chrom = (k_dm * dm_noise / self.p.nu_center**2).to(u.us)

            # white noise
            white_sigmas = [ self.white_model(self.p, t_visit, el)
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

def get_full_white_noise_toa_error(obs: Observatory, p: PsrSimConfig, t_obs_single: u.Quantity, elevation: u.Quantity, rng):
    """
    Calculates the total white noise TOA error for a single observation.
    This function encapsulates the full multi-channel, multi-noise component model.
    """
    n_mc_internal = 100 # Internal MC loop to average over scintillation states
    
    # ---------- band definition from p -----------------
    channel_centers = p.channel_centers            # numpy array of Quantity shape (n_channels,)
    channel_bw      = p.channel_bw                 # Quantity
    n_channels      = p.n_channels
    nu_center       = p.nu_center
    bw_total        = p.bw_total

    # Number of pulses in a single observation
    N_pulses = (t_obs_single / p.P_psr).decompose().value

    # --- Radiometer Noise ---
    gains = rng.exponential(scale=1.0, size=(n_channels, n_mc_internal))
    S_chan_mean = p.S1400_psr_mean * (channel_centers[:, None] / p.nu_center)**p.psr_spectral_index
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
    dt_iss = 220*u.s * np.sqrt(p.dist_psr.to(u.kpc).value) * nu_center.to(u.GHz).value**(-11/5)
    N_nu = 1 + 0.2 * (bw_total / (1 / (2*np.pi*tau_d_center)))
    N_t = 1 + 0.2 * (t_obs_single / dt_iss) # number of time bins
    sigma_scatt_total = tau_d_center / np.sqrt((N_t * N_nu).decompose()) # scintillation noise
    
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

if __name__ == '__main__':

    # --- 1. Load Pulsar Catalogue ---

    catalogue = yaml.safe_load_all(open("pulsar_classes.yaml"))
    obs  = Observatory("ovro")
    camp = CampaignConfig(duration=5*u.year,
                      total_obs_time=40*u.hour,
                      schedulers=[SchedulerSpec(7, 30),    # 30-min weekly
                                  SchedulerSpec(30.44, 120)])  # 2-h monthly

    white_model_full = WhiteNoiseModel(obs)          # full Monte-Carlo
    white_model_fast = lambda p, t, e: 1.5 * u.us    # constant stand-in

    for src in catalogue:
        p_cfg = config_from_dict(src)

        # choose the model you want to benchmark
        runner = CampaignRunner(p_cfg, camp, obs,
                                white_model=white_model_full, # or white_model_fast
        )  
        
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


    
    
    
    
    
