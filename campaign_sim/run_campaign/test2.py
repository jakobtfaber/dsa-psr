import yaml
import numpy as np
import pandas as pd
from astropy import units as u
from dataclasses import dataclass
from astropy.coordinates import SkyCoord
from astroplan import FixedTarget, Observer
from astropy.coordinates import EarthLocation, AltAz
from astropy.time import Time
from astropy.table import Table
from astropy.utils import iers
iers.conf.auto_download = False
import matplotlib.pyplot as plt
import seaborn as sns
from functools import lru_cache

# --- 1. Dataclass Definitions for Structured I/O ---

@dataclass
class PsrSimConfig:
    """Input parameters for a pulsar timing simulation."""
    name: str
    target_coord: SkyCoord
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
    nu_center: u.Quantity = 1.4 * u.GHz
    bw_total: u.Quantity = 312.5 * u.MHz
    n_channels: int = 32

    @property
    def nu_low(self):
        return self.nu_center - self.bw_total / 2

    @property
    def nu_high(self):
        return self.nu_center + self.bw_total / 2

    @property
    def channel_centers(self):
        return np.linspace(self.nu_low.value, self.nu_high.value, self.n_channels) * self.nu_low.unit

    @property
    def channel_bw(self):
        return self.bw_total / self.n_channels

@dataclass
class CampaignConfig:
    """Parameters for a multi-year timing campaign."""
    duration: u.Quantity
    total_obs_time: u.Quantity
    schedulers: list
    start_time: Time = Time('2025-01-01T00:00:00', scale='utc')
    verbose: bool = False

def config_from_dict(d: dict) -> PsrSimConfig:
    """Factory to build PsrSimConfig from a dictionary."""
    return PsrSimConfig(
        name=d['class_name'],
        target_coord=SkyCoord(d['target_coord'], frame='icrs', unit=(u.hourangle, u.deg)),
        P_psr=d['P_ms'] * u.ms,
        W_psr_intrinsic_fwhm=0.06 * d['P_ms'] * u.ms,
        S1400_psr_mean=d['S_1400_mJy'] * u.mJy,
        DM_psr=d['DM'] * u.pc / u.cm**3,
        dist_psr=d.get('dist_kpc', 2.0) * u.kpc,
        psr_spectral_index=-1.6,
        sefd_zenith=0.6 * u.Jy,
        n_pol=2,
        jitter_fJ=1/3,
        jitter_mI=1,
        sigma_spin_noise=d['sigma_spin_noise_ns'] * u.ns
    )

@dataclass
class Observatory:
    """A class to hold observatory-specific parameters and methods."""
    name: str
    min_elev: u.Quantity = 30 * u.deg

    def __init__(self, name, min_elev=30*u.deg):
        self.name = name
        self.min_elev = min_elev
        self.site = Observer.at_site(name)
        self._elev_cache = {}

    def sefd(self, nu, elev, sefd_zenith):
        T_sky_ref, nu_ref, T_rec = 20*u.K, 0.408*u.GHz, 25*u.K
        T_sky = T_sky_ref * (nu/nu_ref)**-2.75
        T_sys = T_rec + T_sky
        T_sys_14 = T_rec + T_sky_ref*(1.4*u.GHz/nu_ref)**-2.75
        sefd_freq = sefd_zenith * (T_sys/T_sys_14)
        elev_clipped = np.maximum(elev, self.min_elev)
        return sefd_freq / np.sin(elev_clipped)

    def elevation(self, target: FixedTarget, times: u.Quantity, start_time: Time) -> u.Quantity:
        target_name = target.name
        mjd_array = (start_time + times).mjd
        mjd_key = tuple(mjd_array.tolist())
        cache_key = (self.name, target_name, mjd_key)
        if cache_key in self._elev_cache:
            return self._elev_cache[cache_key]
        
        altaz_frame = self.site.altaz(start_time + times, target)
        alts = altaz_frame.alt
        alts_clipped = np.maximum(alts, self.min_elev)
        self._elev_cache[cache_key] = alts_clipped
        return alts_clipped

# --- NEW: Expanded Scheduler Definitions ---
@dataclass
class UniformSchedulerSpec:
    """Scheduler for a uniform, repeating cadence."""
    cadence_days: float
    def build(self, campaign: CampaignConfig):
        n_visits = int((campaign.duration / (self.cadence_days * u.day)).decompose())
        if n_visits == 0: return f"Uniform ({self.cadence_days}d)", np.array([])*u.day, 0*u.s
        t_visit = (campaign.total_obs_time / n_visits).to(u.s)
        epochs = np.arange(n_visits) * self.cadence_days * u.day
        return f"Uniform ({self.cadence_days:.0f}d)", epochs, t_visit

@dataclass
class BlitzSchedulerSpec:
    """Scheduler for intense "blitz" campaigns."""
    blitzes_per_year: int
    days_in_blitz: int
    def build(self, campaign: CampaignConfig):
        n_blitzes = int(campaign.duration.to(u.year).value * self.blitzes_per_year)
        if n_blitzes == 0: return "Blitz", np.array([])*u.day, 0*u.s
        t_obs_per_blitz = campaign.total_obs_time / n_blitzes
        t_visit = t_obs_per_blitz / self.days_in_blitz
        
        all_epochs = []
        for i in range(n_blitzes):
            blitz_start = (i / self.blitzes_per_year * u.year).to(u.day)
            epochs_in_blitz = blitz_start + np.arange(self.days_in_blitz) * u.day
            all_epochs.extend(epochs_in_blitz)
        return f"Blitz ({self.blitzes_per_year}/yr)", u.Quantity(all_epochs), t_visit.to(u.s)
        
@dataclass
class LognormalSchedulerSpec:
    """Scheduler with observations clustered towards the start of the campaign."""
    num_observations: int
    def build(self, campaign: CampaignConfig):
        t_visit = (campaign.total_obs_time / self.num_observations).to(u.s)
        rng = np.random.default_rng(seed=456)
        log_times = rng.lognormal(mean=np.log(100), sigma=1.2, size=self.num_observations)
        epochs = np.sort(log_times / np.max(log_times) * campaign.duration.to(u.day).value) * u.day
        return f"Lognormal ({self.num_observations} obs)", epochs, t_visit


@dataclass
class WhiteNoiseModel:
    def __init__(self, obs: Observatory, n_mc_internal: int = 30, rng_seed: int = 42):
        self.obs = obs
        self.NMC = n_mc_internal
        self.rng = np.random.default_rng(rng_seed)

    def __call__(self, p, t_obs, elevation):
        return get_white_noise_sigma_array(
            self.obs, p, t_obs, elevation,
            rng=self.rng, n_mc_internal=self.NMC
        )

class CampaignRunner:
    def __init__(self, pulsar: PsrSimConfig, campaign: CampaignConfig,
                 observer: Observatory, white_model: WhiteNoiseModel | None = None):
        self.p = pulsar
        self.camp = campaign
        self.obs = observer
        self.white_model = white_model or WhiteNoiseModel(observer)
        self.rng = np.random.default_rng(123)

    def run(self):
        tgt = FixedTarget(coord=self.p.target_coord)
        dm_sigma = dm_rms_kolmogorov(self.camp.duration)
        dm_t, dm_series = generate_red_noise_series(self.camp, dm_sigma, 8/3, self.rng)
        spin_t, spin_series = generate_red_noise_series(self.camp,
                                   self.p.sigma_spin_noise, 4.0, self.rng)

        results = {}
        for spec in self.camp.schedulers:
            try:
                name, epochs, t_visit = spec.build(self.camp)
            except ValueError as e:
                print(f'Error building schedule: {e}')
                continue

            if self.camp.verbose:
                print(f'→ strategy = {name} '
                      f'({len(epochs)} visits, {t_visit.to(u.min):.0f} each)')

            loc = EarthLocation.of_site(self.obs.name)
            altaz_frame = AltAz(obstime=self.camp.start_time + epochs, location=loc)
            elevs = tgt.coord.transform_to(altaz_frame).alt
            mask = elevs > self.obs.min_elev
            epochs, elevs = epochs[mask], elevs[mask]

            dm_noise = np.interp(epochs.to(u.day).value,
                                 dm_t.to(u.day).value, dm_series.value) * dm_series.unit
            spin_noise = np.interp(epochs.to(u.day).value,
                                   spin_t.to(u.day).value, spin_series.value) * spin_series.unit
            k_dm = 4.148808e3 * u.s * u.MHz**2 * u.cm**3 / u.pc
            chrom = (k_dm * dm_noise / self.p.nu_center**2).to(u.us)

            white_sigmas = get_white_noise_sigma_array(
                self.obs, self.p, t_visit, elevs, self.rng
            )
            white_draws = self.rng.normal(0, white_sigmas.value) * u.us

            raw_toas = chrom + spin_noise + white_draws
            tab = Table({
                'epoch_day': epochs,
                'total_toa_raw': raw_toas,
                'white_noise_sigma': white_sigmas
            })
            results[name] = analyze_residuals(tab)
        return results

# --- Physics Helper Functions ---

def dm_rms_kolmogorov(T: u.Quantity, C_DM: u.Quantity = 3e-6 * (u.pc/u.cm**3)**2 / u.yr**(5/3)):
    T_yr = T.to(u.yr)
    return 0.5 * np.sqrt(C_DM) * T_yr**(5/6)

def get_scattering_timescale(p: PsrSimConfig, nu, rng, n_simulations=1):
    scatter_dex = 0.65
    log_tau = (
        -6.46
        + 0.154 * np.log10(p.DM_psr.value)
        + 1.07 * np.log10(p.DM_psr.value)**2
        - 3.86 * np.log10(nu.to(u.GHz).value)
    )
    arr = np.asarray(log_tau)
    size = n_simulations if np.ndim(nu.value) == 0 else (nu.size, n_simulations)
    stochastic = rng.normal(arr[..., None], scatter_dex, size=size)
    return (10**stochastic * u.ms).to(u.s)

def get_white_noise_sigma_array(obs: Observatory, p: PsrSimConfig, t_obs: u.Quantity, elevations: u.Quantity, rng: np.random.Generator, n_mc_internal: int = 100) -> u.Quantity:
    n_ch = p.n_channels
    n_vis = elevations.size
    gains = rng.exponential(1.0, size=(n_ch, n_mc_internal, n_vis))
    S_mean = p.S1400_psr_mean * (p.channel_centers[:, None, None] / p.nu_center)**p.psr_spectral_index
    S_inst = S_mean * gains
    tau_d_chan = get_scattering_timescale(p, p.channel_centers, rng, n_mc_internal)
    W_obs = np.sqrt(p.W_psr_intrinsic_fwhm**2 + tau_d_chan[:, :, None]**2)
    sefd_chan = obs.sefd(p.channel_centers[:, None], elevations[None, :], p.sefd_zenith)
    sefd_chan = sefd_chan[:, None, :]
    
    duty = (W_obs / p.P_psr).decompose()
    S_peak = np.where(duty > 0, S_inst / duty, 0 * u.Jy)
    snr = np.where(S_peak > 0 * u.Jy,
                   S_peak / sefd_chan * np.sqrt(p.n_pol * p.channel_bw * t_obs),
                   np.inf*u.dimensionless_unscaled)
    
    # Correct radiometer error calculation
    N_pulses = (t_obs / p.P_psr).decompose().value
    sigma_sq = (W_obs / (snr * np.sqrt(N_pulses)))**2
    
    inv_var = np.where(np.isfinite(sigma_sq), 1 / sigma_sq, 0 * u.us**-2)
    inv_sum = np.sum(inv_var, axis=0)
    sigma_rad = np.where(inv_sum > 0, 1 / np.sqrt(inv_sum), np.inf * u.us)
    sigma_rad_mean = np.mean(sigma_rad.value, axis=0) * u.us
    
    sigma_jit = ((p.jitter_fJ * p.W_psr_intrinsic_fwhm * np.sqrt(1 + p.jitter_mI**2))
                 / (2 * np.sqrt(2 * N_pulses * np.log(2)))).to(u.us)
                 
    tau_center = get_scattering_timescale(p, p.nu_center, rng, n_mc_internal)
    dt_iss = 220 * u.s * np.sqrt(p.dist_psr.to(u.kpc).value) * (p.nu_center.to(u.GHz).value)**(-11/5)
    N_t = 1 + 0.2 * (t_obs / dt_iss)
    N_nu = 1 + 0.2 * (p.bw_total / (1 / (2 * np.pi * tau_center)))
    sigma_sc = tau_center / np.sqrt((N_t * N_nu).decompose())
    sigma_sc_mean = np.mean(sigma_sc.value) * u.us
    
    return np.sqrt(sigma_rad_mean**2 + sigma_jit**2 + sigma_sc_mean**2)


# Global cache for red noise to avoid re-computation
_red_noise_cache: dict[
    tuple[float, float, str, float],
    tuple[u.Quantity, u.Quantity]
] = {}

def generate_red_noise_series(campaign, rms_amplitude: u.Quantity, gamma: float, rng: np.random.Generator):
    """
    Generates a realistic time series of red noise with a power-law spectrum.
    
    This function creates colored noise with a spectrum S(f) ~ f^-gamma. It includes
    caching to speed up repeated calls with the same parameters.
    """
    # Build a hashable key from span, amplitude.value, amplitude.unit, and gamma
    span_key = (
        campaign.duration.to(u.day).value,
        rms_amplitude.value,
        str(rms_amplitude.unit),
        gamma
    )
    # Check cache first
    try:
        return _red_noise_cache[span_key]
    except KeyError:
        n_days = int(span_key[0])
        times = np.arange(n_days) * u.day
        freqs = np.fft.rfftfreq(n_days, d=1) # Frequencies in 1/day
        
        # Define the power-law spectrum
        psd = np.where(freqs > 0, freqs**(-gamma), 0)
        
        # Generate random phases and amplitudes in the Fourier domain
        series = rng.normal(size=freqs.size) + 1j * rng.normal(size=freqs.size)
        
        # Create the noise in the Fourier domain and inverse transform
        ts_raw = np.fft.irfft(series * np.sqrt(psd), n=n_days)
        
        # Scale the time series to the desired physical RMS amplitude
        ts_unitstd = ts_raw / np.std(ts_raw)
        ts_scaled = ts_unitstd * rms_amplitude
        
        # Store in cache and return
        _red_noise_cache[span_key] = (times, ts_scaled)
        return times, ts_scaled

def analyze_residuals(results_table: Table):
    """
    Performs a timing fit for P and P-dot on the raw TOAs and calculates
    the post-fit residuals. It also adds a metric to check for phase connection.
    """
    # Ensure there are enough data points to fit a 3-parameter model
    if len(results_table) < 3:
        results_table['post_fit_residual'] = results_table['total_toa_raw']
        results_table.meta['phase_connected'] = False
        return results_table

    epochs = results_table['epoch_day'].quantity.to(u.year).value
    toas = results_table['total_toa_raw'].quantity.to(u.s).value
    weights = 1.0 / results_table['white_noise_sigma'].quantity.to(u.s).value**2

    # Design matrix for a quadratic fit (offset, f, fdot)
    M = np.vander(epochs, 3, increasing=True)
    W = np.diag(weights)

    try:
        # Perform weighted least-squares fit
        M_T_W = M.T @ W
        cov_matrix = np.linalg.inv(M_T_W @ M)
        best_fit_params = cov_matrix @ M_T_W @ toas
        
        # Subtract the best-fit model from the raw TOAs
        model_toas = (M @ best_fit_params) * u.s
        post_fit_residuals = results_table['total_toa_raw'] - model_toas.to(u.us)

        # --- Phase Connection Metric ---
        # The covariance matrix is for fit parameters [c0, c1, c2] where t is in years.
        # var(f) is related to var(c1) [units: s^2/yr^2]
        # var(fdot) is related to var(c2) [units: s^2/yr^4]
        var_f_yr = cov_matrix[1, 1] * (u.s/u.year)**2
        var_fdot_yr = cov_matrix[2, 2] * (u.s/u.year**2)**2
        
        # Time gaps between observations in seconds
        dt_s = np.diff(results_table['epoch_day'].quantity.to(u.s))
        
        # Propagate uncertainty in phase. sigma_phi^2 ~ (dt * sigma_f)^2
        # Convert variances to 1/s^2 and 1/s^4
        sigma_f_s = np.sqrt(var_f_yr.to(u.s**-2))
        sigma_fdot_s = np.sqrt(var_fdot_yr.to(u.s**-4))
        
        phase_err_sq_f = (dt_s * sigma_f_s)**2
        phase_err_sq_fdot = (0.5 * dt_s**2 * sigma_fdot_s)**2
        
        # Maximum phase uncertainty in units of full rotations (turns)
        max_phase_uncertainty_turns = np.sqrt(np.max(phase_err_sq_f + phase_err_sq_fdot).value) / (2 * np.pi)

        results_table.meta['phase_connected'] = max_phase_uncertainty_turns < 0.5
        
    except np.linalg.LinAlgError:
        # If the fit fails (e.g., singular matrix), timing is not connected
        post_fit_residuals = results_table['total_toa_raw']
        results_table.meta['phase_connected'] = False

    results_table['post_fit_residual'] = post_fit_residuals
    return results_table

# --- Main Execution Block ---

if __name__ == '__main__':
    with open('pulsar_catalog.yaml') as f:
        raw_docs = yaml.safe_load_all(f)
        catalogue = [doc for doc in raw_docs if isinstance(doc, dict)]

    obs = Observatory('ovro')
    
    # --- NEW: Expanded set of schedulers ---
    campaign_schedulers = [
        UniformSchedulerSpec(cadence_days=7),      # Weekly, 30 min
        UniformSchedulerSpec(cadence_days=14),     # Bi-weekly, 30 min
        UniformSchedulerSpec(cadence_days=30.44),  # Monthly, 60 min
        LognormalSchedulerSpec(num_observations=150),
        BlitzSchedulerSpec(blitzes_per_year=2, days_in_blitz=10),
    ]

    camp = CampaignConfig(
        duration=5 * u.year,
        total_obs_time=100 * u.hour,
        schedulers=campaign_schedulers,
        verbose=True
    )

    all_results = []
    for src in catalogue:
        p_cfg = config_from_dict(src)
        runner = CampaignRunner(p_cfg, camp, obs)
        res = runner.run()
        
        for name, table in res.items():
            all_results.append({
                "Pulsar Type": p_cfg.name,
                "Strategy": name,
                "Post-fit RMS (us)": table['post_fit_residual'].quantity.std().to(u.us).value,
                "Phase Connected": table.meta.get('phase_connected', False)
            })

    # --- Final Heatmap Visualization ---
    df_summary = pd.DataFrame(all_results)
    pivot_table = df_summary.pivot(index="Pulsar Type", columns="Strategy", values="Post-fit RMS (us)")
    
    annot_data = pivot_table.copy().astype(str)
    for r_idx, psr_type in enumerate(pivot_table.index):
        for c_idx, strat in enumerate(pivot_table.columns):
            is_connected = df_summary[
                (df_summary['Pulsar Type'] == psr_type) & (df_summary['Strategy'] == strat)
            ]['Phase Connected'].iloc[0]
            
            val = pivot_table.iloc[r_idx, c_idx]
            check_mark = "✓" if is_connected else "✗"
            annot_data.iloc[r_idx, c_idx] = f"{val:.2f}\n{check_mark}"
            
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_table, annot=annot_data, fmt="s", cmap="viridis_r", linewidths=.5,
                cbar_kws={'label': 'Post-fit Timing RMS (µs)'})
    plt.title("Optimal Cadence Strategy by Pulsar Type for DSA-2000", fontsize=16)
    plt.ylabel("Pulsar Archetype", fontsize=12)
    plt.xlabel("Observing Strategy", fontsize=12)
    plt.tight_layout()
    plt.show()
