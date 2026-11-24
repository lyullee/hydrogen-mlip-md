"""Unit-explicit estimators; input positions must be unwrapped Cartesian Angstrom."""
import numpy as np


def diffusion(positions, time_fs, symbols, masses, fit_fraction=(0.2, 0.5), max_lag=1000):
    r = np.asarray(positions, dtype=float)
    t = np.asarray(time_fs, dtype=float)
    symbols = np.asarray(symbols)
    masses = np.asarray(masses, dtype=float)
    if r.ndim != 3 or r.shape[2] != 3 or r.shape[:2] != (len(t), len(symbols)):
        raise ValueError("positions must have shape (frames, atoms, 3).")
    if masses.shape != symbols.shape or not np.all(np.isfinite(masses)) or np.any(masses <= 0):
        raise ValueError("Positive finite atomic masses are required.")
    if len(t) < 20 or not np.isfinite(r).all() or not np.isfinite(t).all():
        raise ValueError("At least 20 finite, uniformly spaced frames are required.")
    dt = np.diff(t)
    if np.any(dt <= 0) or not np.allclose(dt, dt[0], rtol=1e-5, atol=1e-8):
        raise ValueError("Time must be increasing and uniformly spaced in fs.")
    h = symbols == "H"
    host = ~h
    if not h.any() or not host.any():
        raise ValueError("Both hydrogen and host atoms are required.")
    lo, hi = fit_fraction
    if not 0 < lo < hi <= 0.5:
        raise ValueError("Fit fractions must satisfy 0 < start < end <= 0.5.")
    if not isinstance(max_lag, int) or max_lag < 10:
        raise ValueError("max_lag must be an integer >= 10.")
    # Remove host mass-weighted center-of-mass motion before measuring H transport.
    com = np.average(r[:, host, :], weights=masses[host], axis=1)
    rh = r[:, h, :] - com[:, None, :]
    limit = min(len(t) // 2, max_lag)
    lags = np.arange(1, limit + 1)
    xyz = np.array([np.mean((rh[k:] - rh[:-k]) ** 2, axis=(0, 1)) for k in lags])
    msd = xyz.sum(axis=1)
    lag_fs = lags * dt[0]
    # Fractions refer to total production duration; reject a truncated fit window.
    select = (lag_fs >= (t[-1] - t[0]) * lo) & (lag_fs <= (t[-1] - t[0]) * hi)
    if select.sum() < 5:
        raise ValueError("Fit window has fewer than 5 lags. Increase max_lag or reduce fit fractions.")
    x, y = lag_fs[select], msd[select]
    slope, intercept = np.polyfit(x, y, 1)
    residual = np.sum((y - (slope * x + intercept)) ** 2)
    total = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - residual / total) if total > 1e-24 else None
    alpha = float(np.polyfit(np.log(x), np.log(y), 1)[0]) if np.all(y > 1e-20) else None
    warnings = []
    duration_ok = bool(t[-1] - t[0] >= 10000)
    if not duration_ok:
        warnings.append("Production trajectory is shorter than the 10 ps reporting floor; D is withheld. Longer runs still need duration convergence.")
    if h.sum() < 5:
        warnings.append("Fewer than 5 H atoms: repeat independent seeds and check concentration effects.")
    accepted = bool(slope > 0 and r2 is not None and r2 >= 0.9 and alpha is not None and 0.7 <= alpha <= 1.3)
    if not accepted:
        warnings.append("Normal diffusion checks failed; a diffusion coefficient is withheld (ballistic, trapped or insufficient sampling).")
    warnings.append("Overlapping time origins are correlated; no statistical confidence interval is inferred.")
    # 1 A^2/fs = 1e-5 m^2/s; in 3 dimensions D = slope / 6.
    return {
        "D_m2_s": float(slope / 6 * 1e-5) if accepted and duration_ok else None,
        "raw_slope_A2_fs": float(slope), "intercept_A2": float(intercept),
        "fit_r2": r2, "log_log_exponent": alpha, "normal_diffusion_checks_passed": accepted,
        "sampling_duration_check_passed": duration_ok,
        "fit_start_fs": float(x[0]), "fit_end_fs": float(x[-1]),
        "duration_fs": float(t[-1] - t[0]), "hydrogen_atoms": int(h.sum()),
        "lag_fs": lag_fs.tolist(), "msd_A2": msd.tolist(),
        "msd_xyz_A2": xyz.tolist(), "warnings": warnings,
    }


def trapping_energy(bulk, bulk_h, defect, defect_h):
    """Positive value means favorable trapping relative to the supplied bulk H site."""
    energies = np.asarray([bulk, bulk_h, defect, defect_h], dtype=float)
    if not np.isfinite(energies).all():
        raise ValueError("Non-finite energy.")
    return float((bulk_h - bulk) - (defect_h - defect))
