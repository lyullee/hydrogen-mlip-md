"""ASE workflows with fixed cells and explicit convergence failures."""
import csv
from collections import Counter
from pathlib import Path
import numpy as np
from ase import units
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.optimize import FIRE
from .analysis import diffusion, trapping_energy
from .structures import validate_structure, check_pair, endpoint_distance


def number(options, key, default, lower, upper, integer=False):
    value = options.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{key} must be a finite number.")
    if not lower <= value <= upper or (integer and int(value) != value):
        raise ValueError(f"{key} must be {'an integer ' if integer else ''}in [{lower}, {upper}].")
    return int(value) if integer else float(value)


def check_options(options, allowed):
    extra = set(options) - set(allowed)
    if extra:
        raise ValueError(f"Unknown workflow options: {extra}")


def relax(atoms, calc, directory, name, fmax=0.03, steps=500):
    a = atoms.copy()
    a.calc = calc
    with FIRE(a, logfile=str(directory / f"{name}.log"), trajectory=str(directory / f"{name}.traj")) as opt:
        converged = opt.run(fmax=fmax, steps=steps)
    if not converged:
        raise RuntimeError(f"{name}: relaxation did not converge within {steps} steps.")
    energy = float(a.get_potential_energy())
    if not np.isfinite(energy) or not np.isfinite(a.get_forces()).all():
        raise RuntimeError(f"{name}: non-finite calculator output.")
    write(directory / f"{name}.extxyz", a)
    return a


def md(structures, calc, options, directory, progress=None):
    check_options(options, {"temperature_K", "timestep_fs", "equilibration_steps", "production_steps",
                            "sample_interval", "friction_fs_inv", "seed", "fmax", "relax_steps", "fit_fraction", "max_lag"})
    a = structures["structure"]
    validate_structure(a, require_h=True)
    temp = number(options, "temperature_K", 600, 1, 5000)
    dt = number(options, "timestep_fs", 0.5, 0.001, 1)
    eq = number(options, "equilibration_steps", 2000, 1, 10000000, True)
    steps = number(options, "production_steps", 20000, 20, 10000000, True)
    interval = number(options, "sample_interval", 10, 1, 10000, True)
    if steps // interval < 19:
        raise ValueError("At least 20 production samples are required.")
    friction = number(options, "friction_fs_inv", 0.002, 1e-6, 1)
    seed = number(options, "seed", 42, 0, 2**32 - 1, True)
    fmax = number(options, "fmax", 0.03, 1e-6, 1)
    relax_steps = number(options, "relax_steps", 500, 1, 100000, True)
    max_lag = number(options, "max_lag", 1000, 10, 1000000, True)
    fit = options.get("fit_fraction", [0.2, 0.5])
    # Validate analysis window before starting an expensive simulation.
    frames = steps // interval + 1
    lag_fraction = np.arange(1, min(frames // 2, max_lag) + 1) / (frames - 1)
    if len(fit) != 2 or not 0 < fit[0] < fit[1] <= 0.5 or np.sum((lag_fraction >= fit[0]) & (lag_fraction <= fit[1])) < 5:
        raise ValueError("Invalid/undersampled fit_fraction or max_lag; increase max_lag for long runs.")
    if progress:
        progress(0.01, "MD 시작 전 원자 위치를 이완하는 중입니다.")
    a = relax(a, calc, directory, "md_initial", fmax, relax_steps)
    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(a, temperature_K=temp, rng=rng)
    Stationary(a)
    # No wrapping is performed: ASE Cartesian positions remain continuous across cells.
    dyn = Langevin(a, timestep=dt * units.fs, temperature_K=temp,
                   friction=friction / units.fs, fixcm=False, rng=rng,
                   logfile=str(directory / "md.log"), loginterval=interval)
    if progress:
        def update_progress():
            phase = "평형화" if dyn.nsteps < eq else "생산"
            progress(0.05 + 0.9 * dyn.nsteps / (eq + steps),
                     f"MD {phase} · {dyn.nsteps:,}/{eq + steps:,} 단계 · 현재 {a.get_temperature():.0f} K")
        dyn.attach(update_progress, interval=max(1, (eq + steps) // 100))
    samples, times = [], []
    thermo = []
    production_start = eq
    with Trajectory(str(directory / "production.traj"), "w", atoms=a) as trajectory:
        def observe():
            if not np.isfinite(a.positions).all() or not np.isfinite(a.get_forces()).all():
                raise RuntimeError("MD became non-finite; inspect timestep and model validity.")
            if a.get_temperature() > max(5 * temp, 2000):
                raise RuntimeError("MD temperature excursion; inspect equilibration and timestep.")
            if dyn.nsteps >= production_start and (dyn.nsteps - production_start) % interval == 0:
                samples.append(a.positions.copy())
                times.append((dyn.nsteps - production_start) * dt)
                thermo.append([times[-1], a.get_temperature(), a.get_potential_energy(), a.get_kinetic_energy()])
                trajectory.write(a)
        # Monitor each integration step, save only requested production samples.
        dyn.attach(observe, interval=1)
        dyn.run(eq + steps)
    r = np.asarray(samples)
    np.savez_compressed(directory / "unwrapped.npz", positions_A=r, time_fs=times,
                        symbols=a.get_chemical_symbols(), masses_amu=a.get_masses(),
                        unwrapped=np.array(True), cell_A=a.cell.array)
    with open(directory / "thermodynamics.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_fs", "temperature_K", "potential_eV", "kinetic_eV"])
        writer.writerows(thermo)
    result = diffusion(r, times, a.get_chemical_symbols(), a.get_masses(), fit, max_lag)
    result.update({"ensemble": "NVT Langevin", "target_temperature_K": temp,
                   "mean_temperature_K": float(np.mean(np.asarray(thermo)[:, 1])),
                   "H_atomic_fraction": a.get_chemical_symbols().count("H") / len(a),
                   "seed": seed, "equilibration_steps": eq, "production_steps": steps,
                   "timestep_fs": dt, "friction_fs_inv": friction})
    temperature_ok = abs(result["mean_temperature_K"] / temp - 1) <= 0.2
    result["temperature_sampling_check_passed"] = temperature_ok
    if not temperature_ok:
        result["D_m2_s"] = None
        result["warnings"].append("Mean production temperature differs from target by >20%; D is withheld. Extend equilibration and inspect sampling.")
    result["warnings"].append("Check timestep, thermostat friction, cell size and independent-seed convergence before interpreting D.")
    if progress:
        progress(1.0, "MD와 결과 저장을 마쳤습니다.")
    return result


def trap(structures, calc, options, directory, progress=None):
    check_options(options, {"fmax", "relax_steps"})
    b, bh, d, dh = (structures[k] for k in ("bulk", "bulk_h", "defect", "defect_h"))
    check_pair(b, bh)
    check_pair(d, dh)
    if bh.get_chemical_symbols().count("H") != 1 or dh.get_chemical_symbols().count("H") != 1:
        raise ValueError("Trapping workflow requires exactly one H per hydrogenated structure.")
    if not np.allclose(b.cell.array, d.cell.array, atol=1e-6, rtol=0):
        raise ValueError("Bulk and defect cells must match for this fixed-cell comparison.")
    fmax = number(options, "fmax", 0.03, 1e-6, 1)
    steps = number(options, "relax_steps", 500, 1, 100000, True)
    energies = {}
    for index, key in enumerate(("bulk", "bulk_h", "defect", "defect_h")):
        if progress:
            progress(index / 4, f"포획에너지 · {key} 구조 이완 중 ({index + 1}/4)")
        energies[key] = float(relax(structures[key], calc, directory, key, fmax, steps).get_potential_energy())
    if progress:
        progress(1.0, "네 구조의 계산을 마쳤습니다.")
    return {"trapping_energy_eV": trapping_energy(*[energies[k] for k in ("bulk", "bulk_h", "defect", "defect_h")]),
            "energies_eV": energies, "sign_convention": "positive = favorable trapping relative to supplied bulk site",
            "warnings": ["0 K fixed-cell energy difference; no zero-point or entropy correction.",
                         "Inspect relaxed H positions: migration out of the intended site changes the interpretation.",
                         "Positive trapping alone does not establish reduced or increased embrittlement."]}


def neb(structures, calc, options, directory, progress=None):
    from ase.mep import NEB
    check_options(options, {"fmax", "relax_steps", "neb_steps", "images", "spring_eV_A2"})
    start, end = structures["start"], structures["end"]
    validate_structure(start, True)
    validate_structure(end, True)
    if start.get_chemical_symbols() != end.get_chemical_symbols() or not np.allclose(start.cell.array, end.cell.array, atol=1e-6, rtol=0):
        raise ValueError("NEB endpoints need identical cell and exact atom identity/order.")
    fmax = number(options, "fmax", 0.03, 1e-6, 1)
    steps = number(options, "relax_steps", 500, 1, 100000, True)
    n = number(options, "images", 7, 3, 31, True)
    neb_steps = number(options, "neb_steps", 1000, 1, 100000, True)
    spring = number(options, "spring_eV_A2", 0.1, 0.001, 10)
    if progress:
        progress(0.01, "NEB 시작·끝 구조를 이완하는 중입니다.")
    a = relax(start, calc, directory, "start", fmax, steps)
    b = relax(end, calc, directory, "end", fmax, steps)
    if endpoint_distance(a, b) < 0.1:
        raise ValueError("Relaxed endpoints collapsed to the same state (<0.1 A apart).")
    images = [a] + [a.copy() for _ in range(n - 2)] + [b]
    for image in images:
        image.calc = calc
    # Sequential evaluation intentionally shares one checkpoint, avoiding GPU copies.
    band = NEB(images, climb=True, method="improvedtangent", k=spring, allow_shared_calculator=True)
    band.interpolate(method="idpp", mic=True)
    with FIRE(band, logfile=str(directory / "neb.log"), trajectory=str(directory / "neb.traj")) as opt:
        if progress:
            opt.attach(lambda: progress(0.1 + 0.85 * opt.nsteps / neb_steps,
                                        f"NEB 최적화 · {opt.nsteps}/{neb_steps} 단계"), interval=10)
        ok = opt.run(fmax=fmax, steps=neb_steps)
    if not ok:
        raise RuntimeError("NEB did not converge; no barrier is reported.")
    energies = np.asarray([im.get_potential_energy() for im in images], dtype=float)
    if not np.isfinite(energies).all():
        raise RuntimeError("Non-finite NEB energies.")
    write(directory / "neb_path.extxyz", images)
    if progress:
        progress(1.0, "NEB 계산과 경로 저장을 마쳤습니다.")
    return {"forward_barrier_eV": float(energies.max() - energies[0]),
            "reverse_barrier_eV": float(energies.max() - energies[-1]),
            "relative_energies_eV": (energies - energies[0]).tolist(), "images": n,
            "warnings": ["0 K potential-energy barrier, not a finite-temperature free-energy barrier.",
                         "Inspect the path and check image-count/force convergence; supplied atom mapping is preserved."]}


def tensile(structures, calc, options, directory, progress=None):
    check_options(options, {"axis", "max_strain", "increments", "fmax", "relax_steps"})
    clean, h = structures["clean"], structures["hydrogenated"]
    check_pair(clean, h)
    if not np.allclose(clean.cell.array, np.diag(np.diag(clean.cell.array)), atol=1e-8):
        raise ValueError("Tensile workflow requires an orthorhombic cell aligned to x/y/z.")
    axis = number(options, "axis", 2, 0, 2, True)
    maximum = number(options, "max_strain", 0.25, 0.001, 1)
    increments = number(options, "increments", 25, 3, 200, True)
    fmax = number(options, "fmax", 0.03, 1e-6, 1)
    steps = number(options, "relax_steps", 500, 1, 100000, True)
    strains = np.linspace(0, maximum, increments + 1)
    curves, peaks = {}, {}
    for key, original in (("clean", clean), ("hydrogenated", h)):
        a = original.copy()
        reference = original.cell.array.copy()
        rows = []
        for index, strain in enumerate(strains):
            if progress:
                offset = 0 if key == "clean" else len(strains)
                progress((offset + index) / (2 * len(strains)), f"인장 · {key} · 변형률 {strain:.3f}")
            cell = reference.copy()
            cell[axis] *= 1 + strain
            a.set_cell(cell, scale_atoms=True)
            a = relax(a, calc, directory, f"{key}_{index:03d}", fmax, steps)
            # ASE tensile stress is positive; conversion eV/A^3 -> GPa.
            stress = float(a.get_stress(voigt=False)[axis, axis] / units.GPa)
            if not np.isfinite(stress):
                raise RuntimeError("Model did not provide finite stress.")
            rows.append([float(strain), stress, float(a.get_potential_energy())])
        curves[key] = rows
        index = int(np.argmax(np.asarray(rows)[:, 1]))
        peaks[key] = {"sampled_max_stress_GPa": rows[index][1], "strain": rows[index][0],
                      "interior_peak_observed": 0 < index < increments}
    complete = all(p["interior_peak_observed"] and p["sampled_max_stress_GPa"] > 0 for p in peaks.values())
    loss = 1 - peaks["hydrogenated"]["sampled_max_stress_GPa"] / peaks["clean"]["sampled_max_stress_GPa"] if complete else None
    with open(directory / "tensile.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["engineering_strain", "clean_stress_GPa", "H_stress_GPa"])
        writer.writerows(zip(strains, np.asarray(curves["clean"])[:, 1], np.asarray(curves["hydrogenated"])[:, 1]))
    warnings = ["0 K quasistatic axial strain with fixed transverse dimensions; not finite-temperature tensile MD.",
                "Ideal-cell tensile response is not experimental ductility, fracture toughness or a HELP/HEDE diagnosis.",
                "Check strain spacing, residual stress, orientation, cell size and H concentration."]
    if not complete:
        warnings.append("A positive interior peak was not observed in both curves; peak reduction is withheld.")
    if progress:
        progress(1.0, "두 인장 곡선 계산과 저장을 마쳤습니다.")
    return {"axis": axis, "curves": curves, "peaks": peaks, "peak_strength_reduction_fraction": loss,
            "H_atomic_fraction": h.get_chemical_symbols().count("H") / len(h), "warnings": warnings}


WORKFLOWS = {"md": (md, ["structure"]), "trap": (trap, ["bulk", "bulk_h", "defect", "defect_h"]),
             "neb": (neb, ["start", "end"]), "tensile": (tensile, ["clean", "hydrogenated"])}
