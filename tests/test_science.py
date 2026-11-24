"""Analytical fixtures test numerics; toy potentials are never exposed in the app."""
import json
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from hydrogen_lab.analysis import diffusion, trapping_energy
from hydrogen_lab.cli import examples
from hydrogen_lab.runner import analyze_npz, run
from hydrogen_lab.structures import iron_examples, validate_structure, check_pair
from hydrogen_lab.models import load_calculator
from hydrogen_lab.workflows import md, trap, neb, tensile, relax


def brownian(frames=2001, nh=128, seed=42):
    rng = np.random.default_rng(seed)
    dt, D = 10.0, 1e-8
    r = np.zeros((frames, nh + 2, 3))
    r[1:, 2:] = np.cumsum(rng.normal(scale=np.sqrt(2 * D / 1e-5 * dt), size=(frames - 1, nh, 3)), axis=0)
    return r, np.arange(frames) * dt, ["Fe", "Fe"] + ["H"] * nh, [55.845] * 2 + [1.008] * nh


def test_einstein_units_and_host_drift_invariance():
    r, t, s, m = brownian()
    a = diffusion(r, t, s, m)
    assert a["normal_diffusion_checks_passed"]
    assert a["D_m2_s"] == pytest.approx(1e-8, rel=0.18)
    drift = np.arange(len(t))[:, None, None] * np.array([0.1, -0.2, 0.05])[None, None, :]
    b = diffusion(r + drift, t, s, m)
    assert b["D_m2_s"] == pytest.approx(a["D_m2_s"], rel=1e-12)


def test_ballistic_and_immobile_do_not_become_diffusion_coefficients():
    r, t, s, m = brownian(frames=101, nh=1)
    r[:, 2, 0] = t * 0.01
    r[:, 2, 1:] = 0
    assert diffusion(r, t, s, m)["D_m2_s"] is None
    r[:] = 0
    result = diffusion(r, t, s, m)
    assert result["D_m2_s"] is None
    assert result["fit_r2"] is None


@pytest.mark.parametrize("mode", ["irregular", "nan", "no_h", "masses", "short", "window"])
def test_invalid_analysis_inputs(mode):
    r, t, s, m = brownian(frames=101, nh=2)
    kw = {}
    if mode == "irregular":
        t[2] += 1
    elif mode == "nan":
        r[0, 0, 0] = np.nan
    elif mode == "no_h":
        s = ["Fe"] * len(s)
    elif mode == "masses":
        m[0] = 0
    elif mode == "short":
        r, t = r[:10], t[:10]
    else:
        kw["fit_fraction"] = (0.6, 0.8)
    with pytest.raises(ValueError):
        diffusion(r, t, s, m, **kw)


def test_trap_reference_cancellation_and_sign():
    assert trapping_energy(-100, -102, -90, -93) == 1
    assert trapping_energy(-100, -103, -90, -92) == -1
    assert trapping_energy(-200, -202, -190, -193) == 1


def test_examples_and_pair_validation(tmp_path):
    examples(tmp_path)
    for a in iron_examples().values():
        validate_structure(a)
    a = iron_examples()
    check_pair(a["bulk"], a["bulk_h"])
    a["bulk_h"].cell[0, 0] += 0.1
    with pytest.raises(ValueError):
        check_pair(a["bulk"], a["bulk_h"])
    result = analyze_npz(tmp_path / "synthetic_diffusion.npz")
    assert result["D_m2_s"] == pytest.approx(1e-8, rel=0.25)


def test_wrapped_npz_is_refused(tmp_path):
    r, t, s, m = brownian(frames=101, nh=1)
    np.savez(tmp_path / "bad.npz", positions_A=r, time_fs=t, symbols=s, masses_amu=m, unwrapped=False)
    with pytest.raises(ValueError, match="unwrapped"):
        analyze_npz(tmp_path / "bad.npz")


def test_short_brownian_run_withholds_D_even_if_linear():
    r, t, s, m = brownian(frames=501, nh=128)
    result = diffusion(r, t, s, m)
    assert result["D_m2_s"] is None
    assert result["sampling_duration_check_passed"] is False


def test_bad_model_hash_and_dft_are_refused(tmp_path):
    checkpoint = tmp_path / "model"
    checkpoint.write_text("not a model")
    with pytest.raises(ValueError, match="SHA-256"):
        load_calculator({"backend": "mace", "path": str(checkpoint), "supported_elements": ["Fe", "H"], "sha256": "wrong"}, ["Fe", "H"])
    with pytest.raises(ValueError, match="DFT"):
        load_calculator({"backend": "espresso"}, ["Fe", "H"])


def test_failed_run_is_recorded_without_overwrite(tmp_path):
    config = {"workflow": "not_supported"}
    for _ in range(2):
        with pytest.raises(ValueError):
            run(config, output_root=tmp_path)
    reports = list(tmp_path.glob("*/report.json"))
    assert len(reports) == 2
    assert all(json.loads(p.read_text())["status"] == "failed" for p in reports)


class Toy(Calculator):
    """Nonphysical analytic fixture with independently known stress/trapping values."""
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        nh = atoms.get_chemical_symbols().count("H")
        nfe = len(atoms) - nh
        eps = atoms.cell[2, 2] / 8 - 1
        stress = eps * (0.3 - eps) * (0.7 if nh else 1)
        self.results = {"energy": -nfe - nh * (3 if nfe == 2 else 2),
                        "forces": np.zeros((len(atoms), 3)), "stress": np.array([0, 0, stress, 0, 0, 0])}


def pair():
    a = Atoms("Fe3", positions=[[0, 0, 0], [4, 4, 4], [0, 4, 0]], cell=[8, 8, 8], pbc=True)
    b = a.copy()
    b += Atoms("H", positions=[[2, 2, 2]])
    return a, b


def test_tensile_peak_sign_units_and_reduction(tmp_path):
    a, b = pair()
    result = tensile({"clean": a, "hydrogenated": b}, Toy(), {"max_strain": 0.3, "increments": 6}, tmp_path)
    assert result["peak_strength_reduction_fraction"] == pytest.approx(0.3)
    assert result["peaks"]["clean"]["strain"] == pytest.approx(0.15)
    from ase import units
    assert result["peaks"]["clean"]["sampled_max_stress_GPa"] == pytest.approx(0.0225 / units.GPa)


def test_tensile_incomplete_curve_withholds_reduction(tmp_path):
    a, b = pair()
    result = tensile({"clean": a, "hydrogenated": b}, Toy(), {"max_strain": 0.05, "increments": 5}, tmp_path)
    assert result["peak_strength_reduction_fraction"] is None


def test_trapping_workflow_uses_four_consistent_references(tmp_path):
    a, b = pair()
    d, dh = a.copy(), b.copy()
    del d[0]
    del dh[0]
    result = trap({"bulk": a, "bulk_h": b, "defect": d, "defect_h": dh}, Toy(), {}, tmp_path)
    assert result["trapping_energy_eV"] == 1


class DoubleWell(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x = atoms.positions[-1, 0]
        force = np.zeros((len(atoms), 3))
        force[-1, 0] = -2 * (x - 1) * (x - 3) * (2 * x - 4)
        self.results = {"energy": (x - 1) ** 2 * (x - 3) ** 2, "forces": force}


def test_neb_recovers_analytic_barrier(tmp_path):
    a = Atoms("FeH", positions=[[0, 0, 0], [1, 2, 2]], cell=[8, 8, 8], pbc=True)
    b = a.copy()
    b.positions[-1, 0] = 3
    result = neb({"start": a, "end": b}, DoubleWell(), {"images": 5}, tmp_path)
    assert result["forward_barrier_eV"] == pytest.approx(1, abs=1e-6)
    assert result["reverse_barrier_eV"] == pytest.approx(1, abs=1e-6)


def test_md_production_sampling_and_export(tmp_path):
    _, a = pair()
    result = md({"structure": a}, Toy(), {"equilibration_steps": 7, "production_steps": 100,
                "sample_interval": 5, "temperature_K": 100, "timestep_fs": 0.1}, tmp_path)
    with np.load(tmp_path / "unwrapped.npz") as data:
        assert data["time_fs"][0] == 0
        assert data["time_fs"][-1] == pytest.approx(10)
        assert len(data["time_fs"]) == 21
        assert data["unwrapped"].item() is True
    assert result["D_m2_s"] is None  # tiny free-flight run is not normal diffusion


def test_no_unconverged_relaxation_is_reported(tmp_path):
    a = Atoms("FeH", positions=[[0, 0, 0], [1.2, 2, 2]], cell=[8, 8, 8], pbc=True)
    with pytest.raises(RuntimeError, match="did not converge"):
        relax(a, DoubleWell(), tmp_path, "bad", fmax=1e-12, steps=1)
