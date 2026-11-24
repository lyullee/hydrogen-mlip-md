import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import numpy as np
from ase.io import read, write
from . import __version__
from .models import load_calculator, sha256
from .structures import validate_structure
from .workflows import WORKFLOWS


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def run(config, base_dir=".", output_root="runs", progress=None):
    """Create a new run directory; failures are also recorded, and never overwrite runs."""
    directory = Path(output_root).resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    versions = {"hydrogen-mlip-md": __version__}
    for name in ("numpy", "ase", "mace-torch", "chgnet", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    report = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "software": versions, "config": config, "workflow": config.get("workflow"),
              "assessment": "Research indicators only; no calibrated embrittlement severity or service-life prediction.",
              "DFT_executed": False, "run_directory": str(directory)}
    save_json(directory / "config.json", config)
    save_json(directory / "report.json", report)
    try:
        if set(config) - {"workflow", "model", "structures", "options"}:
            raise ValueError("Unknown top-level config key.")
        kind = config.get("workflow")
        if kind not in WORKFLOWS:
            raise ValueError("workflow must be md, trap, neb or tensile.")
        function, keys = WORKFLOWS[kind]
        paths = config.get("structures", {})
        if set(paths) != set(keys):
            raise ValueError(f"{kind} requires exactly these structure keys: {keys}")
        structures, records = {}, {}
        for key, filename in paths.items():
            path = (Path(base_dir) / filename).resolve()
            structures[key] = read(path, index=-1)
            validate_structure(structures[key])
            records[key] = {"path": str(path), "sha256": sha256(path),
                            "formula": structures[key].get_chemical_formula(), "atom_count": len(structures[key])}
            write(directory / f"input_{key}.extxyz", structures[key])
        report["inputs"] = records
        spec = dict(config.get("model", {}))
        spec["path"] = str((Path(base_dir) / spec.get("path", "")).resolve())
        symbols = set(s for a in structures.values() for s in a.get_chemical_symbols())
        if progress:
            progress(0.0, "MLIP 모델을 불러오는 중입니다.")
        calc, provenance = load_calculator(spec, symbols)
        report["model"] = provenance
        save_json(directory / "report.json", report)
        result = function(structures, calc, config.get("options", {}), directory, progress=progress)
        result.setdefault("warnings", []).insert(0, "Element coverage does not prove accuracy for H, vacancies, grain boundaries or fracture configurations.")
        report.update(status="completed", result=result)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(directory / "report.json", report)
    return report


def analyze_npz(path, fit_fraction=(0.2, 0.5), max_lag=1000):
    from .analysis import diffusion
    with np.load(path, allow_pickle=False) as data:
        if "unwrapped" not in data or data["unwrapped"].shape != () or data["unwrapped"].item() is not True:
            raise ValueError("NPZ must explicitly declare scalar unwrapped=True; wrapped trajectories are unsupported.")
        result = diffusion(data["positions_A"], data["time_fs"], data["symbols"],
                           data["masses_amu"], fit_fraction, max_lag)
        result["synthetic"] = bool(data["synthetic"].item()) if "synthetic" in data else None
        return result
