import argparse
import json
from pathlib import Path
import numpy as np
from ase.io import write
from .runner import run, save_json, analyze_npz
from .structures import iron_examples


def examples(destination):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    for name, atoms in iron_examples().items():
        write(root / f"{name}.extxyz", atoms)
    base = {"model": {"backend": "chgnet", "path": "../models/chgnet_0.3.0.pth.tar", "device": "cpu",
                       "supported_elements": ["Fe", "H"], "validation_note": ""}}
    configs = {
        "md": ({"structure": "bulk_h.extxyz"}, {"temperature_K": 600, "timestep_fs": 0.5,
                "equilibration_steps": 2000, "production_steps": 20000, "sample_interval": 10, "seed": 42}),
        "trap": ({k: k + ".extxyz" for k in ("bulk", "bulk_h", "defect", "defect_h")}, {}),
        "neb": ({"start": "bulk_h.extxyz", "end": "neb_end.extxyz"}, {"images": 7}),
        "tensile": ({"clean": "bulk.extxyz", "hydrogenated": "bulk_h.extxyz"}, {"axis": 2, "max_strain": 0.25, "increments": 25}),
    }
    for name, (structures, options) in configs.items():
        save_json(root / f"{name}.json", {**base, "workflow": name, "structures": structures, "options": options})
    # This artificial Brownian trajectory tests the estimator; it is not an MLIP/Fe result.
    rng = np.random.default_rng(2026)
    frames, nh, dt = 1001, 64, 10.0
    expected = 1e-8
    positions = np.zeros((frames, nh + 2, 3))
    positions[1:, 2:] = np.cumsum(rng.normal(scale=np.sqrt(2 * expected / 1e-5 * dt), size=(frames - 1, nh, 3)), axis=0)
    np.savez_compressed(root / "synthetic_diffusion.npz", positions_A=positions,
                        time_fs=np.arange(frames) * dt, symbols=["Fe", "Fe"] + ["H"] * nh,
                        masses_amu=[55.845, 55.845] + [1.008] * nh, unwrapped=np.array(True), synthetic=np.array(True))
    save_json(root / "synthetic_diffusion_info.json", {"synthetic": True, "not_a_material_prediction": True,
                                                      "generator_D_m2_s": expected, "seed": 2026})
    return root


def main():
    parser = argparse.ArgumentParser(description="Hydrogen MLIP/MD workflows; no DFT backend")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("examples", help="Write bcc Fe inputs and a clearly marked synthetic analysis demo")
    p.add_argument("--out", default="examples")
    p = sub.add_parser("run", help="Run a JSON-configured MLIP workflow")
    p.add_argument("config")
    p.add_argument("--out", default="runs")
    p = sub.add_parser("analyze", help="Analyze an unwrapped NPZ trajectory without a model")
    p.add_argument("trajectory")
    p.add_argument("--out", default="diffusion_report.json")
    p.add_argument("--max-lag", type=int, default=1000)
    p = sub.add_parser("ui", help="Launch the Korean local interface")
    p.add_argument("--port", type=int, default=8511)
    p = sub.add_parser("fetch-model", help="Explicitly download a registry checkpoint and verify SHA-256")
    p.add_argument("--model", choices=["chgnet", "mace"], required=True)
    p.add_argument("--out", default="models")
    args = parser.parse_args()
    try:
        if args.command == "examples":
            print(examples(args.out).resolve())
        elif args.command == "run":
            path = Path(args.config).resolve()
            report = run(json.loads(path.read_text(encoding="utf-8-sig")), path.parent, args.out)
            print(report["run_directory"])
        elif args.command == "analyze":
            result = analyze_npz(args.trajectory, max_lag=args.max_lag)
            save_json(args.out, {"source": str(Path(args.trajectory).resolve()), "result": result})
            print(Path(args.out).resolve())
        elif args.command == "fetch-model":
            from .registry import fetch_model, REGISTRY
            print(REGISTRY[args.model]["terms"])
            print(fetch_model(args.model, args.out))
        else:
            import subprocess
            import sys
            app = Path(__file__).with_name("app.py")
            raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app),
                                              "--global.developmentMode", "false", "--server.address", "127.0.0.1",
                                              "--server.port", str(args.port),
                                              "--server.headless", "true", "--server.showEmailPrompt", "false",
                                              "--browser.gatherUsageStats", "false"]))
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
