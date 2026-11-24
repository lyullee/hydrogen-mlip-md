"""Explicit local checkpoints only; no automatic model download or DFT calculator."""
from pathlib import Path
import hashlib


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_calculator(spec, symbols):
    allowed = {"backend", "path", "device", "sha256", "supported_elements", "validation_note", "cpu_threads"}
    if set(spec) - allowed:
        raise ValueError(f"Unknown model options: {set(spec) - allowed}")
    backend = spec.get("backend")
    if backend not in {"mace", "chgnet"}:
        raise ValueError("backend must be mace or chgnet. DFT backends are not supported.")
    path = Path(spec.get("path", "")).resolve()
    if not path.is_file():
        raise ValueError("A trusted local MLIP checkpoint file is required.")
    elements = spec.get("supported_elements", [])
    if not set(symbols).issubset(set(elements)):
        raise ValueError("Declare all structure elements, including H, in model.supported_elements.")
    device = spec.get("device", "cpu")
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be cpu, cuda or mps.")
    checksum = sha256(path)
    if spec.get("sha256") and spec["sha256"].lower() != checksum:
        raise ValueError("Checkpoint SHA-256 does not match the configured hash.")
    if device == "cpu":
        import torch
        threads = spec.get("cpu_threads", 2)
        if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 64:
            raise ValueError("cpu_threads must be an integer in [1,64].")
        torch.set_num_threads(threads)
        spec = {**spec, "cpu_threads": threads}
    if backend == "mace":
        from mace.calculators import MACECalculator
        calc = MACECalculator(model_paths=str(path), device=device, default_dtype="float64")
    else:
        from chgnet.model.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator
        calc = CHGNetCalculator(model=CHGNet.from_file(str(path)), use_device=device)
    return calc, {**spec, "path": str(path), "sha256": checksum,
                  "element_support_is_user_declared": True,
                  "validation_status": "user_documented" if spec.get("validation_note") else "unvalidated"}
