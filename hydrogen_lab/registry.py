"""Original repository registry, kept as external checkpoint references only."""
from pathlib import Path
import os
import tempfile
import urllib.request
from .models import sha256

REGISTRY = {
    "chgnet": {
        "filename": "chgnet_0.3.0.pth.tar",
        "url": "https://raw.githubusercontent.com/CederGroupHub/chgnet/main/chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar",
        "sha256": "d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1",
        "terms": "Consult CHGNet upstream model terms; original registry identifies BSD-3-Clause source.",
    },
    "mace": {
        "filename": "mace-omat-0-medium.model",
        "url": "https://github.com/ACEsuit/mace-mp/releases/download/mace_omat_0/mace-omat-0-medium.model",
        "sha256": "d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a",
        "terms": "MACE-OMat-0 Academic Software License; review upstream terms for your use.",
    },
}


def fetch_model(name, destination):
    entry = REGISTRY[name]
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / entry["filename"]
    if target.exists():
        if sha256(target) != entry["sha256"]:
            raise ValueError("Existing checkpoint has a different hash; choose another destination.")
        return target
    handle, temporary = tempfile.mkstemp(prefix="checkpoint_", suffix=".part", dir=root)
    os.close(handle)
    try:
        with urllib.request.urlopen(entry["url"], timeout=60) as source, open(temporary, "wb") as output:
            while block := source.read(1024 * 1024):
                output.write(block)
        if sha256(temporary) != entry["sha256"]:
            raise ValueError("Downloaded checkpoint failed SHA-256 verification.")
        # A concurrent downloader may already have completed this model.
        if target.exists():
            if sha256(target) != entry["sha256"]:
                raise ValueError("Destination changed during download.")
        else:
            os.rename(temporary, target)
        return target
    finally:
        Path(temporary).unlink(missing_ok=True)
