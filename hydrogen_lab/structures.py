from collections import Counter
import numpy as np
from ase import Atom
from ase.build import bulk
from ase.geometry import find_mic


def validate_structure(atoms, require_h=False):
    if len(atoms) < 2 or not atoms.pbc.all() or atoms.cell.rank != 3 or atoms.get_volume() < 1e-6:
        raise ValueError("Use a 3D periodic cell containing at least 2 atoms.")
    if atoms.constraints:
        raise ValueError("Remove constraints; these workflows use unconstrained atomic relaxation.")
    if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
        raise ValueError("Non-finite coordinates/cell.")
    if require_h and "H" not in atoms.get_chemical_symbols():
        raise ValueError("At least one H atom is required.")
    if not any(s != "H" for s in atoms.get_chemical_symbols()):
        raise ValueError("A host material is required.")
    # Neighbor list checks include periodic replicas; avoid dense N-by-N matrices.
    from ase.neighborlist import neighbor_list
    distances = neighbor_list("d", atoms, 0.5)
    if len(distances) and np.min(distances) < 0.5:
        raise ValueError("Atoms are closer than 0.5 A; check structure and H sites.")


def check_pair(clean, hydrogenated):
    validate_structure(clean)
    validate_structure(hydrogenated, require_h=True)
    if "H" in clean.get_chemical_symbols():
        raise ValueError("The clean reference must contain no hydrogen.")
    a = Counter(clean.get_chemical_symbols())
    b = Counter(s for s in hydrogenated.get_chemical_symbols() if s != "H")
    if a != b or not np.allclose(clean.cell.array, hydrogenated.cell.array, atol=1e-6, rtol=0):
        raise ValueError("Paired structures require identical host composition and cell.")


def iron_examples(repeat=3, lattice=2.86):
    if repeat < 2 or lattice <= 0:
        raise ValueError("repeat >= 2 and lattice > 0 are required.")
    host = bulk("Fe", "bcc", a=lattice, cubic=True).repeat((repeat,) * 3)
    h = host.copy()
    # Conventional bcc tetrahedral site, expressed in Cartesian coordinates.
    h.append(Atom("H", np.array([0.5, 0.25, 0.0]) * lattice))
    end = host.copy()
    end.append(Atom("H", np.array([0.5, 0.0, 0.25]) * lattice))
    defect = host.copy()
    del defect[0]
    dh = defect.copy()
    dh.append(Atom("H", np.array([0.35, 0.0, 0.0]) * lattice))
    return {"bulk": host, "bulk_h": h, "defect": defect, "defect_h": dh, "neb_end": end}


def endpoint_distance(a, b):
    return np.linalg.norm(find_mic(b.positions - a.positions, a.cell, a.pbc)[0], axis=1).max()
