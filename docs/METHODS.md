# Methods and provenance

## Scope

This software implements new forward MLIP calculations. No Quantum ESPRESSO, VASP, electronic-structure solver, DFT-D3 calculator, or pseudopotential is invoked. MACE/CHGNet checkpoint training provenance can include DFT; inference-only does not remove that provenance. Production backends are restricted to local MACE and CHGNet checkpoints. No empirical/toy potential can be selected through the application.

## MD and diffusion

The input is a fully periodic, unconstrained, fixed 3D cell. Atomic coordinates are first relaxed with FIRE at fixed cell; the user is responsible for an appropriate model/temperature-dependent lattice constant. Maxwell–Boltzmann momenta and the Langevin thermostat share a seeded NumPy generator. Initial total momentum is removed. Langevin `fixcm=False` is explicit; host center-of-mass drift is removed in postprocessing. Friction has units fs⁻¹, converted to inverse ASE time. Production samples exclude equilibration.

ASE Cartesian positions are never wrapped during these runs. Stored continuous positions therefore retain boundary crossings without a minimum-image reconstruction or a sampling-interval alias. External NPZ inputs must explicitly declare scalar `unwrapped=True`; the program does not infer missing crossings from wrapped trajectories. For external trajectories, establish fixed cell, atom identities, unwrapped coordinates and uniform timestamps before conversion. Changing-cell trajectories are unsupported.

For each frame, subtract the host mass-weighted center of mass from every H coordinate. At each lag k, average squared displacement over H atoms and **all available time origins**:

`MSD(k) = mean_t,H |r_H(t+k) − r_H(t)|²`.

Fit `MSD = slope × lag_time_fs + intercept` by unweighted least squares. The default fit window spans 20–50% of production duration and is limited by `max_lag`. If fewer than five lag points lie in the requested window, analysis fails rather than silently choosing another window. In three dimensions, `D = slope/6 × 10⁻⁵ m²/s`, since `1 Å²/fs = 10⁻⁵ m²/s`. Component MSDs are also retained.

A positive slope, linear-fit R² ≥ 0.9 and log–log MSD exponent 0.7–1.3 are required before returning D. Production must also span at least 10 ps; this reporting floor is a heuristic, not proof of duration convergence. In forward MD runs, mean production temperature must be within 20% of the target or D is withheld. These checks do not prove diffusion convergence; caging can still pass a linear fit. Fewer than 5 H atoms produce a warning. No confidence interval is derived from overlapping/correlated lag samples. Independent seeds and longer-duration, cell-size, timestep, friction and H-concentration convergence are required. Inspect thermodynamics.csv for temperature sampling. There is no NVE energy-drift assessment or automatic quantum correction.

NPZ contract (no pickle): `positions_A` float `[frames, atoms, 3]`, `time_fs` uniformly spaced float `[frames]`, `symbols` string `[atoms]`, `masses_amu` positive float `[atoms]`, `unwrapped` scalar boolean True. MD exports additionally include fixed `cell_A` `[3,3]`.

## Trapping energy

Four separately position-relaxed structures are used: bulk B, bulk with one H BH, defect D, defect with one H DH. All cells must match. B/BH and D/DH must have matching host species counts. A vacancy may change B versus D composition; host chemical potentials cancel within insertion energies.

`E_trap = (E_BH − E_B) − (E_DH − E_D)` [eV].

Positive E_trap means the supplied defect H state is energetically preferred to the supplied bulk H state. No H₂ molecular reference is needed. This is a 0 K electronic-potential energy difference, without vibrational entropy or zero-point energy. Inspect the saved relaxed coordinates to verify that the H site remained the intended site. The program does not perform site classification or distinguish local minima by defect type.

## NEB

Endpoints require the same cell and exact species/atom order, with at least one H. FIRE position relaxation is applied to each endpoint. Coincident relaxed endpoints are rejected. An IDPP minimum-image initialization followed by climbing-image NEB with the improved-tangent method is optimized using FIRE. The one local calculator is shared sequentially; no parallel image evaluation is used. Failed endpoint or band convergence yields no barrier. Forward/reverse barriers equal the maximum sampled band energy minus the corresponding endpoint energy. Image-count and force-threshold convergence remain necessary. These are potential-energy barriers, not finite-temperature free-energy barriers.

## Tensile response

Clean and H-containing cells must match in host composition and initial cell, and must be orthorhombic aligned to Cartesian axes. The selected cell vector is stretched by engineering strain ε: `L(ε)=L₀(1+ε)`. Atomic positions are scaled between increments then relaxed with FIRE at fixed cell. Transverse lengths remain fixed. ASE's tensile-positive Cauchy stress is divided by `ase.units.GPa`; no sign inversion is applied. This is a constrained-transverse-strain stress vs engineering-strain curve at 0 K, not a finite-temperature MD loading protocol or a zero-transverse-stress tensile test.

Only if both sampled stress maxima are positive and interior to their strain ranges is `reduction = 1 − σ_max,H / σ_max,clean` reported. Negative reductions are retained (strengthening in this particular model/cell). An interior maximum alone does not establish fracture or a globally converged ideal strength. Residual stress, strain spacing, metastability and orientation must be checked.

## Fe examples

Conventional bcc Fe, a=2.86 Å, 3×3×3 (54 Fe). BH has H at `(0.5,0.25,0) × a`; the alternate NEB site is `(0.5,0,0.25) × a`. The vacancy removes the Fe atom at the origin, with trial H position `(0.35,0,0) × a` in DH. These are starting structures, not fitted experimental lattice constants or guaranteed local minima. All example sites require relaxation and inspection. Real alloys can be supplied through structure files but are not generated automatically.

## References and model identity

- Original author repository: https://github.com/lyullee/hydrogen-local-energetics
- Original checkpoint registry (hashes transcribed into registry.py): https://github.com/lyullee/hydrogen-local-energetics/blob/main/manifest/model_registry.csv
- Archived CHGNet method: https://github.com/lyullee/hydrogen-local-energetics/blob/main/docs/CHGNET_NEB_METHOD.md — the original generation script was not retained. This implementation is a new workflow, not recovered original code or a replacement archive result.
- ASE MD and time units: https://ase.gitlab.io/ase/ase/md.html
- ASE NEB: https://ase-lib.org/ase/mep/neb.html
- MACE calculator documentation: https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html
- CHGNet usage/model provenance: https://chgnet.lbl.gov/

MACE-OMat-0-medium SHA-256: `d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a` (Academic Software License).

CHGNet-0.3.0-MPtrj SHA-256: `d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1` (consult upstream model terms; source identified as BSD-3-Clause in the original registry).

All references were inspected on 2026-10-22. Weight files are not redistributed. Per-run hashes preserve the actual selected checkpoint even when a different or fine-tuned model is used.
