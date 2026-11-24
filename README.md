# Hydrogen Lab

Local MLIP and molecular-dynamics workflows for hydrogen transport and mechanical response in iron and steels.

Hydrogen Lab combines **MACE or CHGNet** machine-learning interatomic potentials (MLIPs) with the **Atomic Simulation Environment (ASE)**. It provides four calculation workflows for hydrogen diffusion, defect trapping, migration barriers, and tensile response. A Korean-language browser interface and a command-line interface share the same Python engine.

The software brings structure inputs, model selection, calculation settings, trajectory analysis, and execution records into one local workflow. Each calculation records input structures, checkpoint identity, package versions, convergence status, and results. These records support repeatable comparisons between explicitly defined atomistic configurations.

**Current version — 0.1.1.** Python 3.11 or later is required. The Windows launcher automates installation and startup. The Python engine also runs on Linux, including WSL.

## What you can calculate

| Workflow | Required structures | Method | Main results |
|---|---|---|---|
| Hydrogen diffusion (MD) | A periodic host containing H | Fixed-cell relaxation, Langevin NVT dynamics, and time-origin-averaged mean-squared displacement (MSD) | Diffusion coefficient in m²/s when reporting checks pass, component MSDs, fit diagnostics, temperature history, and trajectories |
| Defect trapping | Bulk, bulk + H, defect, and defect + H | Four fixed-cell relaxations and comparison of hydrogen insertion energies | Trapping energy in eV, constituent energies, and relaxed structures |
| Hydrogen migration (NEB) | Initial and final states with identical atom identities and ordering | Endpoint relaxation, IDPP initialization, and climbing-image NEB optimization | Forward and reverse barriers in eV, relative image energies, and the optimized path |
| Tensile response | Matching hydrogen-free and hydrogen-containing cells | Incremental axial strain with atomic relaxation and fixed transverse dimensions | Stress–strain curves in GPa, sampled peak stresses, and a peak-stress reduction fraction when both curves contain suitable maxima |

Existing continuous-coordinate NPZ trajectories can also be analyzed without loading an MLIP checkpoint. A supplied synthetic Brownian trajectory demonstrates the diffusion estimator and the interface.

The built-in examples use a 3 × 3 × 3 conventional bcc Fe supercell with 54 Fe atoms, a corresponding single-vacancy cell, and hydrogen-containing variants with one H atom. The starting lattice parameter is 2.86 Å. Custom alloy and defect structures can be supplied through files after selecting a checkpoint appropriate for their elements and local environments.

## Quick start on Windows

Download and extract the repository ZIP, or clone the repository.

~~~powershell
git clone https://github.com/lyullee/hydrogen-mlip-md.git
cd hydrogen-mlip-md
~~~

Double-click **`START.cmd`** in the project folder. On the first run, the launcher prepares a Python environment, installs the CHGNet `backend`, downloads the registered checkpoint, and verifies its SHA-256 hash. It then opens the application at http://127.0.0.1:8511.

The launcher reopens the page when the same application is already using the port. Startup logs are saved under `.runtime/`. First-time setup requires internet access.

For an explicit installation and launch sequence, run the following from the project folder.

~~~powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Backend chgnet -DownloadCheckpoint
powershell -ExecutionPolicy Bypass -File .\launch.ps1
~~~

Select a workflow, confirm the local model path and device, and review the structure and JSON settings. The default `backend` is CHGNet and the default device is CPU. The model connection button evaluates energy, forces, and stress on a small Fe–H structure. Calculation forms display progress and save results under `runs/`.

| Interface label | Meaning |
|---|---|
| 새 MLIP 계산 | New MLIP calculation |
| MD 궤적 분석 | MD trajectory analysis |
| 실행 기록·방법 | Run history and methods |
| Fe–H 모델 연결 확인 | Check the Fe–H model connection |
| 계산 실행 | Run the calculation |
| 합성 데이터로 화면 체험 | Explore the interface with synthetic data |

## Installation options

### Interface and trajectory analysis

The base package includes the interface, ASE workflows, and trajectory analysis. Install it with the following commands to explore the synthetic example or analyze existing NPZ trajectories.

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m hydrogen_lab.cli ui
~~~

Open http://127.0.0.1:8511 after the server starts. The **ui** command starts the interface directly. `START.cmd` and `launch.ps1` additionally prepare the default CHGNet environment and checkpoint.

### MACE or CHGNet calculations

Install the optional dependency group for the `backend` you intend to use, then fetch its registered checkpoint.

~~~powershell
# CHGNet
.\.venv\Scripts\python.exe -m pip install -e ".[chgnet]"
.\.venv\Scripts\python.exe -m hydrogen_lab.cli fetch-model --model chgnet --out models

# MACE
.\.venv\Scripts\python.exe -m pip install -e ".[mace]"
.\.venv\Scripts\python.exe -m hydrogen_lab.cli fetch-model --model mace --out models
~~~

The registry contains `chgnet_0.3.0.pth.tar` and `mace-omat-0-medium.model`. Another trusted local checkpoint compatible with the selected `backend` can also be used. Set its path, element coverage, device, and optional expected hash in the calculation configuration.

The device options are cpu, cuda, and mps. Availability depends on the operating system, hardware, PyTorch installation, and `backend`. CPU model loading defaults to two PyTorch threads, configurable through `model.cpu_threads` in the range 1–64. GPU and MPS execution require validation in the target environment.

### Linux and WSL

Use the Python entry points from the repository folder.

~~~bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[chgnet]"
.venv/bin/python -m hydrogen_lab.cli fetch-model --model chgnet --out models
.venv/bin/python -m hydrogen_lab.cli ui
~~~

The examples below use the Windows Python executable path. On Linux or WSL, use `.venv/bin/python` in its place.

## Command-line workflows

Generate example inputs or use the structures and configurations supplied with the repository. The examples command writes the named files in its destination, so choose a separate destination when preserving edited inputs.

~~~powershell
.\.venv\Scripts\python.exe -m hydrogen_lab.cli --help
.\.venv\Scripts\python.exe -m hydrogen_lab.cli examples --out examples

# Review the corresponding JSON configuration before each calculation.
.\.venv\Scripts\python.exe -m hydrogen_lab.cli run examples/md.json --out runs
.\.venv\Scripts\python.exe -m hydrogen_lab.cli run examples/trap.json --out runs
.\.venv\Scripts\python.exe -m hydrogen_lab.cli run examples/neb.json --out runs
.\.venv\Scripts\python.exe -m hydrogen_lab.cli run examples/tensile.json --out runs

# Analyze the supplied synthetic Brownian trajectory.
.\.venv\Scripts\python.exe -m hydrogen_lab.cli analyze examples/synthetic_diffusion.npz --out demo_report.json
~~~

Each successful run command prints the new run directory. Errors are reported to the terminal and recorded in that run's `report.json`. A UTC timestamp and a unique suffix identify each run directory.

### Example MD configuration

The following configuration uses the supplied Fe–H structure and registered CHGNet checkpoint.

~~~json
{
  "workflow": "md",
  "model": {
    "backend": "chgnet",
    "path": "../models/chgnet_0.3.0.pth.tar",
    "device": "cpu",
    "supported_elements": ["Fe", "H"],
    "cpu_threads": 2,
    "sha256": "d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1",
    "validation_note": ""
  },
  "structures": {
    "structure": "bulk_h.extxyz"
  },
  "options": {
    "temperature_K": 600,
    "timestep_fs": 0.5,
    "equilibration_steps": 2000,
    "production_steps": 20000,
    "sample_interval": 10,
    "friction_fs_inv": 0.002,
    "fmax": 0.03,
    "relax_steps": 500,
    "fit_fraction": [0.2, 0.5],
    "max_lag": 1000,
    "seed": 42
  }
}
~~~

Save this configuration as `examples/md.json` for the relative paths shown. Structure and checkpoint paths are resolved relative to the configuration file. The output root is resolved relative to the current working directory.

The `supported_elements` field records the user's declaration of checkpoint coverage. The `validation_note` field stores supporting information about model applicability. With `sha256` present, loading requires an exact match to that hash. Update the expected hash when selecting a different checkpoint.

For the registered MACE checkpoint, set `backend` to `mace` and `path` to `../models/mace-omat-0-medium.model`. Its registered SHA-256 is d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a.

The default MD example provides 1 ps of equilibration and 10 ps of production. It is a starting configuration for exploring the workflow. Quantitative studies require convergence checks for duration, timestep, cell size, hydrogen concentration, thermostat settings, and independent random seeds. Runtime depends strongly on checkpoint, atom count, device, and trajectory length.

## Calculation methods

### Hydrogen diffusion

The workflow relaxes atomic positions with FIRE at fixed cell dimensions, initializes velocities, and performs Langevin NVT dynamics. Production samples are collected after equilibration. Continuous Cartesian coordinates retain periodic boundary crossings.

The analysis subtracts the host's mass-weighted center-of-mass motion from the hydrogen coordinates. It averages squared displacements over H atoms and available time origins at each lag, then fits the three-dimensional Einstein relation.

~~~text
MSD(tau) = mean over H atoms and time origins of |r_H(t + tau) - r_H(t)|²
D = slope(MSD versus tau) / 6
1 Å²/fs = 10^-5 m²/s
~~~

The default fit interval covers 20–50% of production duration. The `max_lag` option limits the number of lag frames used for analysis. Increase it when extending a trajectory so that the requested fit interval remains covered.

| Reporting check | Requirement |
|---|---|
| Production duration | At least 10 ps |
| Linear-fit slope | Positive |
| Linear-fit R² | At least 0.9 |
| Log–log MSD exponent | Between 0.7 and 1.3 |
| Requested fit interval | At least five lag points |
| Mean production temperature, for forward MD | Within 20% of the target |

The duration threshold and fit criteria screen for basic sampling problems. Physical convergence still requires independent assessment. When duration, diffusion-fit, or temperature checks fail, the report retains diagnostics and sets `D_m2_s` to null. An invalid or undersampled fit interval raises an error. Overlapping time origins are correlated, and statistical uncertainty should be assessed with independent runs.

### Defect trapping

Four structures are relaxed independently at fixed cell dimensions. The trapping energy compares hydrogen insertion at the supplied bulk and defect sites.

~~~text
E_trap = [E(bulk + H) - E(bulk)] - [E(defect + H) - E(defect)]
~~~

A positive value favors the supplied defect site relative to the supplied bulk site. Both hydrogen-containing structures must contain exactly one H atom. Paired host compositions and all cell dimensions must satisfy the input checks. Saved relaxed structures allow inspection of hydrogen migration during optimization.

### Migration barriers

Initial and final structures must have identical cells, species, and atom ordering. After endpoint relaxation, the workflow creates an IDPP-interpolated path and optimizes a climbing-image NEB band with FIRE. The images option counts all images, including both endpoints. Calculator evaluations are sequential.

Forward and reverse barriers are the maximum sampled band energy relative to the corresponding endpoint. Endpoint and band convergence are required. Barrier interpretation also requires inspection of the path and convergence with image count and force threshold.

### Tensile response

Hydrogen-free and hydrogen-containing structures are strained along a selected Cartesian axis. Each increment scales the chosen cell vector and relaxes atomic positions with fixed transverse dimensions. Results contain engineering strain and tensile-positive Cauchy stress in GPa.

The peak-stress reduction fraction is 1 − sigma_max,H / sigma_max,clean. It is reported when both curves have positive maxima inside the sampled strain range. A negative value indicates higher sampled peak stress in the hydrogen-containing configuration. Assess strain increments, residual stress, orientation, cell size, and metastable structures when interpreting the curves.

Detailed equations, sign conventions, and implementation references are available in the methods guide. [Methods and provenance](docs/METHODS.md)

## Inputs and saved results

### Structure requirements

All workflows use three-dimensional periodic cells with finite coordinates and unconstrained atoms. Extended XYZ files preserve the cell and periodic-boundary metadata used by the examples. The browser accepts .extxyz, .xyz, .cif, .vasp, and .traj uploads. The CLI reads formats supported by ASE and uses the last frame of a multi-frame file.

Paired hydrogen-free and hydrogen-containing structures must have the same host composition and cell. NEB additionally requires exact atom ordering. Tensile calculations require an orthorhombic cell aligned with Cartesian axes. Prepare and inspect custom alloy and defect structures before calculation.

### NPZ trajectory format

Existing trajectories can be analyzed after conversion to the following array format.

| Array | Shape | Content |
|---|---|---|
| `positions_A` | (frames, atoms, 3) | Continuous, `unwrapped` Cartesian coordinates in Å |
| `time_fs` | (frames,) | Increasing, uniformly spaced production timestamps in fs |
| symbols | (atoms,) | Element symbols stored as strings |
| `masses_amu` | (atoms,) | Positive masses in atomic mass units |
| `unwrapped` | Scalar | Boolean True |
| `cell_A` | (3, 3) | Fixed-cell matrix included in forward MD exports |

Use production frames from a fixed-cell trajectory and maintain atom identity throughout. Store arrays without Python object pickling. Establish continuous coordinates during simulation or through a suitable unwrapping procedure before conversion.

### Run directory contents

Each forward calculation saves `config.json`, `report.json`, and input copies named `input_*.extxyz`. The report includes UTC timestamps, package versions, structure and checkpoint SHA-256 hashes, resolved paths, status, and workflow results or an error description.

| Workflow | Additional output files |
|---|---|
| MD | `production.traj`, `unwrapped.npz`, `thermodynamics.csv`, `md.log`, and initial-relaxation files |
| Trapping | Relaxed .extxyz structures, .traj optimization histories, and .log files for all four states |
| NEB | Endpoint-relaxation files, `neb.traj`, `neb.log`, and `neb_path.extxyz` |
| Tensile | `tensile.csv` and relaxed structures, trajectories, and logs for each strain step |

The interface provides result plots, a JSON download, and access to saved run reports. The standalone analyze command writes the requested JSON report directly.

## Validation and interpretation

The existing **22 automated tests passed on Python 3.12 under WSL on 2026-11-24**. They cover diffusion units and host-drift removal, invalid inputs, trapping-energy signs, model-hash checks, relaxation failures, MD sampling, an analytical NEB fixture, tensile-response reporting, interface forms, the synthetic demo, and automatic checkpoint-path selection. The tests exercise numerical logic with analytical fixtures, synthetic trajectories, and test potentials.

Archived CHGNet CPU records include energy, force, and stress evaluation, a short MD run, and a vacancy-trapping calculation through the interface. The short MD run records a withheld diffusion coefficient because its production duration and temperature sampling fail the reporting criteria. The example trapping result is approximately 0.1945 eV. These records document execution checks, with quantitative material validation requiring further work. Their configurations retain original local paths and need path updates for replay. [Validation records and scope, in Korean](docs/VALIDATION.md)

The calculation scope has several implications for research use.

- **Model applicability.** Establish checkpoint accuracy for the target elements, H environments, defects, and strains using reference calculations or experiments. The MACE adapter, GPU execution, and quantitative migration and tensile predictions require further validation.
- **Thermodynamic conditions.** Trapping, NEB, and tensile workflows use 0 K potential-energy calculations with fixed-cell atomic relaxation. Vibrational, entropic, zero-point, and hydrogen tunneling contributions require additional treatment.
- **Structural representation.** The supplied examples represent bcc single-crystal Fe and a vacancy. Studies of commercial steels require structures representing their intended composition, microstructure, and defect population.
- **Mechanical interpretation.** The tensile workflow measures a constrained atomistic stress response. Experimental ductility, fracture toughness, crack growth, HELP/HEDE mechanisms, and service-life assessment require additional evidence and methods.
- **Transport interpretation.** Classical MD, finite trajectory length, H concentration, and finite cell size set the scope of diffusion estimates. Independent seeds and convergence studies are needed for quantitative comparisons.

On the publishing PC, a Windows SciPy DLL import was blocked during the 2026-11-24 test attempt. The automated suite was completed in WSL. Windows installations should verify scientific-package imports and the model connection in the target environment.

## Development and repository layout

Install the test dependencies and run the existing suite from the repository folder.

~~~powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest -q
~~~

~~~text
hydrogen_lab/     CLI, browser interface, analysis, models, structures, and workflows
examples/         bcc Fe structures, JSON configurations, and synthetic trajectory
tests/            Numerical and interface tests
validation/       Archived execution-check records
docs/             Methods, provenance, and validation notes
START.cmd         Windows entry point
setup.ps1         Windows environment and checkpoint setup
launch.ps1        Windows server startup
SHA256SUMS.json   SHA-256 manifest for the distributed files
~~~

## Research provenance and licensing

Hydrogen Lab was developed using the author's hydrogen-local-energetics repository as a reference for calculation methods and checkpoint identities. That repository contains the accompanying research reproducibility materials. This application implements forward calculations on user-supplied structures and records the identity of each selected model. [Research reproducibility repository](https://github.com/lyullee/hydrogen-local-energetics)

The application performs inference with pretrained MLIPs, whose training provenance may include DFT calculations. Its calculation dependencies are the selected MLIP `backend` and ASE. Model weights are obtained separately through the explicit download command or a trusted local checkpoint.

The Hydrogen Lab source code is distributed under the MIT License. External packages and model weights carry their respective licenses. The registered MACE-OMat-0 checkpoint is identified under the Academic Software License in the model registry. Review upstream terms for the checkpoint and intended use. [Source license](LICENSE) · [Checkpoint registry](hydrogen_lab/registry.py)
