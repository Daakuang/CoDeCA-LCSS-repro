# CoDeCA L-CSS reproduction code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21757462.svg)](https://doi.org/10.5281/zenodo.21757462)

This repository contains the code and numerical data for the manuscript
"Deployment-Aware Controller and Control Architecture Co-Design via
Mixed-Integer Output-Feedback SLS."

The release covers the finite three-follower platoon study in the paper:

- joint actuator, sensor-package, and communication-service selection;
- hard-budget mixed-integer OF-SLS synthesis;
- PBH and quadratically invariant comparison families;
- canonical platoon information-flow baselines;
- regularization-for-design (RFD) paths and fixed-architecture re-synthesis;
- FIR-horizon, fixed-hardware, and service-menu sensitivity data; and
- generation of the reported comparison figure with Matplotlib.

The repository contains only the paper-specific finite formulation and its
standard solver path. Experimental acceleration algorithms under separate
development are not part of this release.

## Archived numerical evidence

The locked result files are under `results/lcss_v2/joint_platoon/`. The main
files are:

- `publication.json`: sanitized machine-readable publication result;
- `platoon_budget_comparison_joint_source_data.json`: plotted source data;
- `platoon_budget_comparison_joint.pdf` and `.png`: final figure;
- `rfd_native_scan.json`: the 16-by-11 native RFD path; and
- `baseline_envelopes.json` and `qi_envelope.json`: comparison envelopes; and
- `case_sensitivity.json`: fixed-hardware and delay-three sensitivity results.

`PUBLIC_RELEASE_MANIFEST.json` records the SHA-256 hash and byte count of every
file in the release.

## Environment

The locked run used Python 3.13.3 on Windows 11 with NumPy 2.2.6, SciPy 1.15.3,
Matplotlib 3.10.3, pytest 9.0.2, CVXPY 1.7.x, and Gurobi/gurobipy 13.0.1.
The optimization runs require a working Gurobi license.

Create an isolated environment and install the package:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[test]"
```

`requirements-lock.txt` records the exact package versions used for the locked
run.

## Validate the public package

```powershell
.venv\Scripts\python.exe -m pytest -q
```

The tests check the plant, deployment menus, OF-SLS timing, hardware and
service gates, PBH/QI enumeration, fixed-architecture re-synthesis, and figure
schema. Tests that invoke Gurobi require the licensed environment.

## Recreate the figure from archived data

```powershell
.venv\Scripts\python.exe -m repro.joint_plotting `
  --publication-source results/lcss_v2/joint_platoon/publication.json `
  --output-directory results/lcss_v2/reproduced_joint
```

## Recompute the main optimization

The full run uses seed 23, one Gurobi thread, and FIR horizon 10:

```powershell
.venv\Scripts\python.exe -m repro.joint_publication `
  --repository-root . `
  --no-resume `
  --no-plot `
  --output results/lcss_v2/reproduced_joint/publication.json `
  --seed 23 `
  --threads 1 `
  --horizon 10
```

Compute the native RFD path and merge it into the recomputed result:

```powershell
.venv\Scripts\python.exe -m repro.rfd_joint_experiment `
  --output results/lcss_v2/reproduced_joint/rfd_native_scan.json

.venv\Scripts\python.exe -m repro.merge_native_rfd `
  --publication results/lcss_v2/reproduced_joint/publication.json `
  --rfd results/lcss_v2/reproduced_joint/rfd_native_scan.json
```

Then run `repro.joint_plotting` on the recomputed `publication.json`.

Recompute the fixed-hardware and delay-three sensitivity cases with the same
standard formulation:

```powershell
.venv\Scripts\python.exe -m repro.case_sensitivity `
  --output results/lcss_v2/reproduced_sensitivity/case_sensitivity.json
```

The complete catalog is computationally substantial. The archived result and
source-data files allow the reported points and figure to be checked without
rerunning every mixed-integer and fixed-architecture problem.

## License

The code is released under the MIT License. The archived numerical data and
figures may be reused with citation to the associated release and article.
