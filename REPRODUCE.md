# Reproducing the results

Everything in `paper/main.pdf` is generated from result files in `cutting_sim/output/` by `paper/make_tables.py` and
`paper/make_figures.py`. Numbers are never typed by hand, except those marked red (placeholders, see `paper/PLACEHOLDERS.md`).

## 1. Environment
```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1     # avoids BLAS oversubscription inside multiprocessing pools (otherwise ~10x slower)
```
Tested with Python 3.13.12 on macOS (Apple silicon, 10 cores). Scripts use `multiprocessing` with 10 workers and numba (first call compiles; cached afterwards).

## 2. Data (not redistributed)
| Dataset | Source | Place in |
|---|---|---|
| MSM (Kim et al., Sci. Data 2026), datasets 3, 5, 6, 14 and label files | Kaggle, DOI 10.34740/kaggle/ds/8392825 (CC BY 4.0; needs a Kaggle account) | `data/msm/` |
| LUH (Denkena et al., Data in Brief 2023, DOI 10.1016/j.dib.2023.109574) | repository named in the article | `data/luh_raw/h5/` (405 `.h5` files) |
`data/` is git-ignored. Dataset 14 is large (about 1.2 GB for the accelerometer file).
Never commit credentials; use your own Kaggle token via `~/.kaggle/kaggle.json`.

## 3. Run order (from `cutting_sim/`)
| Step | Command | Output |
|---|---|---|
| real features/windows | `python real_features.py 5 6 ; python real_windows.py 5 6` | `output/real_windows.npz` |
| healthy-baseline C2ST (3 splits) | `for s in 0 1 2; do NONLIN=1 SEED=$s python noise_model.py; NONLIN=0 SEED=$s python noise_model.py; done` | `output/noise_model_gate_nonlin*_seed*.json` |
| fault gains | `python rca_gain2.py` | `output/rca_gain_effect.json` |
| causal RCA (main) | `python rca_causal.py` | `output/rca_causal.json` |
| ablations (3 seeds) | `python rca_ablation.py` | `output/rca_ablation.json` |
| classifiers / gain sweep | `python rca_run.py ; python rca_sweep.py` | `output/rca_results.json`, `rca_sweep.json` |
| real defective tool | `python real_windows.py real_windows_defect.npz 3 14 ; python real_defect_check.py` | `output/real_defect_check.json` |
| chipped-flute test | `python defect_mechanism_check.py` | `output/defect_mechanism_check.json` |
| operating regime | `python regime_check.py` | `output/regime_check.json` |
| stiffer-regime robustness | `KSCALE=10 TAG=_k10 python rca_gain2.py && KSCALE=10 TAG=_k10 python rca_causal.py` | `output/rca_causal_k10.json` |
| LUH cross-machine | `python luh_transfer.py ; python luh_baselines.py ; python luh_dynamics_shift.py` | `output/luh_*.json` |
| sim-to-real wear | `for t in 100 60 40; do THR=$t python sim_to_real_wear.py; done` | `output/sim_to_real_wear*.json` |
| FRF import test | `python test_frf_import.py` | console |
Then, from the repository root: `make paper`.

## 4. Seeds, determinism, caveats
* Seeds are fixed in the scripts (`SEED` environment variable for the C2ST splits; ablation uses three seeds). Gradient boosting and logistic regression are deterministic; DANN uses 10 seeds.
* Wall-clock times were not recorded systematically; most steps take minutes, `rca_ablation.py` and dataset downloads take longest.
* `rca_causal.py` and `rca_ablation.py` import the RCA algorithms from `run_full_paper_benchmark.py` at the repository root. These are **simplified re-implementations**, not the authors' code (see the paper, Table "RCA algorithms").
* LUH raw features are cached in `cutting_sim/output/luh_features.npz`; delete to recompute.
* Environment variables: `KSCALE` (modal-stiffness multiplier for the robustness check), `TAG` (suffix of result files), `THR` (wear threshold), `NONLIN`/`SEED` (baseline model variant and split).

## 5. Checks
`make test` runs the FRF import test; `python -c "import rca_causal"` from `cutting_sim/` checks that imports resolve.
