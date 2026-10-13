# Deploy

**Static demo (GitHub Pages)** – `docs/index.html` replays recorded simulator sessions (fixed condition, severity 90 %). Enable: repo Settings → Pages → Branch `cutting-sim-experiments` (or main after merge), folder `/docs`. URL: https://danghaidang04.github.io/RCA-for-Chatter-Simulation/

**Full app** – needs Python (numba), so not GitHub Pages. Run the Docker image on a free host (Hugging Face Space of type Docker, Render, Fly). Set the secret `OPENROUTER_API_KEY` (and optionally `OPENROUTER_MODEL`) in the host's secret settings, never in the repo. Then open
`https://danghaidang04.github.io/RCA-for-Chatter-Simulation/live.html?api=https://<your-backend-url>`
to use the GitHub page as the front end with live simulation. Set `ALLOWED_ORIGINS` if the front end is on another domain.

**Data the backend needs** (git-ignored): `cutting_sim/output/real_windows.npz` (20 MB, derived from the MSM real-machine dataset: check its licence before publishing) and `prototype_kg_bn_llm/data/*.npz` is NOT needed at run time. Public hosting means anyone can spend your OpenRouter credit: keep the model cheap, set a spending cap on the key.

**Known limits shown in the demo**: top-1 root-cause ranking ~0.6 on the simulator (see README); the vibration detector misses IMBALANCE and RUNOUT; detector calibration is deferred (OPEN_ITEMS.md).
