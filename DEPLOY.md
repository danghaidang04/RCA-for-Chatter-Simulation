# Deploy

Two independent options. The static page needs no server; the full app needs Python.

## A. Static demo on GitHub Pages (recorded sessions, no LLM)
1. Repo → Settings → Pages → Source: *Deploy from a branch* → branch `cutting-sim-experiments` (or `main` after merging), folder `/docs` → Save.
2. After about a minute: `https://<user>.github.io/RCA-for-Chatter-Simulation/`.
3. After changing the app, rebuild the page: `cd prototype_kg_bn_llm && python -m app.build_static`, commit `docs/`, push.

## B. Full live app on a Hugging Face Docker Space (free CPU)
Needs a Hugging Face account. The backend needs `cutting_sim/output/real_windows.npz` (20 MB, derived from the MSM real-machine dataset, git-ignored). **Check the dataset licence before putting it in a public Space; use a private Space if unsure.**

1. `./deploy/make_space.sh` builds `build/space/` (about 20 MB: code, model tables, the data file split into 3 parts of at most 7 MB because Hugging Face rejects plain git files above 10 MiB; the Dockerfile joins and checks them, `Dockerfile`, `README.md` with the Space header).
2. On huggingface.co: New Space → SDK **Docker** → name it → Public or **Private**.
3. Push the folder:
   ```
   cd build/space
   git init -b main && git add . && git commit -m "milling monitor"
   git remote add space https://huggingface.co/spaces/<user>/<space-name>
   git push --force space main   # --force: the new Space already has its own starter README; ours replaces it
   # username = your HF user name, password = an HF access token with write permission
   ```
   (Do not paste the token anywhere else. Create it under Settings → Access Tokens.)
4. Space → Settings → *Variables and secrets* → New secret `OPENROUTER_API_KEY` = your key. Optional variable `OPENROUTER_MODEL`.
5. Wait for the build (a few minutes). The app is at `https://<user>-<space-name>.hf.space`.
6. Optional: use the GitHub Pages front end with this backend: `https://<user>.github.io/RCA-for-Chatter-Simulation/live.html?api=https://<user>-<space-name>.hf.space` (the backend allows the `github.io` origin by default; set `ALLOWED_ORIGINS` for another domain).

### Public link (anyone can use it)
- Create the Space as **Public** (or switch it in Settings → Change visibility). Everyone with the link can then use it without an account.
- The app protects your credit and the free CPU: each visitor can use the LLM at most `LLM_PER_IP_HOUR` times per hour (default in the Dockerfile: 0 = no limit) and the whole demo `LLM_PER_DAY` times per day (0 = no limit); when a limit is set and reached, the page says so and uses the rule-based text. At most `MAX_SESSIONS` machines (Dockerfile default 8; 0 = no limit) run at once and idle ones are dropped after `SESSION_IDLE_S` seconds (default 600); extra visitors see "The demo is busy". Change them under Space → Settings → Variables.
- Also put a spending limit on the OpenRouter key itself (openrouter.ai → Keys), because the limits above are in memory and reset when the Space restarts.
- The packaged data file comes from the MSM dataset; check its licence allows redistribution before making the Space public.

### Things to know
- A public Space lets anyone press the LLM button and spend your OpenRouter credit. Use a cheap model, put a spending cap on the key, or keep the Space private.
- Free Spaces may sleep when idle; the first request then takes about a minute (the model loads in about 15 to 25 s).
- The Docker image and the packaged folder were checked locally (clean Python environment, server start, `/api/sld`, a simulation step), but the Docker build itself has not been run on a Hugging Face builder.
- Other hosts (Render, Fly, a VPS): build the root `Dockerfile` the same way; it listens on `$PORT` (default 7860).

## Local
`./run_app.sh` then http://127.0.0.1:8000. Key in `prototype_kg_bn_llm/.env`.
