# Prototype: Knowledge Graph + Bayesian network + LLM explanation for milling root cause analysis

Prototype, not a validated diagnostic tool. Everything below is computed on **simulated** windows (plus two real checks); no real multi-cause labelled cuts exist.

## What it does

```
 Knowledge graph (KG)                Bayesian network (BN)                      LLM (explains only)
 causes -> mechanisms -> features    states: HEALTHY + 5 causes                 input: evidence bundle (JSON)
 sign, rationale, provenance tier    9 sensor nodes, Gaussian CPDs              output: summary, supporting evidence,
 verification actions                shared full covariance                     conflicts, caveats, checks (JSON)
        |                            edge mask from the KG (or data-validated)          |
        | structure prior            weights learned from data + bootstrap CI           | validated by code:
        +--------------------------> posterior over causes (temperature-calibrated)    |  - numbers must exist in evidence
                                     per-node contributions (sum = log-lik ratio)      |  - causes must be in the allowed set
                                     out-of-model check (Mahalanobis residual)         |  - no decision language
                                              |                                        |  - the ranking is never taken from the LLM
                                              +----------- evidence bundle -----------+
```
The ranking and probabilities always come from the BN. The LLM only verbalises them; its output is rejected and replaced by a rule-based text if it breaks any rule (`llm_reasoner.validate_reasoning`).

## Answer to "how to generate edge weights and connections?"

A Bayesian network does not come with its edges: someone has to supply the **structure** (connections) and the **parameters** (weights). The user's understanding that this is "already in the Bayesian network" is right for the parameters (they are learned by maximum likelihood) but not for the structure, which must come from somewhere.

| | How it is obtained here | How it is checked |
|---|---|---|
| **Connections** | proposed by the KG (physics reasoning, tier ASSUMED), compiled from `kg/milling_kg.json` to 19 direct cause -> sensor edges | tested against data: bootstrap CI (resampling *cuts*) of each weight; KG edges whose CI contains 0 are flagged for removal; non-KG pairs with a significant and large weight are flagged for addition **for an expert to approve** (`report/edges.csv`) |
| **Weights** | w(c,k) = (mean of node k under cause c - mean under healthy) / sd of k, fitted on the training cuts only | sign compared with the sign predicted by the KG; CI from 300 cut-level bootstrap resamples |

Important: these weights inherit the simulator's assumptions (gains, fault severities, nominal parameters). Real weights need real labelled cuts.

## Results (simulated, 5 causes, chance = 0.20; m = 10 windows; 300 draws per cause; test cuts held out)

| BN variant | edges | top-1 nominal | top-1 shifted generator | ECE nominal / shifted |
|---|---|---|---|---|
| KG as written | 19 | 0.38 | 0.24 | 0.21 / 0.08 |
| KG pruned by data (supported edges only) | 11 | 0.39 | 0.24 | 0.21 / 0.08 |
| **KG validated by data** (supported + data-suggested edges) | 23 | **0.84** | 0.44 | 0.03 / 0.38 |
| Fully connected | 45 | 0.88 | 0.45 | 0.03 / 0.36 |
| Shuffled edges, same out-degree (mean of 3) | 19 | 0.22 | 0.25 | 0.28 / 0.26 |

What this means (honest reading):
1. **The KG as written does not help inference**: 8 of its 19 edges are not supported (CI contains 0) and 12 other pairs are significant and large (`report/edges.csv`; 11 supported, 14 irrelevant). It is only slightly better than shuffled edges, which agrees with the earlier finding that the paper's 19-edge graph carries little information.
2. **KG -> data validation -> expert approval works as a workflow**: the validated KG recovers most of the accuracy of the fully connected model with half the edges, and every edge keeps its provenance and explanation for tracing. This is a statement about *simulated* data, where the simulator defines the truth.
3. **Edge weights alone are weak RCA scorers.** A matched filter on the weights gives top-1 0.38 / 0.31 (validated KG) and 0.44 / 0.33 (full); a CausalRCA-style weighted PageRank gives 0.25 / 0.25 and 0.20 / 0.20 (chance). The Bayesian posterior (which also uses the covariance between nodes) gives 0.84-0.88 nominal. So weights are necessary parameters of the RCA model but do not by themselves rank causes well.
4. **Calibration**: without the temperature, the product of correlated windows gives ECE up to 0.7; one scalar temperature fitted on validation cuts brings it to 0.03 for the data-validated models on the nominal generator, but under the shifted generator the confidence is again too high (ECE 0.36-0.38) while top-1 falls to 0.44-0.45. The "near 1.0" accuracy is circular, as in the paper.
5. **Out-of-model check (numeric, not LLM)**: threshold = 95th percentile of the Mahalanobis residual on nominal test draws (44.3). The real defective tool D* has residual 3472 (about 78x the threshold) although the BN assigns STIFFNESS posterior 1.00: the model is confident and wrong, and the check says so. The real *healthy* tool also exceeds the threshold (72, about 1.6x), a sign of the remaining simulation-real gap; the flag rate on the shifted simulated generator is only 9%, so the threshold is not a calibrated detector.

## Run

```
cd prototype_kg_bn_llm
python make_data.py        # once, minutes: simulated windows (needs ../cutting_sim and its real-data caches)
python real_nodes.py       # once: real D* and healthy-tool nodes
python experiment.py       # variants, edge table, calibration, misfit threshold -> report/
python run_demo.py         # 4 cases -> report/case_*.md, demo_table.md (rule-based explanations, nothing leaves the machine)
python -m unittest discover -s tests -v      # 10 tests
```

## Using a real LLM through OpenRouter (explicit consent each run)

1. Put your key **only** in `prototype_kg_bn_llm/.env` (git-ignored), copying `.env.example`; never paste it into chat or commit it. The code never prints it.
2. Review `report/evidence_*.json`: this is exactly what is sent (simulation outputs, real-data node shifts, KG text; no secrets).
3. `python run_demo.py --llm openrouter --send [--model anthropic/claude-haiku-5.5]`. Without `--send` the call is refused.
4. Model choice (OpenRouter model list, 13 Oct 2026): the Claude 5.x slugs (`anthropic/claude-haiku-5.5` cheap default, `anthropic/claude-sonnet-5.5`) support structured outputs but do not accept a temperature parameter, so runs are logged (`report/llm_calls/`, git-ignored: model, usage, prompt, response) instead of relying on determinism. The request uses `response_format: json_schema` with `strict` and `provider.require_parameters: true`.
5. If the LLM output fails validation, the report says so and uses the rule-based text; the validated flag is stored with each case.

**Not evaluated**: no LLM was called in this prototype's tests or results. The claim "the LLM explains faithfully" is therefore not supported by evidence yet; the validator only guarantees that numbers and cause names come from the evidence, not that the reasoning is correct. A human evaluation of the explanations (faithfulness, usefulness) is the next step.

## Continual learning across machines (simulated; `continual.py`, `report/continual.png`)

The concern: if the KG weights are updated from one machine's data, the foundation is forgotten and a new machine cannot adapt. Setup: 12 simulated machines, each with fixed DDE parameters (`make_machines.py`; between-machine weight distance is 15x the within-machine bootstrap spread); the base = weights fitted on the nominal generator; 3 machines held out as "new"; confirmed cases only; 3 random machine orderings; mean ± sd.

| Arm | what it does |
|---|---|
| naive overwrite (EMA) | running update of the weights (rate 0.05), overwrites |
| anchored to base | EMA plus a pull towards the base weights (L2-SP style) |
| pooled replay | keeps all cases (average over machines) |
| base + per-machine overlay (hierarchical) | w = w_global + delta_machine; global is a precision-weighted consolidation (base as prior); a new machine starts from the global layer |
| pseudo-label EMA | EMA with the model's own predictions as labels (not confirmed) |

Results (top-1 over 5 causes, chance 0.20):
* **Forgetting grows with the number of machines learned.** Retention on the machines already learned: naive overwrite 0.55 (L=1), 0.44 (L=3), 0.40 (L=6, equal to the untouched base 0.40); base + overlay 0.53, 0.55, 0.62 (specialist ceiling 0.58-0.67).
* **A new machine without confirmed cases is not helped by any arm** (k=0: 0.28-0.35, base 0.35); the naive arm is 3-7 points below the base, i.e. forgetting the foundation costs something.
* **From k=5-10 confirmed cases per cause on, base + overlay is best at L=6**: k=10: 0.446 ± 0.018 vs 0.381 ± 0.010 (adapt from the base, no history) vs 0.357 ± 0.031 (naive); at L=1 it is not better.
* **Pseudo-labels drift**: 0.26-0.27 on new machines (below the base) and retention 0.32.
* Edge-structure proposals (support on >= 6 of 9 machines): 0 non-base edges to add, 1 base edge to drop: the structure is shared; the machines differ in the weights.
* Caveats: machine identity must be known at inference to select the overlay; simulated machines share the same real-data healthy baseline and sensors; hierarchical gains are modest in absolute terms (new-machine accuracy stays well below the specialist ceiling 0.53); EWC reduces to the precision-weighted prior in this Gaussian model, so no separate arm.

## Retrieval layer (LightRAG-style, `retrieval.py`, `documents/corpus_demo.json`)
Entity/relation graph over a small corpus, dual-level keyword retrieval (entities vs relations/themes) plus one-hop expansion; no LLM indexing; it can only supply passages for the explanation and never changes the ranking. The demo corpus has 11 short passages (verbatim quotes read from the Sandvik page or quoted in search results; paraphrases are marked). It is not the `lightrag` library, which was not installed or tested. Documents we could not fetch are listed in `documents/DOWNLOAD_REQUEST.md`.

## Web app (`app/`): live milling simulation, alarm, root cause, explanation

```
cd prototype_kg_bn_llm
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000      # then open http://127.0.0.1:8000
```
Flow (all in the browser): **1** set the cutting condition and the explanation backend, **Start** -> **2** the machine runs: a schematic side view, a top view with the rotating tool whose centre follows the simulated vibration, four live sensor traces (tool-workpiece vibration, accelerometer, microphone, force) and the anomaly detector; the first 10 windows learn the machine's own healthy baseline -> **3** tick one fault, set severity and onset, **Apply**: the fault grows over a few windows -> when the detector (sensors only) crosses its threshold: **red lamp and alarm sound**; 10 more windows are collected -> **4** root-cause ranking (Bayesian network) with a causal graph (cause -> mechanism -> band groups, line width = learned weight, red = top cause) and an out-of-model check -> **5** explanation (rule-based or LLM via OpenRouter, validated, never changes the ranking).

**Where to put the OpenRouter key:** create `prototype_kg_bn_llm/.env` (copy `.env.example`) with the line `OPENROUTER_API_KEY=...`. It is read on the server per request (no restart) and never sent to the browser. Default model `google/gemma-3-4b-it` (small and cheap); change with `OPENROUTER_MODEL=...` in the same file. The LLM option also needs the consent checkbox on the page. A first real call with this model passed validation (`report/llm_calls/` keeps the log, git-ignored).

How it works and what is diagnosed:
* One generator for display, training and diagnosis (`app/live.py`): accelerometer and microphone = real-data healthy baseline + g·(faulty − healthy DDE run); vibration and force are the raw simulator outputs, display only. The simulator runs in the stable regime (modal stiffness x10), because at nominal parameters it is above its own stability limit.
* Ranking model (`cond24.py`): condition-aware Gaussian Bayesian network on 24 spectral features (the condition is a parent of the means), fitted on simulated cuts; temperature T for m = 10 windows. A 9-node version that follows the knowledge graph is much worse (top-1 0.48-0.58 against 0.80 in-distribution), so the KG traces and explains but does **not** constrain this model.
* Anomaly detector (`app/stream.py`, `app/calibrate_detector.py`): CUSUM of the Mahalanobis distance of each window from the machine's own healthy baseline (learned in the first 10 windows); thresholds tuned on healthy sessions with a hold-out check.
* Scripted checks: `python -m app.eval_matrix` (ranking), `python -m app.calibrate_detector` (detector). Numbers are shown in the page's About panel and below.

Honest results (simulator only, in-distribution unless stated): see `app/report/eval_matrix.json` and `app/report/detector.json`; the ranking reaches top-1 about 0.6 over the five modelled causes in the scripted matrix (0.795 in the benchmark, 0.47 when the simulator parameters are shifted); lower damping is the hardest cause; imbalance and run-out are not recognised as unknown faults. The page was checked through the API and a JavaScript parse; it has not been looked at in a browser by me.

## Knowledge graph built from documents (`kg_build.py`)
Rule extractor on the 11-passage demo corpus (cause -> mechanism -> sensor-feature triples and cause -> remedy, merged LightRAG-style with support counts, tier ASSUMED). The symptom -> sensor-feature step uses a hand-written bridge lexicon (documents say "vibration", not "acc_res"), the weakest link. Result (`report/auto_kg_eval.json`): 20 cause-feature edges; 11 shared with the hand-written KG; 55 % of the auto edges are supported by the data against 63 % for the hand-written KG, recall of the data-supported edges 0.35 vs 0.39. The LLM mode (prompt, ontology and output validator) is implemented but not run: it sends the passages to OpenRouter and waits for confirmation.

## Files

| File | Role |
|---|---|
| `kg/milling_kg.json`, `kg.py` | knowledge graph and its compiler to cause -> sensor edges, path tracing |
| `bn.py` | conditional-Gaussian BN, weights, contributions, temperature, weight-only scorers, ECE |
| `experiment.py` | all quantitative results and figures |
| `llm_reasoner.py` | evidence-bound reasoning layer, validators, backends (rule-based, OpenRouter) |
| `run_demo.py` | end-to-end cases and reports |
| `LITERATURE.md` | related work found and verified (KG for machining, KG-constrained Bayesian networks, LLM + KG for CNC) |
| `tests/` | unit tests |

## Limitations
* Simulated evidence only; the 9 sensor nodes come from the simulator-calibrated healthy baseline and an effect-size gain; the simulator's nominal parameters lie above its stability limit (see the paper).
* The KG covers 5 causes and 9 features; mechanism nodes are for tracing and explanation only (latent mechanism CPDs are not identifiable from these data); all relations are ASSUMED physics reasoning.
* Real D* is a defective tool of a different type; its true cause is not among the BN's states.
* Related work: KG-constrained causal Bayesian networks for root cause analysis exist (Wehner et al. 2024), and LLM + KG for CNC fault diagnosis exist; no KG specific to milling chatter was found in a limited search.
