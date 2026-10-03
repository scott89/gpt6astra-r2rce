# GPT-6-Astra R2R-CE reproduction, with an API model in place of GPT

An independent re-implementation of the evaluation protocol in
*"GPT-6-Astra Lights Up Embodied Navigation: Evaluation in Zero-Shot Vision-and-Language
Navigation in Continuous Environments"* (arXiv:2609.29861), runnable against any
OpenAI-compatible chat endpoint. No weights are downloaded and no training is involved: the
agent is a zero-shot VLM prompted over monocular RGB frames.

**Headline (this repo, `glm-5.3-flash`, 100 episodes, no guardrails):**
SR 27.0% · OS 35.0% · NE 7.41 m · SPL 22.0% — see [Results](#results) for the comparison to
the paper and for what is and is not comparable.

## The protocol being reproduced

Everything below is fixed by the paper and enforced by `configs/task_r2rce_512.yaml`:

| | |
|---|---|
| Observation | single 512×512 monocular RGB, `hfov 79°`; no depth, no map, no pose, no reference path |
| Action space | discrete: `MOVE_FORWARD` 0.25 m · `TURN_LEFT`/`TURN_RIGHT` 15° · `LOOK_UP`/`LOOK_DOWN` 30° · `STOP` |
| Budget | 500 primitives per episode, 2400 s wall clock |
| Success | agent must *call* `STOP` and end within 3 m geodesic of the goal |
| Data | R2R-CE `val_unseen` (R2R_VLNCE_v1-3_preprocessed), Matterport3D |
| Metrics | SR / OS / NE / SPL, from habitat's native measurements (not recomputed) |

The model answers one JSON object per decision — `{"thought": ..., "actions": [...]}` — and the
harness executes that program against the simulator, then asks again.

## Layout

```
configs/task_r2rce_512.yaml   the paper's interface: sensors, action space, budgets, metrics
zsvln/backend.py              ChatBackend — the single model-swap point (OpenAI-compatible)
zsvln/policy.py               prompt assembly, JSON parsing, optional safety guardrails
zsvln/runner.py               habitat env construction, per-episode loop, metric aggregation
scripts/run_eval.py           CLI: episodes, ordering, sharding, resume, report-only
scripts/paired_compare.py     per-episode A/B of two run dirs (sign test + paired bootstrap)
scripts/make_subset.py        builds the stratified 100-episode subset (subset/r2rce100.json)
scripts/selfcheck.py          20 offline checks: parsing, budget truncation, prompt shape
runs/                         committed result traces (per-episode JSON + stats.json per run)
```

`habitat_extensions` (dataset/measurement/sensor registration) is imported from an existing
checkout by path via `--ext-lib`; this repo does not fork or modify it.

## Setup

Requires `habitat-sim` + `habitat-lab` 0.1.7 and the R2R-CE data. Developed on conda env
`lavira`.

```bash
# data: expect <repo>/data/scene_datasets/mp3d/*.glb and the preprocessed R2R_VLNCE json.gz
ln -s /path/to/your/data ./data           # ./data is gitignored

# credentials: never commit the real key
cp .env.example .env                      # .env is gitignored, chmod 600
echo 'OPENAI_API_KEY=sk-...' >> .env
```

Sanity-check the simulator and prompt path without spending any API quota:

```bash
python scripts/selfcheck.py                                   # parsing / prompt / guardrails
python scripts/run_eval.py --policy scripted --episodes 2 \
  --max-actions 40 --tag scripted-check                       # renders real frames, computes metrics
```

## Running the evaluation

```bash
# 4-episode smoke test across distinct scenes
python scripts/run_eval.py --policy vlm --episodes 4 --tag glm-smoke

# the full 100-episode run, 5-way parallel, then merge the shards into one stats.json
for i in 0 1 2 3 4; do
  python scripts/run_eval.py --policy vlm --episodes 100 --shard $i/5 \
    --max-consecutive-forward 0 --no-stop-verification \
    --reasoning-effort minimal --tag glm-full --resume > runs/glm-full.s$i.log 2>&1 &
done
python scripts/run_eval.py --report-only --tag glm-full
```

`--order round-robin` (default) interleaves scenes so that `--episodes N` samples N different
buildings instead of N episodes of whichever scene sorts first. `--resume` skips episode files
already present, so `--shard I/N` shards are safe to restart independently.

Flags that matter most:

| flag | meaning |
|---|---|
| `--model`, `--base-url`, `--api-key-env` | any OpenAI-compatible endpoint; defaults to Volcengine Ark `glm-5.3-flash` |
| `--reasoning-effort {minimal,low,medium,high,none}` | thinking budget. Ark rejects `enable_thinking:false` and `thinking:{type:"disabled"}`, so this is the only working knob |
| `--max-tokens N` | completion budget (default 512). Reasoning tokens are billed against it — `medium`/`high` need ≥2048 or the reply comes back empty with `finish_reason=length` |
| `--history-frames N` | RGB frames attached per decision (default 4) |
| `--max-program N` | longest accepted action program per decision (default 8) |
| `--shard I/N`, `--only ids`, `--report-only` | fan-out / restrict / re-aggregate a run directory |

Headless rendering is set up in-process (`EGL_PLATFORM=surfaceless`, `DISPLAY` unset,
`MAGNUM_LOG=quiet`, `GLOG_minloglevel=2`).

## Results

`runs/glm-full/` — 100 episodes, 5 parallel shards, ungarded prompt, `reasoning_effort=minimal`.
5,456 API calls, 10.88 M prompt + 0.44 M completion tokens, 0 backend failures, ~65 min wall clock
at 5-way (GPU headroom was never the constraint: ~270 MiB VRAM per simulator).

| | SR | OS | NE | SPL |
|---|---|---|---|---|
| **glm-5.3-flash (this repo)** | **27.0%** | 35.0% | 7.41 m | **22.0%** |
| same, `reasoning_effort=medium`† | 34.7% | 39.8% | 6.84 m | 29.5% |
| GPT-6-Astra (ultra), paper | 81.3% | — | — | 71.5% |
| GPT-6-Astra (medium), paper | 75.7% | — | — | 65.6% |
| SmartWay (trained), quoted in paper | 29% | 51% (oracle) | — | 22.46% |
| Open-Nav / Llama-3.1, quoted in paper | 16% | — | — | 12.90% |

† `runs/glm-med100/`, 98 of the same 100 episodes (two lost, see [Known issues](#known-issues));
paired per episode against the row above, not resampled.

SR's 95 % bootstrap CI is [19 %, 36 %]. Per-scene SR ranges 10 %–50 % (`8194nk5LbLH` best,
`EU6Fwq7SyZv`/`QUCTc6BB5sX`/`oLBMNvg9in8` worst at 10 %).

Structure of the failures:
- **99/100 episodes end by an explicit `STOP`** — hesitation/budget starvation is not the
  bottleneck; *where* it stops is. One episode exhausted the 500-action budget.
- Bimodal effort: 61 episodes finish in ≤120 actions (SR 30 %), 28 burn >300 actions (SR 14 %).
- 15 of the 73 failures are 3–5 m short of the gate.

### Reasoning effort, measured (paired, n=98)

`scripts/paired_compare.py runs/glm-full runs/glm-med100` — same episodes, one variable:

| metric | minimal | medium | delta | 95 % CI of paired delta |
|---|---|---|---|---|
| SR | 27.6 % | 34.7 % | **+7.1 pp** | [−3.1, +17.3], flips 17:10, exact p = 0.248 |
| OS | 35.7 % | 39.8 % | +4.1 pp | [−7.1, +15.3] |
| SPL | 22.5 % | 29.5 % | +7.0 pp | [−1.2, +15.3] |
| NE | 7.46 m | 6.84 m | −0.62 m | [−2.12, +0.80] |
| API latency / episode | 121 s | 224 s | **+104 s** | [+68, +140] — the only significant term |

Every metric leans medium (and >300-action thrash falls 28 → 21, near-misses 3–5 m fall 14 → 11),
but the paired sign test cannot separate +7 pp from noise at this n. The cost side is unambiguous:
completion tokens 0.44 M → 1.14 M with reasoning 146 k → 760 k (5.2×), API time 3.32 → 6.11
process-hours, wall clock 65 → 140 min — while the *prompt* bill barely moves (10.88 M → 11.44 M).

The usable ladder on this endpoint, measured on a text probe: `minimal` = `low` < `medium` <
`high`, and omitting the parameter (or any unknown value — the endpoint does not validate it)
selects the *heaviest* thinking mode. `--reasoning-effort none` therefore does not disable
thinking; it enables the default maximum and truncates the reply at `--max-tokens 512`.

### Why 27 % ≠ 81.3 % is not a clean model comparison

1. `reasoning_effort=minimal`. The paper's headline row is its *ultra* reasoning setting. Raising
   it recovers about 7 pp SR (above), so the setting matters but does not bridge the gap; deeper
   modes also need `--max-tokens` ≥ 2048 or the completion window fills with thinking.
2. The 100 tasks are a locally built stratified sample (10 largest `val_unseen` scenes × 10 evenly
   strided episodes), not the paper's undisclosed fixed 100-task list.
3. Memory is the last 4 RGB frames plus a compacted action trace; the paper's agent carries an
   explicit spatial memory.

## Guardrails: a negative result worth keeping

Two guardrails were implemented and measured — a cap on consecutive blind `MOVE_FORWARD`, and a
look-before-you-leap camera re-check inserted before the first proposed `STOP`:

| config (same 10 episodes) | SR | OS | NE | SPL | prompt tokens |
|---|---|---|---|---|---|
| none, serial | 0.60 | 0.70 | 6.36 | 0.429 | 448 k |
| none, 5-way parallel | 0.40 | 0.50 | 5.82 | 0.324 | 998 k |
| forward cap only | 0.30 | 0.30 | 6.72 | 0.264 | 787 k |
| both guardrails | 0.30 | 0.40 | 5.47 | 0.201 | 960 k |

The decisive observation is row 1 vs row 2: **the same ungarded configuration, re-run, flipped
success in 6 of 10 episodes.** At `temperature=0` the trajectory still diverges on any perturbation,
so ±0.2 SR at n=10 is noise, and neither guardrail can be shown to help or hurt at that sample size.
Both remain in the code, off by default (`--max-consecutive-forward 0 --no-stop-verification`), as
the configuration the full run used.

Mechanisms worth knowing before re-enabling them:
- Truncating a forward chain forces a mid-transit re-observation, and the model then tends to step
  again — overshooting past the 3 m gate; in one episode it degraded into 214 re-decisions that
  consumed the whole budget.
- Injected `LOOK_DOWN`/`LOOK_UP` get imitated: healthy episodes show the 2 injected tilts, broken
  ones showed 34 / 184 model-initiated tilts.

## Reproducing these numbers

`runs/*/stats.json` and every per-episode trace (thought text, executed actions, latency, token
counts, termination reason) are committed, so the aggregates above can be recomputed without
re-running anything:

```bash
python scripts/run_eval.py --report-only --tag glm-full            # rebuild one run's stats.json
python scripts/paired_compare.py runs/glm-full runs/glm-med100     # per-episode A/B + significance
```

## Known issues

- **Two episodes are missing from `runs/glm-med100/` (601, 1687), so that run is n=98.** Two
  independent causes, both now fixed or worked around:
  1. `--resume` used to drop completed episodes *before* `--shard` sliced the list, and all shards
     share one `episodes/` directory. Each worker therefore filtered a differently-sized global
     list, its slice shifted, one episode got queued twice and another fell out of every slice.
     Fixed by slicing first and filtering after; 5-way now partitions the 100 exactly once.
  2. With `reasoning_effort=medium`, two decisions produced 8.5 k / 9.4 k characters of thinking
     that exhausted `--max-tokens 2048`, returning `content=""` / `finish_reason=length`; after 6
     retries the episode is dropped and the sweep continues. Recover with
     `--only 601,1687 --max-tokens 4096 --resume`.
  Both episodes failed under `minimal` (NE 3.66 m / 6.69 m), so counting them as failures would move
  medium's SR 34.7 % → 34.0 % — immaterial to the comparison above.
- `record["backend_stats"]` in run directories created before the fix (`glm-full` and earlier) is a
  *shard-cumulative* snapshot, not per-episode; summing it over-attributes cost by roughly 10×.
  The episode-level `prompt_tokens` / `api_latency_s` fields and `stats.json` were always correct.
  `runs/glm-med100/` onward carries true per-episode values.
- Simulator processes share one GPU by design; VRAM scales with the scene's GLB, not with the
  policy, so concurrency is limited by endpoint rate limits rather than the card (~270 MiB per sim,
  5-way ≈ 3.1 GiB of 32 GiB).
