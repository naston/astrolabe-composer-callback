# Evaluation results — post-training benchmark logging

`astrolabe-callbacks` logs two kinds of metric:

- **During training** — the framework callbacks (`AstrolabeComposerLogger`, etc.) and the raw-loop `Run` stream `train/*` and `val/*` metrics as your model trains. These land on astrolabe's dashboard **Training tab**.
- **After training** — `log_eval_table` / `start_eval_run` log benchmark-suite results (GLUE, MMLU, custom held-out sets) under the `eval/<task_set>/<metric>` namespace on a *separate* Aim run. These land on the dashboard **Eval tab**.

This doc covers the second kind. It's the same library either way — your training/eval repo depends on `astrolabe-callbacks` and nothing else.

## The two namespaces

| Pattern | What it represents | Lives on | Dashboard |
|---|---|---|---|
| `val/<metric>` | During-training validation (every N batches) | the training run | Training tab (time-series) |
| `eval/<task_set>/<metric>` | Post-training benchmark suite | a **separate** eval run | Eval tab (table + chart) |

The split is deliberate: `val/*` answers "is training converging?" and lives on the training run's metric stream. `eval/<task_set>/<metric>` answers "how does the trained model do on this benchmark?" and lives on its own Aim run, tagged for discovery.

## Contract

An eval Aim run carries three identity tags so astrolabe's dashboard can discover it from the model-run page:

| Tag | Value |
|---|---|
| `astrolabe.kind` | `"eval"` (discriminator) |
| `astrolabe.task_set` | `"glue"`, `"mmlu"`, `"agent-rollouts-2026q2"`, … (section label) |
| `astrolabe.model_run_hash` | the training Aim run's hash (the join key) |

Metrics track under the path convention:

```
eval/<task>/<metric>
```

- segment 1: literal `eval` (routes to the Eval tab)
- segment 2: the task name (becomes a row in the table — `cola`, `sst2`, `stem`, …)
- segment 3: the metric label (becomes a column header — `matthews`, `accuracy`, …)

You don't have to remember any of this. The helpers below set the tags + emit the paths for you.

## Primary: `log_eval_table` (one-shot post-training)

The common case: you ran your benchmark, you have a dict of scores, you want them on the dashboard.

```python
from astrolabe_callbacks import log_eval_table

log_eval_table(
    model_run_hash="abc123...",    # the training Aim run's hash
    task_set="glue",
    rows={
        "cola": ("matthews",          0.822),
        "sst2": ("accuracy",          0.943),
        "mnli": ("accuracy_matched",  0.864),
        "avg":  ("mean",              0.876),
    },
)
```

That's the entire API for the 80% case. The helper opens an Aim run, sets the three identity tags, tracks each row under `eval/<task>/<metric>` at `step=0`, and closes the run. You get an Aim run hash back if you need it.

### Connecting to Aim

`log_eval_table` and `start_eval_run` accept an optional `aim_url`. It resolves the same way as every other entry point in this library:

1. `ASTROLABE_AIM_URL` environment variable (set by astrolabe on GPU instances), else
2. the `aim_url=` argument, else
3. `aim://localhost:43800` (the default SSH reverse tunnel astrolabe opens).

If your eval script runs on the same instance astrolabe provisioned for training, you can omit `aim_url` entirely — the env var is already set. Pass it explicitly only when running somewhere astrolabe didn't configure (e.g. `aim_url="aim://my-nuc:43800"`, or a filesystem path like `aim_url="/var/lib/astrolabe/aim"` for a script running directly on the NUC).

### The `avg` column

If you want an "average across tasks" column rendered on the dashboard table, **log it as a row**:

```python
rows={
    "cola": ("matthews", 0.822),
    "sst2": ("accuracy", 0.943),
    "mnli": ("accuracy_matched", 0.864),
    "avg":  ("mean", 0.876),   # ← rendered as the last column
}
```

The dashboard renders any row keyed `"avg"` as the last column by convention. The library doesn't compute the aggregate itself — that's your call. Mean? Harmonic mean? GLUE-paper-canonical subset? Whatever makes sense for your benchmark.

## Multi-seed: average before logging

If your benchmark runs N seeds and you want to report a single number per task, **compute the aggregate in your script before calling `log_eval_table`**:

```python
import statistics
from astrolabe_callbacks import log_eval_table

# Run N seeds, collect per-seed scores
scores_per_task = {"cola": [], "sst2": [], "mnli": []}
for seed in (0, 1, 2):
    seed_scores = run_glue_eval(model, seed=seed)
    for task, score in seed_scores.items():
        scores_per_task[task].append(score)

# Average before logging
log_eval_table(
    model_run_hash="abc123...",
    task_set="glue",
    rows={
        "cola": ("matthews",         statistics.mean(scores_per_task["cola"])),
        "sst2": ("accuracy",         statistics.mean(scores_per_task["sst2"])),
        "mnli": ("accuracy_matched", statistics.mean(scores_per_task["mnli"])),
    },
)
```

This boundary is deliberate. The library captures *one number per task per eval run*; the researcher decides what that number means. If you want to expose per-seed scores instead of averaged ones:

- **One eval run per seed** — call `log_eval_table` N times with distinct `task_set` labels like `glue-seed-0`, `glue-seed-1`. The dashboard surfaces each as its own section.
- **Distinct metric names per seed** — call `log_eval_table` once with rows like `("matthews_seed_0", 0.822)`, `("matthews_seed_1", 0.825)`. Each becomes its own column in the same task's row.

## Advanced: `start_eval_run` (streams + custom logging)

For mid-training rolling evals — running CoLA every 10K training steps to track convergence — you need finer control over the Aim run's lifecycle. Use the lower-level helper:

```python
from astrolabe_callbacks import start_eval_run

eval_run = start_eval_run(
    model_run_hash="abc123...",
    task_set="cola-trace",   # a different task_set than the one-shot 'glue' eval
)

# During training, periodically:
for checkpoint_step in (10_000, 20_000, 30_000, 40_000, 50_000):
    score = compute_cola_score(model_at(checkpoint_step))
    eval_run.track(score, name="eval/cola/matthews", step=checkpoint_step)

# When the eval session is complete:
eval_run.close()
```

The dashboard's Eval tab dispatches by data shape:

- **All metrics at `step=0`** → renders as a **table** (leaderboard view, one row per run).
- **Any metric with `step > 0`** → renders as a **trace** (chart with one line per run, x=step).

So `log_eval_table` always produces table blocks; `start_eval_run` with multi-step tracking produces trace blocks. If you want both views of the same task set (a final-step table + a convergence trace), emit them as two separate eval Aim runs with different `task_set` labels.

## What the dashboard does with this

On the experiment-detail page, astrolabe's **Eval tab** discovers all eval Aim runs that point at any run currently in scope (the experiment's selected version + any `--include` comparison runs). For each `(task_set, model_run)` pair, it renders one section:

- **Header**: `task_set` label, eval's `creation_time`, a summary value.
- **Body**: either a leaderboard table (one row per run, one column per task) or a trace chart (one line per run), dispatched by data shape.

If multiple eval runs exist for the same `(model_run, task_set)` pair (re-evaluating later with updated eval code, for instance), the dashboard shows the **newest by `creation_time`**. Older eval runs stay in Aim for forensics — not hidden, just not the default surface. This is the "re-eval = new session" semantic.

## Gotchas

- **No slashes in task names or metric labels.** The path convention is exactly three segments: `eval/<task>/<metric>`. A slash in either field scrambles the dashboard's column parsing. `log_eval_table` rejects this at the call site with `EvalInputError`.
- **`step=0` is load-bearing.** It's how the dashboard knows the eval is a one-shot, not a trace. If you call `track()` yourself via `start_eval_run`, use `step=0` for one-shot results.
- **Scores must be numeric.** `bool`, `None`, and strings are rejected. (Yes, `bool`-as-score is a Python footgun — `True` would silently log as `1.0` — so the helper rejects it explicitly.)
- **The model run hash is the training Aim run's hash, not its name.** If you can't get the hash directly, query Aim for the experiment's latest run.
- **Empty `rows` is rejected.** A blank eval section adds noise; if you have nothing to log, don't call the helper.
- **Forgetting to close the run** (`start_eval_run` only) leaves `end_time=0`, making the dashboard treat the eval as in-flight. `log_eval_table` handles close for you automatically — even when an exception is raised during the track loop.

## Where the helpers live

```python
from astrolabe_callbacks import (
    log_eval_table,        # primary, one-shot
    start_eval_run,        # escape hatch, streams + custom
    EvalInputError,        # raised on malformed input
)
```

These need only the base install (`pip install astrolabe-callbacks`) — no framework extra. Your eval script doesn't need a training-framework callback; it just needs to reach the Aim server (the `aim_url` convention above).

## See also

- [`docs/contract.md`](contract.md) — the train/val metric contract for the framework callbacks.
- The framework docs under [`docs/frameworks/`](frameworks/) — during-training logging per framework.
- astrolabe's dashboard **Eval tab** — the consumer surface this feeds.
