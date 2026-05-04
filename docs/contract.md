# Contract

What every framework callback in `astrolabe-callbacks` guarantees, and what it deliberately doesn't.

## Required behavior

These four behaviors hold for **every** framework callback (Composer, Lightning, HF Trainer, raw PyTorch). If a callback violates any of these, it's a bug — file an issue.

### 1. Honors `ASTROLABE_EXPERIMENT_NAME`

When the env var is set, it wins over any constructor argument. Astrolabe sets it on every orchestrated run; the callback picks it up automatically. Empty-string env values fall through to the constructor arg (so accidentally-empty shells don't break standalone use).

### 2. Honors `AIM_RUN_TAGS`

Format: `key1=val1,key2=val2,...`. Whitespace tolerated. Each key/value becomes a tag on the Aim run. Astrolabe writes `astrolabe.experiment`, `astrolabe.version`, `astrolabe.submit_id`, `astrolabe.user` here on every orchestrated run. Standalone users can write whatever they want.

### 3. Connects to `ASTROLABE_AIM_URL`

Defaults to `aim://localhost:43800` (the standard astrolabe SSH-tunneled URL). Override via env or constructor arg.

### 4. Closes the run cleanly at end of training

`astrolabe.status` tag is written before the Aim run is closed:

- `completed` — training finished cleanly (the framework's "fit end" hook fired)
- `failed` — training raised an exception (the framework's exception hook fired)
- `interrupted` — reserved for explicit user-side cancellation; not written automatically

Close failures are silent — by the time we close, the data has been streamed and a close failure is cosmetic.

## Pass-through philosophy

> **We log every metric you produce. We synthesize one (`wall_time`). We never invent the rest.**

This is the core contract. The callback exists to forward your metrics to Aim with astrolabe's tag conventions applied — not to decide what's worth logging.

| Framework | How user metrics flow through |
|---|---|
| Composer | `LoggerDestination.log_metrics(metrics, step)` receives every `logger.log_metrics(...)` call from your training YAML and any user code |
| Lightning | `on_train_batch_end` and `on_validation_end` scan `trainer.callback_metrics` (populated by `self.log(...)` in your LightningModule) |
| HF Trainer | `on_log(logs)` and `on_evaluate(metrics)` receive HF's pre-aggregated dicts |
| Raw PyTorch | You call `run.log_train(**metrics)` / `run.log_eval(**metrics)` / `run.log(name, value)` explicitly |

### Names we rewrite (cosmetic only)

Each framework emits a few well-known metric names that users don't choose. We rename those for cleaner display:

- **Composer**: `loss/train/total` → `train/loss`, `metrics/train/<x>` → `train/<x>`, `metrics/eval/<x>` → `eval/<x>`
- **Lightning**: `val_<x>` and `val/<x>` → `eval/<x>` (keeps the eval namespace consistent with the other frameworks)
- **HF Trainer**: `loss` → `train/loss`, `learning_rate` → `train/lr`, `grad_norm`/`epoch` → `train/<x>`, `eval_<x>` → `eval/<x>`

User-chosen names are **never** rewritten. If you call `self.log("my_thing/foo", x)` in a LightningModule, it lands in Aim as `my_thing/foo`. If you log `MaskedLanguagePerplexity` in a Composer eval suite, it lands as `eval/MaskedLanguagePerplexity` (the `metrics/eval/` prefix Composer emitted is renamed; the `MaskedLanguagePerplexity` name is yours).

### Eval namespace forward-compatibility

In v0.2.0 the eval namespace is `eval/`. In v1.0.0 it flips to `val/` to align with astrolabe v1.7's eval-runs schema (which separates *during-training* eval — what this library writes — from *post-training* `eval_results.json` evaluation). The flip is a single-line change in `_core.EVAL_METRIC_PREFIX`; everything cascades.

If you're starting fresh today, write to the eval namespace and accept that it will be renamed in a future version. If you have existing dashboards keying on `eval/`, plan to update them around v1.0.0.

## Failure handling

By default, **the callback never crashes training.** The cost of training time is too high to drop a run because of a logging hiccup. Three failure modes:

### Connection failures

Examples: `aim` not installed, Aim server unreachable, network partition during init.

**Default behavior**: a single `WARNING` log line is emitted; the callback degrades to a no-op for the rest of the run. Subsequent metric writes silently skip; close is a no-op.

**Strict mode (`ASTROLABE_CALLBACK_STRICT=1`)**: raises `RuntimeError`. Use this in CI pipelines where silent metric loss is a worse failure than a crash.

### Per-metric track failures

Examples: Aim server dies mid-run, malformed metric value (NaN/inf), schema rejection.

**Default behavior**: failure logged at `DEBUG` level *once per metric name* (rate-limited via a per-run set of complained-about names). Training continues. You'll see the warning on the first failure for each metric; subsequent failures of the same metric are silent. Other metrics keep flowing.

**Strict mode**: re-raises whatever Aim raised. Training crashes.

### Close failures

Examples: Aim server gone at end of training, network failure during close.

**Default behavior**: silent. By the time close runs, the data has been streamed; a close failure is cosmetic.

**Strict mode**: still silent. The cost of crashing at end-of-training to surface a close failure isn't worth it; logged at `DEBUG`.

## Distributed training

Every callback gates Aim writes on rank-zero. Detection order:

1. `torch.distributed.is_initialized()` — if true, uses `dist.get_rank() == 0`
2. `RANK` env var — set by `torchrun` and most cluster launchers
3. `LOCAL_RANK` env var — set per-node
4. Falls back to `True` (treat as rank-zero) for single-process training

Non-rank-zero processes never open a run, never write metrics, never close. They participate in the framework's lifecycle hooks but no-op every Aim interaction. N processes produce one Aim run.

## What this library deliberately does NOT do

- **No metric whitelists.** We don't pick which of your metrics are "important enough" to log.
- **No metric aggregation.** Each framework already aggregates per-batch metrics into per-step or per-epoch scalars; we trust that.
- **No re-namespacing of user-chosen names.** Your `MyCustomThroughput` lands as `MyCustomThroughput`.
- **No multi-Aim-backend support.** One Aim URL per run; choose between the SSH tunnel and a direct connection at config time.
- **No checkpoint upload.** Use astrolabe's git tag provenance or your training framework's checkpoint sink.
- **No auto-detection of the framework.** You import the class that matches your framework explicitly. The base install (no extras) doesn't pull in any framework deps.
