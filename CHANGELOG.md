# Changelog

## v1.0.0 — unreleased

### Breaking changes

- **`EVAL_METRIC_PREFIX` flipped from `"eval"` to `"val"`.** During-training validation metrics emitted by Composer (`metrics/eval/*`), HuggingFace Trainer (`eval_*`), Lightning (`val_*` / `val/*`), and the raw-PyTorch `Run.log_eval()` helper now land in Aim as `val/<name>` instead of `eval/<name>`. The change aligns with astrolabe v1.7's eval-runs schema, which uses `eval/<task_set>/<metric>` exclusively for post-training benchmark suites tracked on separate eval Aim runs (`astrolabe.kind="eval"`). The split puts during-training validation on the dashboard's Training tab and benchmark results on the new Eval tab — sharing the `eval/` prefix made them visually indistinguishable.

  **Migration**: training repos that pin this package update their pin and re-run; the new prefix lands automatically. Legacy production runs already in Aim keep their `eval/*` metric names — those metrics still chart correctly on the Training tab, they just sit under a now-deprecated prefix. No retroactive migration.

  **Why a major version**: anyone reading metric names back (custom dashboards, downstream tools that parse the `eval/` prefix) breaks. Single-call-site fix (`s/eval\//val\//g` on the consumer side) but it warrants a major.

## v0.2.2 — unreleased

- **Buffer heartbeat logging.** The write-buffer's drainer thread now emits a periodic INFO snapshot (default every 5 minutes, configurable) so operators tailing logs can see how buffer health evolves over the lifetime of a run — `Aim buffer (heartbeat): N submitted, N drained, N retried, N dropped, queue depth N`. Skipped during quiet periods (no counter changed since last heartbeat) so a healthy idle drainer doesn't clutter logs. Pure observability; no notifications, no failure-handling changes. Motivation: previously a degrading buffer was invisible until `close_run`'s post-mortem summary; now you get a timeline to correlate against network events.

## v0.2.0 — 2026-05-04

**Renamed**: `astrolabe-composer-callback` → `astrolabe-callbacks`. The new package supports four ML training frameworks instead of one.

### What's new

- **Four framework callbacks** behind optional extras:
  - `astrolabe-callbacks[composer]` — `AstrolabeComposerLogger` (was `AstrolabeLogger`)
  - `astrolabe-callbacks[lightning]` — `AstrolabeLightningLogger` (new)
  - `astrolabe-callbacks[hf]` — `AstrolabeHFTrainerCallback` (new)
  - `astrolabe-callbacks[all]` — everything
- **`Run` context manager** for raw PyTorch / JAX / Accelerate / custom loops. Same env-var contract; `with Run() as run: run.log_train(loss=x, step=n)`.
- **Pass-through philosophy**: every metric the user logs flows to Aim under the user's chosen name. The library only synthesizes `wall_time`. No metric whitelists, no surprise prefixes. Replaces the v0.1.x behavior where only `train/loss` and a few framework metrics were logged.
- **Strict mode**: `ASTROLABE_CALLBACK_STRICT=1` flips graceful-degrade warnings into raised exceptions for fail-fast CI behavior.
- **Centralized eval namespace**: `_core.EVAL_METRIC_PREFIX` is the single point of truth — flips from `eval/` to `val/` in v1.0.0 alongside astrolabe v1.7's eval-runs schema. One-line change cascades to every framework.
- **Rank-zero gating** built into every callback. N-process distributed runs produce one Aim run, not N.
- **`track_safely` failures now log at `WARNING`** (rate-limited to once per metric name) instead of `DEBUG`. The previous default level meant `track()` failures were invisible under loguru's stock `INFO` filter — exactly the diagnostic gap that hid the multi-rank issue we hit in ProjectOrion's muon experiment. Strict mode still re-raises.
- **Write-buffer + retry layer** (`_MetricBuffer`) owns reliability for `track_safely`. Every track call now enqueues into a bounded in-process queue (default 100k items, ~20 MB at typical scalar-tuple sizes) and a daemon drainer thread does the actual `run.track()` with exponential-backoff retry on any exception. Eliminates two failure modes that bit ProjectOrion's muon run: gRPC transient failures (Aim's client raises on connection blip → silently drops) and Aim's internal queue overflow under sustained per-batch logging across an SSH-tunneled WAN. The buffer is **default-on, no flag** — silent reliability gaps are exactly the bug class users don't think to opt into fixing, and the overhead is microseconds per submit. Strict mode bypasses (synchronous + raise). `close_run` blocks on buffer drain (30s timeout) so end-of-training doesn't truncate in-flight metrics, then logs a one-line summary (`Aim buffer: N submitted, N drained, N retried, N dropped, N unflushed`) so the operator sees network-side activity even when training succeeds. **What this does NOT fix**: silent drops where `run.track()` returns success without raising — those need an architectural fix (the tracking-API sidecar in astrolabe's roadmap).
- **Dependency drift monitoring** via scheduled GitHub Actions workflow (`dep-watch.yml`). Aim daily, frameworks weekly. New version → auto-test → commit on pass / open `dep-drift` issue on fail.

### Breaking changes (migration)

If you were using `astrolabe-composer-callback==0.1.x`:

```diff
- pip install astrolabe-composer-callback
+ pip install astrolabe-callbacks[composer]
```

```diff
- from astrolabe_composer_callback import AstrolabeLogger
+ from astrolabe_callbacks import AstrolabeComposerLogger
```

```diff
  trainer = Trainer(
      model=...,
-     callbacks=[AstrolabeLogger(repo="aim://localhost:43800")],
+     loggers=[AstrolabeComposerLogger()],
  )
```

Three changes to be aware of:

1. **Package + class names changed.** No back-compat shim — the v0.1.4 release stays on PyPI under the old name as the migration anchor; the new package is `astrolabe-callbacks`.
2. **Attachment point changed for Composer.** `AstrolabeComposerLogger` is now a `LoggerDestination` (was a plain `Callback`). Composer 0.20+ rejects `LoggerDestination` instances passed to `callbacks=` with a clear error; older Composer versions silently dropped every `log_metrics` call. Move to `loggers=`.
3. **Constructor parameter renamed.** `repo=` is now `aim_url=`. `log_interval=` is gone — `LoggerDestination.log_metrics` fires per Composer's own logging cadence (`console_log_interval` in your training YAML), not a separate per-callback throttle.

### Other behavior changes

- **Default constructor `tags={}` no longer disables env fallback.** `AIM_RUN_TAGS` env wins when set; constructor arg is the standalone fallback. Old docstring claimed `tags={}` disabled env fallback — never accurate.
- **`hparams` are now logged via `log_hyperparameters(...)`**, not pulled from `state.model.config` automatically. Composer's Trainer calls `log_hyperparameters` at fit start with whatever you've configured.
- **wall_time correction** (excluding setup + eval pauses) preserved from v0.1.4. Anchored at first batch; eval pauses subtracted.

### Why pass-through?

v0.1.x hardcoded `train/loss` as the only training metric extracted from `state.loss`. If you logged `MaskedLanguagePerplexity`, throughput counters, or any custom metric via `logger.log_metrics(...)`, it never made it to Aim. v0.2.0 fixes this — the user picks what to track in their training YAML, and the callback forwards everything.

---

## Pre-rename history (`astrolabe-composer-callback`)

### v0.1.4

- wall_time now excludes setup time AND eval pauses (training-only elapsed) for apples-to-apples comparison across runs with different eval cadence.

### v0.1.3

- Set Aim run name from Composer's `state.run_name` so the dashboard shows meaningful names instead of `Run: <hash>`.

### v0.1.2

- Env vars (`ASTROLABE_EXPERIMENT_NAME`, `AIM_RUN_TAGS`) now win over constructor args. Astrolabe is the orchestrator; its identity is authoritative.

### v0.1.1

- Don't import `Logger`/`State` from `composer.core`; they moved to `composer.loggers` around 0.21 and importing the old path raised `ImportError`.

### v0.1.0

- Initial release. Composer Callback that logs `train/loss` and `wall_time` per batch, eval metrics from `state.eval_metrics`, and applies tags from `AIM_RUN_TAGS` env var.
