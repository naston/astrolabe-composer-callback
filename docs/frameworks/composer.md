# Composer

Cookbook for `AstrolabeComposerLogger` — MosaicML Composer integration.

## Minimal example

```python
from composer import Trainer
from astrolabe_callbacks import AstrolabeComposerLogger

trainer = Trainer(
    model=...,
    train_dataloader=...,
    loggers=[AstrolabeComposerLogger()],
)
trainer.fit()
```

That's it. The callback reads astrolabe env vars (`ASTROLABE_EXPERIMENT_NAME`, `AIM_RUN_TAGS`, `ASTROLABE_AIM_URL`), opens an Aim run, and forwards every `logger.log_metrics(...)` call from Composer's training loop to Aim under the run's name.

> **Critical:** attach via `loggers=`, not `callbacks=`. `AstrolabeComposerLogger` is a `LoggerDestination`, and Composer's `Logger` only broadcasts user metrics to destinations registered there. Composer 0.20+ rejects `LoggerDestination` instances passed to `callbacks=` with a clear error; older versions silently drop every `log_metrics` call. The class still implements all `Callback` lifecycle hooks (it inherits from `Callback` via `LoggerDestination`), so you get `wall_time` tracking, eval-pause correction, and clean close — but only if you place it in `loggers=`.

## Full example with explicit config

```python
from composer import Trainer
from astrolabe_callbacks import AstrolabeComposerLogger

logger_callback = AstrolabeComposerLogger(
    aim_url="aim://my-aim-server.example.com:43800",
    experiment_name="bert-pretrain",
    tags={
        "thesis": "scale-laws",
        "model_size": "base",
        "dataset": "c4",
    },
)

trainer = Trainer(
    model=model,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    optimizers=optimizer,
    schedulers=scheduler,
    max_duration="1ep",
    loggers=[logger_callback],
)
trainer.fit()
```

When `astrolabe submit` orchestrates the run, env vars override the constructor args:

```bash
ASTROLABE_EXPERIMENT_NAME=bert-pretrain \
ASTROLABE_AIM_URL=aim://localhost:43800 \
AIM_RUN_TAGS="astrolabe.version=v3,astrolabe.submit_id=abc-123" \
python train.py
```

## What gets logged to Aim

| Source | Aim metric name |
|---|---|
| Composer's automatic train loss | `train/loss` (renamed from `loss/train/total`) |
| Composer's per-batch train metrics (e.g. `metrics/train/Accuracy`) | `train/Accuracy` |
| Composer's eval metrics (e.g. `metrics/eval/MaskedLanguagePerplexity`) | `eval/MaskedLanguagePerplexity` |
| `logger.log_metrics({"my_thing": x})` from your training code | `my_thing` (passed through unchanged) |
| `lr-DecoupledAdamW` from `lr_monitor` callback | `lr-DecoupledAdamW` (passed through unchanged) |
| `throughput/samples_per_sec` from `speed_monitor` callback | `throughput/samples_per_sec` (passed through unchanged) |
| Synthesized | `wall_time` (training-only elapsed seconds, eval-paused) |

## Logging custom metrics

Use Composer's standard `Logger.log_metrics` API from your training step or any callback:

```python
class MyCustomCallback(Callback):
    def batch_end(self, state, logger):
        custom_value = compute_something(state)
        logger.log_metrics({"my_custom/metric": custom_value})
```

That metric flows through `AstrolabeComposerLogger.log_metrics` and lands in Aim as `my_custom/metric`.

## Hyperparameters

`AstrolabeComposerLogger.log_hyperparameters(...)` writes to Aim's `hparams` field. Composer calls it automatically at the start of training with whatever you've configured. To log additional hparams from your own code:

```python
class MyHparamsLogger(Callback):
    def fit_start(self, state, logger):
        logger.log_hyperparameters({
            "custom_setting": 42,
            "ablation_id": "exp-007",
        })
```

## Common patterns

### Distributed training

No special setup. Composer's launcher (`composer -n 8 train.py` or `torchrun`) initializes `torch.distributed`; the callback detects rank-zero automatically and only writes from rank 0. The other 7 ranks open the callback but their writes are no-ops.

### Multiple eval suites

If you have multiple eval suites (e.g. `glue_mnli`, `glue_sst2`), Composer emits metrics under `metrics/<suite_name>/<metric>`. The default suite name `"eval"` collapses to `eval/`; named suites keep their name.

```python
# Named suite "glue_mnli" with metric "Accuracy"
# → Composer emits: metrics/glue_mnli/Accuracy
# → AstrolabeComposerLogger writes to Aim: glue_mnli/Accuracy

# Default suite "eval" with metric "MaskedLanguagePerplexity"
# → Composer emits: metrics/eval/MaskedLanguagePerplexity
# → AstrolabeComposerLogger writes to Aim: eval/MaskedLanguagePerplexity
```

### Combining with other loggers

`loggers=[AstrolabeComposerLogger(), TensorboardLogger(), WandBLogger()]` — they coexist. Each is an independent destination. Aim gets the astrolabe-tagged copy; TB and W&B get untagged copies.

## Gotchas

### Don't pass `AstrolabeComposerLogger` to `callbacks=`

```python
# WRONG — Composer 0.20+ raises; older silently drops user metrics
trainer = Trainer(callbacks=[AstrolabeComposerLogger()])

# RIGHT
trainer = Trainer(loggers=[AstrolabeComposerLogger()])
```

### `mlm_probability=0.3` etc. are model hparams, not metrics

If you want them in the run's metadata, log via `logger.log_hyperparameters`, not `logger.log_metrics`. Hparams land alongside the run; metrics land in time-series.

### Composer's automatic metrics are aggregated

The `train/loss` metric you'll see in Aim is Composer's per-batch loss with whatever smoothing/aggregation Composer applies (typically EMA over `console_log_interval`). If you want a raw per-step loss, log it explicitly: `logger.log_metrics({"train/raw_loss": state.loss.item()})` in your callback.

### Migrating from `astrolabe-composer-callback==0.1.x`

Three changes:

1. Package name: `pip uninstall astrolabe-composer-callback && pip install astrolabe-callbacks[composer]`
2. Import: `from astrolabe_callbacks import AstrolabeComposerLogger` (was `from astrolabe_composer_callback import AstrolabeLogger`)
3. Placement: `Trainer(loggers=[...])` (was `Trainer(callbacks=[...])`)

The behavior also changed — v0.1.x only logged `train/loss` and a few framework metrics. v0.2.0 passes through *every* `logger.log_metrics` call. If you were writing custom metrics via a separate callback to a separate Aim run, you can delete that and let `AstrolabeComposerLogger` capture them.
