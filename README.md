# astrolabe-callbacks

Framework-agnostic Aim logging for ML training. One install, four frameworks: **MosaicML Composer, PyTorch Lightning, HuggingFace Trainer, raw PyTorch.** Designed to pair with [astrolabe](https://github.com/naston/astrolabe) but works standalone.

> **Pass-through philosophy.** This library streams *every* metric you log to Aim — under the names you chose, with the structure you chose. The only metric we synthesize is `wall_time`. There are no hidden defaults, no metric whitelists, no surprise prefixes. Whatever you `self.log()` (Lightning) or `logger.log_metrics()` (Composer) or pass to `run.log_train()` (raw PyTorch) lands in Aim under that name.

## Install

```bash
pip install astrolabe-callbacks[composer]    # MosaicML Composer
pip install astrolabe-callbacks[lightning]   # PyTorch Lightning
pip install astrolabe-callbacks[hf]          # HuggingFace Trainer
pip install astrolabe-callbacks[all]         # everything
```

Base install (no extras) pulls only `aim` and `loguru` — fine for the raw-PyTorch path.

## Quickstart by framework

<details>
<summary><strong>MosaicML Composer</strong></summary>

```python
from composer import Trainer
from astrolabe_callbacks import AstrolabeComposerLogger

trainer = Trainer(
    model=...,
    train_dataloader=...,
    loggers=[AstrolabeComposerLogger()],   # NOTE: loggers=, not callbacks=
)
trainer.fit()
```

Composer's `Logger` broadcasts every `logger.log_metrics(...)` call to registered `LoggerDestination`s. Attach via `loggers=`, not `callbacks=`, or you'll silently drop user-named metrics. See [docs/frameworks/composer.md](docs/frameworks/composer.md).
</details>

<details>
<summary><strong>PyTorch Lightning</strong></summary>

```python
from lightning.pytorch import Trainer
from astrolabe_callbacks import AstrolabeLightningLogger

trainer = Trainer(
    ...,
    callbacks=[AstrolabeLightningLogger()],
)
trainer.fit(model, train_loader, val_loader)
```

Inside your `LightningModule`, log whatever you want — it all flows through:

```python
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)
    self.log("loss", loss, on_step=True, on_epoch=True)
    self.log("throughput", samples_per_sec, on_step=True)
    self.log("perplexity", torch.exp(loss), on_step=True)
    return loss
```

See [docs/frameworks/lightning.md](docs/frameworks/lightning.md).
</details>

<details>
<summary><strong>HuggingFace Trainer</strong></summary>

```python
from transformers import Trainer, TrainingArguments
from astrolabe_callbacks import AstrolabeHFTrainerCallback

trainer = Trainer(
    model=model,
    args=TrainingArguments(...),
    train_dataset=train_ds,
    eval_dataset=val_ds,
)
trainer.add_callback(AstrolabeHFTrainerCallback())
trainer.train()
```

Works with TRL (`SFTTrainer`, `DPOTrainer`, `RewardTrainer`) unchanged. See [docs/frameworks/huggingface.md](docs/frameworks/huggingface.md).
</details>

<details>
<summary><strong>Raw PyTorch / Accelerate / JAX / custom loops</strong></summary>

```python
from astrolabe_callbacks import Run

with Run() as run:
    for batch_idx, batch in enumerate(dataloader):
        loss = model(batch)
        loss.backward()
        optimizer.step()
        run.log_train(loss=loss.item(), step=batch_idx)

    for val_batch in val_loader:
        val_loss = model(val_batch).item()
        run.log_eval(loss=val_loss, step=batch_idx)
```

`run.log_train(**metrics)` namespaces under `train/`. `run.log_eval(**metrics)` namespaces under `eval/` (will become `val/` in v1.0.0). For arbitrary names, use `run.log(name, value)`. See [docs/frameworks/pytorch.md](docs/frameworks/pytorch.md).
</details>

## Configuration

All four callbacks read the same environment variables. Astrolabe sets them automatically when orchestrating a run; you can also set them yourself for standalone use.

| Variable | Purpose | Default |
|---|---|---|
| `ASTROLABE_EXPERIMENT_NAME` | Aim experiment name | constructor arg or `None` |
| `ASTROLABE_AIM_URL` | Aim tracking URL | `aim://localhost:43800` |
| `AIM_RUN_TAGS` | Tags applied to the run, format `k1=v1,k2=v2` | constructor arg or empty |
| `ASTROLABE_AIM_REPO_PATH` | When set (v2.0+), callback starts a local `aim server` on the compute host writing to this path. NUC-side `astrolabe-sync` sidecar pulls chunks from here every ~3s. Unset = legacy reverse-SSH-tunnel mode. | unset (tunnel mode) |
| `ASTROLABE_CALLBACK_STRICT` | `1` to raise on Aim failures instead of degrading | unset (graceful degrade) |
| `RANK` / `LOCAL_RANK` | Distributed rank (set by `torchrun`) | rank-zero |

**Env wins over constructor args.** Astrolabe is the orchestrator — its identity is authoritative. Constructor args are the standalone fallback.

### Transport modes (v2.0+)

Two transport modes are supported:

- **Tunnel mode (default)**: callback connects directly to the NUC's Aim server via the engine-managed reverse SSH tunnel at `aim://localhost:43800`. Simple, ~40 writes/sec ceiling under realistic emission patterns due to per-call SSH framing overhead.
- **Local-aim mode (opt-in, NUC sets `ASTROLABE_AIM_REPO_PATH`)**: callback starts a local `aim server` subprocess on the compute host. Writes stay on localhost (~1900 writes/sec). The NUC-side `astrolabe-sync` sidecar pulls per-run chunks every ~3s via SSH+rsync.

The mode is selected by the engine, not by the callback. Researchers don't change anything in training code — both modes use the same callback API.

See [astrolabe's `docs/aim-live-sync.md`](https://github.com/naston/astrolabe/blob/main/docs/aim-live-sync.md) for operational details (NUC admin facing).

### Schema-phase finalize (v2.0+, automatic)

In local-aim mode, the dashboard reads from RocksDB SST files on the NUC's central repo. Metric names sit in RocksDB memtable until `Run.close()` forces a flush — so without intervention, a separate-process reader can't enumerate metrics until the run ends.

The callback handles this automatically: at framework boundary hooks (`batch_end`, `eval_end`, etc.), if any new metric names have been observed since the last finalize, the callback drains the metric buffer, closes the Run, and reopens with `force_resume=True`. The close forces memtable → SST flush; the rsync sidecar picks up the new chunks; the dashboard goes live.

Typical Composer run has 1-2 finalize cycles (training metrics at batch 1, eval metrics on first eval). Optimizer handoffs trigger an additional finalize when the new optimizer's metrics first appear. Capped at 10 finalizes per run; pathological churn beyond that degrades to "visible at close."

No researcher action required.

## Failure handling

By default, this library never crashes training. If Aim is unreachable or misconfigured, you get a single `WARNING` log line at startup and every subsequent operation no-ops. Your training continues; you just lose the metrics for that run.

Set `ASTROLABE_CALLBACK_STRICT=1` to flip this — connection failures and per-metric write failures raise instead of swallowing. Useful for CI pipelines where silent degradation hides bugs.

Full contract: [docs/contract.md](docs/contract.md).

## What we name and what we pass through

We **only** synthesize `wall_time` (training-only elapsed time, excluding setup and eval pauses). Everything else is the metric you logged.

| Framework | Names we rewrite | Names we pass through |
|---|---|---|
| Composer | `loss/train/total` → `train/loss`, `metrics/train/<x>` → `train/<x>`, `metrics/eval/<x>` → `val/<x>` | everything else |
| Lightning | `val_<x>` and `val/<x>` → `val/<x>` | everything else |
| HF Trainer | `loss` → `train/loss`, `learning_rate` → `train/lr`, `grad_norm`/`epoch` → `train/<x>`, `eval_<x>` → `val/<x>` | everything else |
| Raw PyTorch | `log_train(**m)` → `train/<m>`, `log_eval(**m)` → `val/<m>`, `log(name, ...)` → `<name>` | (you control namespacing) |

> **v1.0.0**: during-training validation metrics emit under `val/` (was `eval/` in v0.x). The `eval/` namespace is now reserved for **post-training benchmark suites** logged via `astrolabe.eval_results.log_eval_table(...)` on dedicated eval Aim runs. See [astrolabe's `docs/eval.md`](https://github.com/naston/astrolabe/blob/main/docs/eval.md) for the post-training side.

Renames are cosmetic, applied to framework-emitted names you didn't choose. **User-named metrics are never rewritten.** If you log a metric called `MaskedLanguagePerplexity`, it lands in Aim as `MaskedLanguagePerplexity`, not buried under any prefix.

## Distributed training

All four callbacks gate Aim writes on rank-zero (detected via `torch.distributed` if initialized, `RANK`/`LOCAL_RANK` env otherwise). N processes produce one Aim run, not N. No configuration required for `torchrun`, Composer's launcher, or Lightning's `strategy="ddp"`.

## Versioning + back-compat

This package replaces `astrolabe-composer-callback`. If you were using it:

```diff
- from astrolabe_composer_callback import AstrolabeLogger
+ from astrolabe_callbacks import AstrolabeComposerLogger
```

```diff
- callbacks=[AstrolabeLogger()]
+ loggers=[AstrolabeComposerLogger()]
```

The `loggers=` placement is the bigger change — Composer rejects `LoggerDestination` instances passed to `callbacks=` since 0.20+. See [CHANGELOG.md](CHANGELOG.md).

## License

MIT.
