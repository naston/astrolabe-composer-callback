# HuggingFace Trainer

Cookbook for `AstrolabeHFTrainerCallback` — `transformers.Trainer` and TRL.

## Minimal example

```python
from transformers import Trainer, TrainingArguments
from astrolabe_callbacks import AstrolabeHFTrainerCallback

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./checkpoints/my-run",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=500,
    ),
    train_dataset=train_ds,
    eval_dataset=val_ds,
)
trainer.add_callback(AstrolabeHFTrainerCallback())
trainer.train()
```

## Full example

```python
callback = AstrolabeHFTrainerCallback(
    aim_url="aim://my-aim-server.example.com:43800",
    experiment_name="llama-finetune",
    tags={"task": "instruction-tuning", "base_model": "llama3-8b"},
    run_name="llama3-8b-sft-v2",
)
trainer.add_callback(callback)
trainer.train()
```

## Works with TRL out of the box

TRL's `SFTTrainer`, `DPOTrainer`, `RewardTrainer`, and `PPOTrainer` all inherit from `transformers.Trainer`. The callback works unchanged:

```python
from trl import SFTTrainer
from astrolabe_callbacks import AstrolabeHFTrainerCallback

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    callbacks=[AstrolabeHFTrainerCallback()],
)
trainer.train()
```

## What gets logged to Aim

| Source | Aim metric name |
|---|---|
| HF Trainer's smoothed train loss | `train/loss` (renamed from `loss`) |
| Learning rate | `train/lr` (renamed from `learning_rate`) |
| Gradient norm | `train/grad_norm` (renamed from `grad_norm`) |
| Epoch counter | `train/epoch` (renamed from `epoch`) |
| Eval loss | `val/loss` (renamed from `eval_loss`) |
| Other eval metrics | `val/<x>` (renamed from `eval_<x>`) |
| Custom metrics from `trainer.log({...})` or `compute_metrics` | passed through unchanged |
| Synthesized | `wall_time` (training-only elapsed seconds) |

## Logging custom metrics

HF's `Trainer.log({...})` and `compute_metrics` callbacks both flow through HF's `on_log` and `on_evaluate` hooks. Anything you put in those dicts shows up in Aim.

```python
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "perplexity": math.exp(eval_pred.loss),
    }

trainer = Trainer(
    ...,
    compute_metrics=compute_metrics,
)
```

In Aim: `val/accuracy`, `val/f1_macro`, `val/perplexity`. The `eval_` prefix HF wraps these in is stripped and re-namespaced under `val/` by the callback (v1.0.0+; pre-v1.0.0 used `eval/`).

## Run name resolution

If `run_name=` isn't passed at construction:

1. `args.run_name` (the `TrainingArguments.run_name` field, if you set it)
2. `Path(args.output_dir).name` (the basename of your output directory)

So `output_dir="./checkpoints/llama3-sft-v2"` produces a run named `llama3-sft-v2` in Aim by default. Set `args.run_name=` for an explicit name.

## Common patterns

### Distributed training

HF Trainer with `accelerate launch` or `torchrun` initializes `torch.distributed` automatically. The callback gates writes on rank-zero. No configuration required.

### DeepSpeed

Pass `deepspeed=<config.json>` in `TrainingArguments`. HF wraps DeepSpeed; the callback sees the standard `on_log`/`on_evaluate` hooks. Works unchanged.

### Sharing the callback across Trainer instances

Construct once, attach to multiple Trainers — but the callback holds an Aim run reference, so a single callback instance can only track one training run at a time. For sequential trainings, construct a new callback for each:

```python
for fold in range(5):
    trainer = Trainer(...)
    trainer.add_callback(AstrolabeHFTrainerCallback(
        run_name=f"cv-fold-{fold}",
    ))
    trainer.train()
```

## Gotchas

### Wall-time precision is log-interval-grained

HF Trainer doesn't expose a clean per-batch hook with metric data — only `on_step_end` (no metrics) and `on_log` (every `logging_steps` batches). The callback anchors `wall_time` at the first `on_step_end` and writes it on every `on_log`. If eval falls between two `on_log` events, the eval time gets included in the next `wall_time` reading.

For most users this is fine. For fine-grained timing analysis, switch to `AstrolabeRun` and write the loop yourself.

### `logging_steps` controls metric cadence

Aim's time-series for `train/loss` reflects HF's `logging_steps` interval. If `logging_steps=50`, you get one data point every 50 steps, smoothed by HF's running-mean. Set `logging_steps=1` for per-step granularity (at some logging cost).

### `eval_strategy="no"` skips on_evaluate

If you set `eval_strategy="no"` (no automatic eval during training), `on_evaluate` never fires and you'll have no `eval/*` metrics in Aim. Set `eval_strategy="steps"` or `"epoch"` to get eval metrics.

### No `on_exception` hook in TrainerCallback

HF's `TrainerCallback` API doesn't expose an exception hook. If training raises, the run closes via Python's normal cleanup — but `astrolabe.status` ends up `completed` because `on_train_end` fired during cleanup, not the failure path. The Aim run will still have all metrics streamed up to the crash. For accurate failure marking, use the `ASTROLABE_CALLBACK_STRICT=1` env var in CI environments where you want training failures to be loud.

### Migrating from `astrolabe-composer-callback`

If you were using a Composer callback, this is a different package and a different framework. Just install `astrolabe-callbacks[hf]` fresh.
