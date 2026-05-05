# Raw PyTorch / JAX / Custom Loops

Cookbook for `Run` (a.k.a. `AstrolabeRun`) — the context manager for code that doesn't have a callback system.

## When to use this

Use this path when:

- You're writing a hand-rolled PyTorch training loop.
- You're using HuggingFace `Accelerate` and writing the loop yourself (most Accelerate users do).
- You're using JAX/Flax and writing a Python-side loop around `jax.jit`'d functions.
- You have a custom training framework with no exposed hooks.

If you're using Composer, Lightning, or HF Trainer, use the matching `Astrolabe<Framework>Logger` instead — those handle lifecycle hooks for you.

## Minimal example

```python
from astrolabe_callbacks import Run

with Run() as run:
    for batch_idx, batch in enumerate(dataloader):
        loss = model(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        run.log_train(loss=loss.item(), step=batch_idx)
```

`Run` reads astrolabe env vars on `__enter__` and opens an Aim run. On exit (clean or exception), it closes the run with the appropriate `astrolabe.status`.

## Full example

```python
from astrolabe_callbacks import Run

with Run(
    aim_url="aim://my-aim-server.example.com:43800",
    experiment_name="custom-rl-loop",
    tags={"thesis": "rl-from-scratch"},
    run_name="ppo-cartpole-v3",
) as run:
    # Training
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            step = epoch * len(train_loader) + batch_idx
            loss = model(batch)
            grad_norm = clip_grad_norm_(model.parameters(), 1.0)

            run.log_train(
                loss=loss.item(),
                grad_norm=grad_norm.item(),
                lr=optimizer.param_groups[0]["lr"],
                step=step,
            )
            loss.backward()
            optimizer.step()

        # Eval at end of each epoch
        run.pause_eval()
        with torch.no_grad():
            val_loss, val_acc = evaluate(model, val_loader)
        run.log_eval(loss=val_loss, accuracy=val_acc, step=step)
        run.resume()
```

## API

```python
with Run(
    aim_url=None,             # ASTROLABE_AIM_URL wins; default aim://localhost:43800
    experiment_name=None,     # ASTROLABE_EXPERIMENT_NAME wins
    tags=None,                # AIM_RUN_TAGS env wins
    run_name=None,            # human-readable name in dashboard
) as run:
    run.log_train(**metrics, step=None)        # → train/<name> for each kwarg + wall_time
    run.log_eval(**metrics, step=None)         # → eval/<name> for each kwarg
    run.log(name, value, step=None, context=None)  # arbitrary; no namespace
    run.set_tag(name, value)                   # late-binding tag
    run.pause_eval()                           # exclude eval from wall_time
    run.resume()                               # resume wall_time accounting

    run.is_active                              # bool — false if rank-nonzero or aim missing
```

## What gets logged to Aim

| Call | Aim metric name |
|---|---|
| `run.log_train(loss=x)` | `train/loss` |
| `run.log_train(my_metric=x, step=5)` | `train/my_metric @ step 5` |
| `run.log_train(...)` | always also writes `wall_time` |
| `run.log_eval(loss=x)` | `eval/loss` |
| `run.log_eval(custom=x)` | `eval/custom` |
| `run.log("my/raw/name", x)` | `my/raw/name` (no namespace prefix) |
| `run.set_tag("my.tag", "value")` | tag on the run, not a time-series metric |

## Common patterns

### JAX / Flax

JAX's training loop is Python; the callback works the same way as raw PyTorch. Just remember to `jax.device_get` tensors before logging if they're on the device:

```python
import jax
from astrolabe_callbacks import Run

with Run(experiment_name="flax-mnist") as run:
    for step in range(num_steps):
        state, loss = train_step(state, next(batches))
        # device_get pulls the JAX scalar to host before logging.
        run.log_train(loss=jax.device_get(loss).item(), step=step)
```

### HuggingFace Accelerate (manual loop)

Most Accelerate users write the loop themselves. `Run` works unchanged:

```python
from accelerate import Accelerator
from astrolabe_callbacks import Run

accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

with Run(experiment_name="my-finetune") as run:
    for batch_idx, batch in enumerate(dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        # Only rank-zero writes; Accelerate's distributed setup is detected.
        run.log_train(loss=loss.item(), step=batch_idx)
```

### Tracking step manually

`step` is optional. If you don't pass it, Aim auto-increments its internal step counter. For most cases, pass `step=batch_idx` (or whatever your loop's natural counter is) so Aim's time-series matches your training step.

```python
# Auto-increment (Aim manages step):
run.log_train(loss=loss.item())

# Explicit step (recommended):
run.log_train(loss=loss.item(), step=global_step)
```

### Distributed training

`torchrun --nproc-per-node=8 train.py` initializes `torch.distributed`; `Run` detects rank-zero automatically. Non-zero ranks enter the context manager but `run.is_active` is `False` and every method no-ops:

```python
with Run() as run:
    if run.is_active:
        print("This is rank 0 — Aim is logging")
    else:
        print("Non-zero rank — calls no-op")
    # Either way, calling log_train works (no-ops on non-rank-zero).
    run.log_train(loss=loss.item(), step=step)
```

You don't need to gate calls yourself — the rank check happens inside the methods.

## Gotchas

### Don't reuse a `Run` instance after exit

```python
# WRONG
run_instance = Run()
with run_instance as run:
    run.log_train(loss=0.5, step=1)
with run_instance as run:    # Run is already closed; methods no-op silently.
    run.log_train(loss=0.4, step=2)

# RIGHT
with Run() as run:
    run.log_train(loss=0.5, step=1)
with Run() as run:           # Fresh instance.
    run.log_train(loss=0.4, step=2)
```

### `pause_eval` + `resume` must be paired

If you call `pause_eval` without a matching `resume`, the wall-time clock stays paused and subsequent `wall_time` writes will be wrong. Use `try`/`finally` if your eval might raise:

```python
run.pause_eval()
try:
    val_loss = evaluate(model, val_loader)
finally:
    run.resume()
```

### `log()` doesn't synthesize wall_time

`log_train(...)` always writes `wall_time` alongside your metrics. `log(name, value)` doesn't — it's the escape hatch for arbitrary names, intentionally minimal. If you want `wall_time` and a custom-named metric, call `log_train()` first then `log()`:

```python
run.log_train(loss=loss.item(), step=step)        # writes train/loss + wall_time
run.log("custom/special_name", x, step=step)      # writes only custom/special_name
```

### Migrating from `astrolabe-composer-callback`

If you were using a Composer callback, this is a different path entirely. The Composer-specific code lives in `AstrolabeComposerLogger`; `Run` is for code without a callback system.
