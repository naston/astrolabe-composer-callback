# astrolabe-composer-callback

A MosaicML Composer callback that streams training metrics to an Aim tracking server, with first-class support for run tagging via env vars. Designed to pair with [astrolabe](https://github.com/naston/astrolabe) but usable standalone.

## Install

```bash
pip install astrolabe-composer-callback
```

Requires Composer (`mosaicml>=0.20`) and Aim (`aim>=3.18`).

## Use

### With astrolabe

When astrolabe orchestrates a training run, it writes `ASTROLABE_EXPERIMENT_NAME` and `AIM_RUN_TAGS` into the GPU instance's environment. The callback picks them up automatically:

```python
from astrolabe_composer_callback import AstrolabeLogger
from composer import Trainer

trainer = Trainer(
    model=...,
    train_dataloader=...,
    callbacks=[AstrolabeLogger(repo="aim://localhost:43800")],
)
trainer.fit()
```

The resulting Aim run lands tagged with `astrolabe.version=v3`, `astrolabe.submit_id=<uuid>`, `astrolabe.experiment=<name>` — astrolabe's dashboard reads those tags to group runs by submit version.

### Standalone

If you're using Composer + Aim without astrolabe, set `tags` explicitly:

```python
AstrolabeLogger(
    repo="aim://localhost:43800",
    experiment_name="my-exp",
    tags={"thesis": "scale-laws", "model": "BERT"},
)
```

Or set `AIM_RUN_TAGS` in your shell:

```bash
AIM_RUN_TAGS="thesis=scale-laws,model=BERT" python train.py
```

## What gets logged

- `train/loss` per batch (read from `state.loss`)
- `wall_time` per batch (seconds since the run started — useful as an x-axis for time-based comparison)
- All eval metrics from `state.eval_metrics` on `eval_end` (with `eval/...` namespace flattening)
- All `tags` from the constructor (or `AIM_RUN_TAGS` env) — applied as Aim run keys
- `hparams` from `state.model.config` if available

## Why a separate package?

Astrolabe's dashboard groups Aim runs by `astrolabe.version` tag. Without something tagging each Aim run on init, version-grouping has to be derived from creation_time clustering — fragile. This callback closes that gap.

The package is named for astrolabe because the conventions (`astrolabe.version`, `astrolabe.submit_id`, `astrolabe.experiment`) are astrolabe-flavored. But the env-tag interface (`AIM_RUN_TAGS`) is generic — a non-astrolabe user can adopt the conventions or set their own tags without code changes.

## API

### `AstrolabeLogger`

```python
AstrolabeLogger(
    repo: str = "aim://localhost:43800",
    experiment_name: str = "",
    log_interval: int = 1,
    tags: dict[str, str] | None = None,
)
```

- **`repo`** — Aim tracking URI.
- **`experiment_name`** — Aim experiment name. Falls back to `ASTROLABE_EXPERIMENT_NAME` env var when empty.
- **`log_interval`** — log every N batches.
- **`tags`** — dict of tags applied to the Aim run on init. Falls back to parsing `AIM_RUN_TAGS` env var when `None`. Pass `tags={}` to explicitly disable env fallback.

### `parse_aim_run_tags(raw: str | None) -> dict[str, str]`

Parses the `key1=val1,key2=val2` format. Exposed for testing or for callers who want to apply the same parsing elsewhere.

## License

MIT.
