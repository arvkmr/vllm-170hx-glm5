# aiserver live setup (snapshot 2026-10-08)

An exact copy of the GLM-5.3 serving setup running on `aiserver` now. The
generic recipe is on `main`. Nothing here was edited: every file is a
byte-for-byte copy, except this README and `.gitignore`.

## What is running

```bash
cd ~/vllm_install/vllm_next
SERVE_SCRIPT=serve-uncensored.sh ./start.sh
```

| | |
|---|---|
| Checkpoint | `~/models/GLM-5.3-UNCENSORED-Int4-Int8Mix-AWQ-g64` (via `serve-uncensored.sh`) |
| Drafter | `~/models/GLM-5.3-DFlash2`, DFlash, k=7 |
| Parallelism | PP=8, TP=1, `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7,8` (GPU 6 left free), `CUDA_DEVICE_ORDER=PCI_BUS_ID` |
| Partition | `11,10,10,10,10,10,9,8` (needs the top-k PP relay) |
| Profile | `agent`: `--max-model-len 524288`, `--max-num-seqs 4`, `--max-num-batched-tokens 512`, FULL_AND_PIECEWISE graphs |
| Memory | `--gpu-memory-utilization 0.96`, KV pool 1,025,573 tokens |
| Bind | `127.0.0.1:8002`. `0.0.0.0:8000` is `~/vision_sidecar/proxy.py` (text pass-through, not in this repo) |

## Layout

The repo root stands in for `~/vllm_install`. Only the two files there that the
recipe uses are included: `install.sh` and `preflight.py` read
`../glm52_mla_fp8.py`, and `apply_engine_patch.py` reads
`glm52_topk_canon.py` next to it.

| Path here | On aiserver |
|---|---|
| `glm52_mla_fp8.py`, `glm52_topk_canon.py` | `~/vllm_install/` |
| `vllm_next/` | `~/vllm_install/vllm_next/` (without `logs/`, `__pycache__/` and `*.bak-*`) |
| `vllm_next/rollback-pp9-20261005/` | PP=9 rollback kit, as on the host |
| `engine/vllm-src.diff` | `git diff HEAD` of `~/vllm_glm53_dflash2/vllm-src` |
| `engine/untracked/` | untracked files in that tree (installed helpers, patch stamp) |
| `engine/HEAD` | the engine's pinned commit |
| `host/crontab.txt` | `crontab -l` for `arveen` |

## The engine differs from what `install.sh` produces

The top-k PP relay (`local-cmp170hx-dsv32-topk-pp-relay`) that PP=8 needs was
edited into the engine tree by hand. The `apply_engine_patch.py` in this
snapshot does not apply it. To rebuild this exact engine:

```bash
cd ~/vllm_glm53_dflash2/vllm-src
git checkout "$(cat <repo>/engine/HEAD)" && git checkout -- .
git apply <repo>/engine/vllm-src.diff
cp -a <repo>/engine/untracked/. .
```

`main`'s `apply_engine_patch.py` includes the relay and produces the same
`model.py`.

## Not in this repo

- Model weights and the drafter.
- `~/vision_sidecar` (`:8000` proxy) and the gateway, tailcat and fan units
  (`tools/` in `ubuntu-setup`).
- `~/gpu-power-log` (started by the third crontab line).
- `~/.config/server-alert/ntfy.env`, which holds the secret ntfy topic.
- Host state outside these files: the core clock profile, the 160 W cap,
  and the driver.
