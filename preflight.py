#!/usr/bin/env python3
"""Fail-closed compatibility and host preflight for the vNext stack."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ENGINE_COMMIT = "378c37b0098a41a5cd25b3bf8b56d158e33a6cbf"
DRAFT_REVISION = "425aa615ce320caac34400208b30808c8f14f76c"
TARGET_SHAPE = {
    "hidden_size": 6144,
    "num_hidden_layers": 78,
    "vocab_size": 154880,
}
DRAFT_SHAPE = {
    "hidden_size": 6144,
    "num_hidden_layers": 6,
    "num_target_layers": 78,
    "vocab_size": 154880,
}
DRAFT_TAPS = [5, 19, 33, 47, 61, 75]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"preflight: {message}")


def load_config(path: Path, label: str) -> dict:
    config = path / "config.json"
    require(config.is_file(), f"missing {label} config: {config}")
    try:
        return json.loads(config.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"preflight: cannot read {label} config: {exc}") from exc


def validate_target(config: dict) -> None:
    require(
        "GlmMoeDsaForCausalLM" in config.get("architectures", []),
        "target is not the 78-layer GlmMoeDsaForCausalLM checkpoint",
    )
    for key, expected in TARGET_SHAPE.items():
        require(config.get(key) == expected, f"target {key} must be {expected}")
    require(config.get("index_topk") == 2048, "target index_topk must be 2048")
    require(config.get("moe_router_dtype") == "float32", "target must request fp32 router logits")


def validate_draft(config: dict) -> None:
    require(config.get("architectures") == ["DFlash2DraftModel"], "wrong drafter architecture")
    for key, expected in DRAFT_SHAPE.items():
        require(config.get(key) == expected, f"drafter {key} must be {expected}")
    dflash = config.get("dflash_config", {})
    require(dflash.get("block_size") == 8, "drafter block_size must be 8")
    require(dflash.get("target_layer_ids") == DRAFT_TAPS, "unexpected target-layer taps")
    require(dflash.get("selector_top_k") == 16, "drafter selector_top_k must be 16")


TOPK_RELAY_MARKER = "local-cmp170hx-dsv32-topk-pp-relay"


def engine_has_topk_relay(source: Path) -> bool:
    model = source / "vllm/models/deepseek_v32/nvidia/model.py"
    return model.is_file() and TOPK_RELAY_MARKER in model.read_text()


def validate_partition(config: dict, text: str, topk_relay: bool = False) -> list[int]:
    try:
        partition = [int(value) for value in text.split(",")]
    except ValueError as exc:
        raise SystemExit("preflight: PP partition must be comma-separated integers") from exc
    stages = int(os.environ.get("PP_SIZE", "10"))
    require(len(partition) == stages, f"PP partition must contain exactly {stages} stages")
    require(all(value > 0 for value in partition), "PP partition entries must be positive")
    require(sum(partition) == TARGET_SHAPE["num_hidden_layers"], "PP partition must sum to 78")

    # A skip-topk layer consumes the most recent index-producing layer's
    # selections. Since that buffer is rank-local in this fork, each stage must
    # start with an index producer. The chosen 10+8...+4 split has this property
    # for freq=4/offset=3 and therefore needs no legacy top-k relay patch.
    freq = int(config.get("index_topk_freq", 1))
    offset = int(config.get("index_skip_topk_offset", 2))
    starts: list[int] = []
    cursor = 0
    for width in partition:
        starts.append(cursor)
        cursor += width
    bad = [layer for layer in starts if max(layer - offset + 1, 0) % freq != 0]
    # With the top-k PP relay (local-cmp170hx-dsv32-topk-pp-relay) the
    # upstream stage ships its selections across the boundary, so a stage may
    # begin on a skip-topk layer (needed for PP=9: 78 layers cannot be split
    # into 9 producer-aligned stages without a ~60 GiB stage).
    require(
        not bad or topk_relay,
        f"PP stages {bad} begin on shared/skip-topk layers and the engine "
        "lacks the top-k PP relay",
    )
    return partition


def validate_engine(source: Path) -> None:
    require((source / ".git").exists(), f"engine source is not a git checkout: {source}")
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    require(head == ENGINE_COMMIT, f"engine HEAD is {head}, expected {ENGINE_COMMIT}")
    marker = source / ".glm53-dflash2-local-patch"
    require(marker.is_file(), "packed-fp8 engine patch has not been applied")
    backend = source / "vllm/v1/attention/backends/mla/triton_mla_sparse.py"
    require("local-cmp170hx-fp8-ds-mla" in backend.read_text(), "packed-fp8 marker missing")
    dsv32 = source / "vllm/models/deepseek_v32/nvidia/model.py"
    require(
        "local-cmp170hx-dsv32-aux-pp" in dsv32.read_text(),
        "DSA aux-over-PP (DFlash relay) patch missing; rerun install.sh",
    )
    kernels = source / "vllm/models/deepseek_v32/common/kernels.py"
    require(
        "local-cmp170hx-dsv32-sm80-fp8" in kernels.read_text(),
        "DSA sm_80 software-fp8 kernel patch missing; rerun install.sh",
    )
    indexer = source / "vllm/model_executor/layers/sparse_attn_indexer.py"
    require(
        "local-cmp170hx-dsv32-sm80-indexer" in indexer.read_text(),
        "DSA sm_80 indexer (Triton logits) patch missing; rerun install.sh",
    )
    topk = source / "vllm/model_executor/layers/indexer_topk.py"
    require(
        "local-cmp170hx-canon-topk" in topk.read_text()
        and (source / "vllm/_glm52_topk_canon.py").is_file(),
        "canonical top-k (v0.26 kernel) patch missing; rerun install.sh",
    )
    require(
        "local-cmp170hx-dsa-dflash-kv"
        in (source / "vllm/v1/core/kv_cache_utils.py").read_text(),
        "DSA DFlash window-bounded drafter KV patch missing; rerun install.sh",
    )
    require(
        "local-cmp170hx-route-v2-dsa"
        in (source / "vllm/ampere_decode/__init__.py").read_text(),
        "fused MoE router gate for the 256 x 6144 router missing; rerun install.sh",
    )
    require(
        "local-cmp170hx-mla-head-bmm"
        in (source / "vllm/model_executor/layers/attention/mla_attention.py").read_text()
        and "local-cmp170hx-mla-head-bmm"
        in (source / "vllm/models/deepseek_v32/attention.py").read_text()
        and (source / "vllm/_glm52_mla_bmm.py").is_file(),
        "MLA per-head decode bmm patch missing; rerun install.sh",
    )
    installed_helper = source / "vllm/_glm52_mla_fp8.py"
    project_helper = Path(__file__).resolve().parent / "glm52_mla_fp8.py"
    require(installed_helper.is_file(), "packed-fp8 reader is missing from the engine")
    require(
        hashlib.sha256(installed_helper.read_bytes()).digest()
        == hashlib.sha256(project_helper.read_bytes()).digest(),
        "installed packed-fp8 reader does not match this migration branch",
    )
    registry = (source / "vllm/model_executor/models/registry.py").read_text()
    require(
        '"GlmMoeDsaForCausalLM": ("vllm.models.deepseek_v32"' in registry,
        "fork does not route the target through its Ampere DeepSeek-v3.2 path",
    )
    require('"DFlash2DraftModel":' in registry, "fork does not register DFlash2")
    runner = (source / "vllm/v1/worker/gpu/model_runner.py").read_text()
    require(
        "configure_aux_hidden_state_relay" in runner,
        "fork lacks the DFlash auxiliary-state PP relay",
    )


def validate_gpu_host(require_idle: bool) -> None:
    try:
        query = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,pci.bus_id,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"preflight: nvidia-smi failed: {exc}") from exc
    gpus = {}
    for row in query.splitlines():
        if not row.strip():
            continue
        fields = [item.strip() for item in row.split(",")]
        require(len(fields) == 4, f"unexpected nvidia-smi row: {row}")
        gpus[fields[0]] = fields
    expected = int(os.environ.get("PP_SIZE", "10")) * int(os.environ.get("TP_SIZE", "1"))
    # CUDA_VISIBLE_DEVICES may select a subset of the host's cards (PP=8 leaves
    # GPU 6 out). Its indices only match nvidia-smi's under PCI bus ordering.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        selected = [item.strip() for item in visible.split(",") if item.strip()]
        require(
            os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID" or len(selected) == len(gpus),
            "a GPU subset in CUDA_VISIBLE_DEVICES needs CUDA_DEVICE_ORDER=PCI_BUS_ID",
        )
    else:
        selected = list(gpus)
    require(len(selected) == expected, f"expected {expected} visible GPUs, found {len(selected)}")
    require(len(set(selected)) == len(selected), f"duplicate GPU in CUDA_VISIBLE_DEVICES={visible}")
    for index in selected:
        require(index in gpus, f"GPU {index} not present (nvidia-smi lists {','.join(gpus)})")
        fields = gpus[index]
        require("CMP 170HX" in fields[2], f"unexpected GPU: {fields[2]}")
        require(int(fields[3]) >= 61440, f"GPU {fields[0]} exposes under 60 GiB")

    if require_idle:
        buses = {gpus[index][1].lower() for index in selected}
        processes = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_bus_id,pid,process_name",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        busy = [line for line in processes.splitlines()
                if line.split(",")[0].strip().lower() in buses]
        require(not busy, "GPUs are busy; refusing to launch:\n" + "\n".join(busy))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument(
        "--partition", default=os.environ.get("VLLM_PP_LAYER_PARTITION", "10,8,8,8,8,8,8,8,8,4")
    )
    parser.add_argument("--host", action="store_true", help="also validate the PP x TP GPU host")
    parser.add_argument("--require-idle", action="store_true")
    args = parser.parse_args()

    target = load_config(args.target.resolve(), "target")
    draft_path = args.draft.resolve()
    draft = load_config(draft_path, "drafter")
    validate_target(target)
    validate_draft(draft)
    revision_file = draft_path / ".pinned-revision"
    require(revision_file.is_file(), "drafter is missing its .pinned-revision stamp")
    require(
        revision_file.read_text().strip() == DRAFT_REVISION,
        f"drafter revision must be {DRAFT_REVISION}",
    )
    engine = args.engine.resolve()
    partition = validate_partition(target, args.partition, engine_has_topk_relay(engine))
    validate_engine(engine)
    if args.host or args.require_idle:
        validate_gpu_host(args.require_idle)
    print(
        "preflight: ok — engine pin, target/drafter architecture, "
        f"and PP partition {','.join(map(str, partition))} agree"
    )


if __name__ == "__main__":
    main()
