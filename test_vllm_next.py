#!/usr/bin/env python3
"""Static contract tests for the isolated vNext migration."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from preflight import (
    DRAFT_TAPS,
    validate_draft,
    validate_partition,
    validate_target,
)

ROOT = Path(__file__).parent
NEXT = ROOT


def target_config() -> dict:
    return {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "hidden_size": 6144,
        "num_hidden_layers": 78,
        "vocab_size": 154880,
        "index_topk": 2048,
        "index_topk_freq": 4,
        "index_skip_topk_offset": 3,
        "moe_router_dtype": "float32",
    }


def draft_config() -> dict:
    return {
        "architectures": ["DFlash2DraftModel"],
        "hidden_size": 6144,
        "num_hidden_layers": 6,
        "num_target_layers": 78,
        "vocab_size": 154880,
        "dflash_config": {
            "block_size": 8,
            "target_layer_ids": DRAFT_TAPS,
            "selector_top_k": 16,
        },
    }


class VllmNextContracts(unittest.TestCase):
    def test_exact_target_and_drafter_contract(self) -> None:
        validate_target(target_config())
        validate_draft(draft_config())

    def test_partition_is_aligned_to_index_producers(self) -> None:
        part = validate_partition(target_config(), "10,8,8,8,8,8,8,8,8,4")
        self.assertEqual(sum(part), 78)

    def test_misaligned_partition_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            validate_partition(target_config(), "8,8,8,8,8,8,8,8,8,6")

    def test_versions_are_immutable_pins(self) -> None:
        versions = (NEXT / "versions.env").read_text()
        self.assertIn(
            "VLLM_COMMIT=378c37b0098a41a5cd25b3bf8b56d158e33a6cbf", versions
        )
        self.assertIn(
            "DFLASH_REVISION=425aa615ce320caac34400208b30808c8f14f76c", versions
        )
        self.assertNotIn("VLLM_COMMIT=main", versions)

    def test_launch_has_dflash_and_fp8_contract(self) -> None:
        serve = (NEXT / "serve.sh").read_text()
        self.assertIn('"method\\":\\"dflash', serve)
        self.assertIn("--kv-cache-dtype fp8_ds_mla", serve)
        self.assertIn("--pipeline-parallel-size 10", serve)
        self.assertIn("VLLM_GLM5_DECODE_IDX_GLUE=0", serve)
        self.assertIn("PROFILE=${PROFILE:-smoke}", serve)

    def test_async_scheduling_is_default(self) -> None:
        # PP + DFlash2 with --no-async-scheduling asserts on the first short
        # prompt (drafts pulled into a request still prefilling in flight).
        for name in ("serve.sh", "smoke_reduced.sh"):
            text = (NEXT / name).read_text()
            self.assertIn('if [ "${ASYNC_SCHED:-1}" != 1 ]', text, name)
            self.assertNotIn("TRITON_MLA_SPARSE --no-async-scheduling", text, name)

    def test_sm80_dsa_patches_are_wired(self) -> None:
        patch = (NEXT / "apply_engine_patch.py").read_text()
        preflight = (NEXT / "preflight.py").read_text()
        for marker in ("local-cmp170hx-dsv32-sm80-fp8", "local-cmp170hx-dsv32-sm80-indexer"):
            self.assertIn(marker, patch)
            self.assertIn(marker, preflight)
        self.assertIn("kernels = patch_dsv32_sm80_fp8(", patch)
        self.assertIn("indexer = patch_indexer_sm80(", patch)

    def test_topk_is_canonical_not_tiefix(self) -> None:
        # TIEFIX/SORTED corrupted decode on this target (2026-09-28); the
        # fork's canonical path costs ~48 ms/step. Use the v0.26 kernel.
        serve = (NEXT / "serve.sh").read_text()
        self.assertIn("GLM53_TOPK_CANON=${GLM53_TOPK_CANON:-1}", serve)
        self.assertIn("VLLM_GLM5_TOPK_TIEFIX=${VLLM_GLM5_TOPK_TIEFIX:-0}", serve)
        self.assertIn("VLLM_GLM5_TOPK_SORTED=${VLLM_GLM5_TOPK_SORTED:-0}", serve)
        self.assertIn("VLLM_GLM5_TOPK_CANONICAL=${VLLM_GLM5_TOPK_CANONICAL:-0}", serve)
        patch = (NEXT / "apply_engine_patch.py").read_text()
        self.assertIn("def patch_topk_canon_decode(", patch)
        self.assertIn("canon_topk_indices_prefill(", patch)

    def test_drafter_kv_is_window_bounded(self) -> None:
        # 64-token MLA blocks cannot host a >=16-token BF16 drafter page, so
        # the drafter would fall back to full-length KV (~0.96M cap).
        serve = (NEXT / "serve.sh").read_text()
        self.assertIn('--block-size "${BLOCK_SIZE:-128}"', serve)
        patch = (NEXT / "apply_engine_patch.py").read_text()
        self.assertIn("kvu = patch_kv_cache_dsa_dflash(", patch)
        self.assertIn("local-cmp170hx-dsa-dflash-kv", (NEXT / "preflight.py").read_text())

    def test_json_defaults_not_inside_param_expansion(self) -> None:
        # ${VAR:-{...}} ends at the first "}" and corrupts overrides.
        serve = (NEXT / "serve.sh").read_text()
        self.assertNotIn(":-{", serve)

    def test_shell_scripts_parse(self) -> None:
        scripts = sorted(NEXT.glob("*.sh"))
        subprocess.run(["bash", "-n", *map(str, scripts)], check=True)

    def test_python_migration_tools_parse(self) -> None:
        for path in NEXT.glob("*.py"):
            compile(path.read_text(), str(path), "exec")


if __name__ == "__main__":
    unittest.main()
