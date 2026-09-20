#!/usr/bin/env python3
"""CPU-only launch/configuration regressions; never starts vLLM or touches GPUs."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import bench_matrix

ROOT = Path(__file__).resolve().parent
PARTITION = "6,7,7,7,7,7,7,7,7,6,6,4"


class LaunchProfiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.install = Path(self.tmp.name)
        bin_dir = self.install / ".venv/bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "python").symlink_to(sys.executable)
        executable = bin_dir / "vllm"
        executable.write_text(
            f"#!{sys.executable}\nimport json, os, sys\n"
            "print(json.dumps({'argv': sys.argv[1:], 'env': dict(os.environ)}))\n"
        )
        executable.chmod(0o755)
        (self.install / "cuda/include").mkdir(parents=True)
        (self.install / "cuda/include/curand.h").touch()
        (self.install / "vllm").mkdir()
        (self.install / "vllm/__init__.py").touch()
        self.kernel = self.install / "vllm/_glm52_mla_fp8.py"
        self.kernel.write_text("KV_ADDRESS_BITS = 64\n")
        self.env = {
            "PATH": os.environ["PATH"], "HOME": os.environ["HOME"],
            "VLLM_INSTALL_DIR": str(self.install),
            "CUDA_HOME": str(self.install / "cuda"),
            "PYTHONPATH": str(self.install),
        }

    def launch(self, script="serve_glm52.sh", **env):
        return subprocess.run(["bash", str(ROOT / script)],
                              env=self.env | env, capture_output=True, text=True)

    def config(self, script="serve_glm52.sh", **env):
        result = self.launch(script, **env)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_default_launchers_and_capacity(self):
        for script in ("serve_glm52.sh", "serve_glm53.sh"):
            with self.subTest(script=script):
                result = self.config(script)
                argv, env = result["argv"], result["env"]
                self.assertEqual(argv[argv.index("--kv-cache-memory") + 1], "21474836480")
                self.assertEqual(env["VLLM_PP_LAYER_PARTITION"], PARTITION)
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], ",".join(map(str, range(12))))
                self.assertEqual(env["GLM52_PP_DECODE_ADAPTIVE"], "12")
                self.assertEqual(env["GLM52_SPLIT_MOE"], "0")
                kernel = json.loads(argv[argv.index("--kernel-config") + 1])
                self.assertEqual(kernel["ir_op_priority"]["rms_norm"], ["vllm_c"])
        # Capacity is limited by the heaviest local cache, not total VRAM.
        start, fp8, bf16 = 0, [], []
        for rank, count in enumerate(map(int, PARTITION.split(","))):
            indexers = sum(i in (0, 1, 2) or (i >= 6 and i % 4 == 2)
                           for i in range(start, start + count))
            start += count
            if rank == 11:  # one MTP layer and its indexer cache
                count += 1
                indexers += 1
            fp8.append(count * 656 + indexers * 132)
            bf16.append(count * 1152 + indexers * 132)
        self.assertEqual(start, 78)
        self.assertEqual(max(fp8), 4856)
        self.assertEqual(max(bf16), 8328)
        self.assertEqual(20 * 1024**3 // (max(fp8) * 64) * 64, 4422272)
        self.assertEqual(20 * 1024**3 // (max(bf16) * 64) * 64, 2578624)
        self.assertGreater((4422272 - 1) * 656, 2**31)

    def test_eight_cards_and_speculation_off(self):
        result = self.config(PP="8", SPEC_TOKENS="0")
        self.assertEqual(result["env"]["CUDA_VISIBLE_DEVICES"], "0,1,2,3,4,5,6,7")
        self.assertEqual(result["env"]["GLM52_PP_DECODE_ADAPTIVE"], "8")
        self.assertIn("8258584576", result["argv"])
        self.assertNotIn("--speculative-config", result["argv"])
        self.assertEqual(self.config(SPEC_TOKENS="0")["env"]["VLLM_PP_LAYER_PARTITION"], PARTITION)
        self.assertEqual(self.config(VLLM_PP_LAYER_PARTITION="")["env"]["VLLM_PP_LAYER_PARTITION"], PARTITION)

    def test_memory_and_kernel_overrides(self):
        self.assertNotIn("--kv-cache-memory", self.config(KV_CACHE_MEM="")["argv"])
        self.assertIn("12884901888", self.config(KV_CACHE_MEM="12884901888")["argv"])
        self.assertNotIn("--kernel-config", self.config(KERNEL_CFG_JSON="")["argv"])
        custom = self.config(PP="2", VLLM_PP_LAYER_PARTITION="39,39", SPEC_TOKENS="0")
        self.assertNotIn("--kv-cache-memory", custom["argv"])
        self.kernel.write_text("# Old installed kernel lacks 64-bit addressing\n")
        failure = self.launch()
        self.assertNotEqual(failure.returncode, 0)
        self.assertIn("Install the updated FP8 kernel", failure.stderr)
        self.assertEqual(self.launch(KV_CACHE_DTYPE="auto").returncode, 0)

    def test_invalid_configurations_fail_before_launch(self):
        for env in (
            {"PP": "0"}, {"TP": "2"}, {"SPEC_TOKENS": "-1"},
            {"SPEC_TOKENS": "bad"}, {"KV_CACHE_MEM": "0"},
            {"CUDA_VISIBLE_DEVICES": ""}, {"CUDA_VISIBLE_DEVICES": "0,1"},
            {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7,8,9,10,10"},
            {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7,8,9,10,"},
            {"VLLM_PP_LAYER_PARTITION": PARTITION + ","},
            {"VLLM_PP_LAYER_PARTITION": "06,7,7,7,7,7,7,7,7,6,6,4"},
            {"VLLM_PP_LAYER_PARTITION": "0,13,7,7,7,7,7,7,7,6,6,4"},
            {"VLLM_PP_LAYER_PARTITION": "6,7,7"},
        ):
            with self.subTest(env=env):
                result = self.launch(**env)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")


class Benchmarks(unittest.TestCase):
    def test_worker_failures_propagate(self):
        with patch.object(bench_matrix, "stream_once", side_effect=RuntimeError("request failed")):
            with self.assertRaisesRegex(RuntimeError, "request failed"):
                bench_matrix.parallel_requests(["first", "second"], 1, 0)

    def test_default_contexts_fit_server(self):
        with patch.object(sys, "argv", ["bench_matrix.py", "--only", "single"]), \
             patch.object(bench_matrix, "run_case", return_value=None) as run, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            bench_matrix.main()
        self.assertEqual([c.args[1] for c in run.call_args_list], [32768, 262144])
        self.assertIn("1024K skipped", output.getvalue())


if __name__ == "__main__":
    unittest.main()
