#!/usr/bin/env python3
"""PP channel-skew tripwire (env GLM52_PP_TRIPWIRE=1; idempotent patch).

Hypothesis under test: vLLM ships every PP hop as TWO independently-ordered
streams -- tensor shapes via gloo (send_object) and tensor bytes via NCCL --
matched only by arrival order. If any path ever emits one without the other,
the streams skew permanently and every later recv pairs metadata k with
tensor k+1: NCCL count mismatch, OOB write on the RECEIVER. This would
explain the rank-1-always illegal-memory-access under spec + diverse
concurrency (rank 1 = first receiver; same-size lockstep payloads make a
skew harmless, diverse sizes make it a crash lottery).

Mechanism: sender appends a "__tripwire" int64 tensor [seq, n_tensors,
total_numel] to every tensor dict (it travels the NCCL channel with its
metadata in the gloo channel). Receiver verifies DEFERRED by one hop (the
previous hop's tripwire is checked at the next recv, after its async D2H
copy has certainly landed) -- no added synchronization, race-preserving.
On mismatch: RuntimeError with both streams' evidence.
"""

import os
import sys

import vllm

VLLM = os.path.dirname(vllm.__file__)
PS = "distributed/parallel_state.py"


def edit(path, old, new, label):
    full = os.path.join(VLLM, path)
    src = open(full).read()
    if new in src:
        print(f"  = {label} (already applied)")
        return
    if old not in src:
        print(f"  ! {label}: ANCHOR NOT FOUND", file=sys.stderr)
        sys.exit(1)
    if src.count(old) != 1:
        print(f"  ! {label}: anchor not unique", file=sys.stderr)
        sys.exit(1)
    open(full, "w").write(src.replace(old, new))
    print(f"  + {label}")


def main():
    # helpers + state on GroupCoordinator
    edit(
        PS,
        "    def isend_tensor_dict(\n"
        "        self,\n"
        "        tensor_dict: dict[str, torch.Tensor | Any],\n",
        "    def _tw_enabled(self):\n"
        "        import os as _os\n"
        "        return _os.environ.get(\"GLM52_PP_TRIPWIRE\", \"0\") == \"1\"\n"
        "\n"
        "    def _tw_next_send(self, dst):\n"
        "        d = getattr(self, \"_tw_send_seq\", None)\n"
        "        if d is None:\n"
        "            d = self._tw_send_seq = {}\n"
        "        d[dst] = d.get(dst, 0) + 1\n"
        "        return d[dst]\n"
        "\n"
        "    def _tw_check_prev(self, src):\n"
        "        # verify the PREVIOUS hop's tripwire (its async D2H has landed\n"
        "        # by now); store nothing if tripwire disabled.\n"
        "        pend = getattr(self, \"_tw_pending\", None)\n"
        "        if not pend or src not in pend:\n"
        "            return\n"
        "        cpu_t, exp_seq, exp_n, exp_numel = pend.pop(src)\n"
        "        got = cpu_t.tolist()\n"
        "        if got != [exp_seq, exp_n, exp_numel]:\n"
        "            raise RuntimeError(\n"
        "                f\"GLM52 PP TRIPWIRE: channel skew from src={src}: \"\n"
        "                f\"tensor channel carried {got}, metadata channel \"\n"
        "                f\"expected [seq={exp_seq}, n={exp_n}, numel={exp_numel}]\"\n"
        "            )\n"
        "\n"
        "    def isend_tensor_dict(\n"
        "        self,\n"
        "        tensor_dict: dict[str, torch.Tensor | Any],\n",
        "tripwire: helpers",
    )
    # sender: append tripwire tensor before split
    edit(
        PS,
        "        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)\n"
        "        self.send_object(metadata_list, dst=dst)",
        "        if self._tw_enabled():\n"
        "            _n = sum(\n"
        "                1 for v in tensor_dict.values()\n"
        "                if isinstance(v, torch.Tensor) and v.numel() > 0\n"
        "            )\n"
        "            _numel = sum(\n"
        "                v.numel() for v in tensor_dict.values()\n"
        "                if isinstance(v, torch.Tensor)\n"
        "            )\n"
        "            _seq = self._tw_next_send(dst)\n"
        "            tensor_dict = dict(tensor_dict)\n"
        "            tensor_dict[\"__tripwire\"] = torch.tensor(\n"
        "                [_seq, _n, _numel], dtype=torch.int64, device=\"cuda\"\n"
        "            )\n"
        "        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)\n"
        "        self.send_object(metadata_list, dst=dst)",
        "tripwire: sender append",
    )
    # receiver: intercept the tripwire at the TOP of the metadata loop.
    # It must never enter tensor_dict: the generic all-gather branch
    # (taken whenever all_gather_group is passed, even at TP=1, since
    # numel % 1 == 0) registers a postprocess closure that does
    # tensor_dict[key] = all_gather(...) -- re-inserting the key AFTER any
    # post-loop pop, which is how it leaked into consumers as a
    # KeyError('__tripwire'). Post its irecv in metadata order (NCCL
    # matching), then defer verification.
    edit(
        PS,
        "        recv_metadata_list = self.recv_object(src=src)\n"
        "        tensor_dict: dict[str, Any] = {}\n"
        "        handles: list[Handle] = []\n"
        "        postprocess: list[Callable[[], None]] = []\n",
        "        recv_metadata_list = self.recv_object(src=src)\n"
        "        if self._tw_enabled():\n"
        "            self._tw_check_prev(src)\n"
        "        tensor_dict: dict[str, Any] = {}\n"
        "        handles: list[Handle] = []\n"
        "        postprocess: list[Callable[[], None]] = []\n",
        "tripwire: receiver check-prev",
    )
    edit(
        PS,
        "        for key, value in recv_metadata_list:\n"
        "            if isinstance(value, TensorMetadata):\n"
        "                full_tensor = torch.empty(\n"
        "                    value.size, dtype=value.dtype, device=value.device\n"
        "                )\n"
        "                if full_tensor.numel() == 0:",
        "        for key, value in recv_metadata_list:\n"
        "            if (\n"
        "                key == \"__tripwire\"\n"
        "                and self._tw_enabled()\n"
        "                and isinstance(value, TensorMetadata)\n"
        "            ):\n"
        "                # Receive in metadata order but keep it OUT of tensor_dict\n"
        "                # (the generic all-gather branch would re-insert it from a\n"
        "                # postprocess closure after any post-loop pop).\n"
        "                _twt = torch.empty(\n"
        "                    value.size, dtype=value.dtype, device=value.device\n"
        "                )\n"
        "                handles.append(\n"
        "                    torch.distributed.irecv(\n"
        "                        _twt, src=self.ranks[src], group=group\n"
        "                    )\n"
        "                )\n"
        "                _exp_seq = getattr(self, \"_tw_recv_seq\", None)\n"
        "                if _exp_seq is None:\n"
        "                    _exp_seq = self._tw_recv_seq = {}\n"
        "                _exp_seq[src] = _exp_seq.get(src, 0) + 1\n"
        "                _n = sum(\n"
        "                    1 for k, v in recv_metadata_list\n"
        "                    if k != \"__tripwire\" and isinstance(v, TensorMetadata)\n"
        "                    and len(v.size) and int(torch.tensor(v.size).prod()) > 0\n"
        "                )\n"
        "                _numel = sum(\n"
        "                    int(torch.tensor(v.size).prod()) if len(v.size) else 0\n"
        "                    for k, v in recv_metadata_list\n"
        "                    if k != \"__tripwire\" and isinstance(v, TensorMetadata)\n"
        "                )\n"
        "                def _tw_defer(\n"
        "                    twt=_twt, seq=_exp_seq[src], n=_n, numel=_numel, s=src\n"
        "                ):\n"
        "                    cpu_t = twt.to(\"cpu\", non_blocking=True)\n"
        "                    pend = getattr(self, \"_tw_pending\", None)\n"
        "                    if pend is None:\n"
        "                        pend = self._tw_pending = {}\n"
        "                    pend[s] = (cpu_t, seq, n, numel)\n"
        "                postprocess.append(_tw_defer)\n"
        "                continue\n"
        "            if isinstance(value, TensorMetadata):\n"
        "                full_tensor = torch.empty(\n"
        "                    value.size, dtype=value.dtype, device=value.device\n"
        "                )\n"
        "                if full_tensor.numel() == 0:",
        "tripwire: receiver in-loop intercept",
    )


if __name__ == "__main__":
    main()
