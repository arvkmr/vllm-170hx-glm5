"""Graph taps (GLM52_TAP=<layer idx csv>): an opaque custom op inserted at chosen points of
the model code so intermediate tensors of the COMPILED graph can be fingerprinted on real
steps. Inert (no op inserted, graph unchanged) unless GLM52_TAP is set at import time.
GLM52_TAP_CALLS (default 3) = calls per tag to report."""
import hashlib
import os
import sys

import torch

from vllm.utils.torch_utils import direct_register_custom_op

_LAYERS = {s.strip() for s in os.environ.get("GLM52_TAP", "").split(",") if s.strip()}
_CALLS = int(os.environ.get("GLM52_TAP_CALLS", "3") or 3)
_cnt: dict[str, int] = {}


def _tap_impl(x: torch.Tensor, tag: str) -> None:
    from vllm import _glm52_steptrace as t
    if not t._st.get("real"):
        return
    n = _cnt.get(tag, 0)
    if n >= _CALLS:
        return
    _cnt[tag] = n + 1
    try:
        b = x.detach().contiguous()
        b = b.view(torch.uint8) if b.dtype != torch.bool else b.to(torch.uint8)
        h = hashlib.sha256(b.cpu().numpy().tobytes()).hexdigest()[:12]
    except Exception as e:  # noqa: BLE001
        h = f"ERR:{e!r}"[:40]
    print(f"[glm52-tap] {tag} call{n} {h} {tuple(x.shape)}/{tuple(x.stride())}/{str(x.dtype).replace('torch.', '')}",
          file=sys.stderr, flush=True)


def _tap_fake(x: torch.Tensor, tag: str) -> None:
    return None


direct_register_custom_op(op_name="glm52_tap", op_func=_tap_impl, mutates_args=["x"], fake_impl=_tap_fake)


def _li(layer) -> str:
    # Dynamo-traceable (no `re`): int index or a "...layers.<i>..." prefix string.
    if isinstance(layer, int):
        return str(layer)
    s = str(layer)
    if "layers." in s:
        return s.split("layers.", 1)[1].split(".", 1)[0]
    return "?"


def tap(x, layer, name):
    """Insert a tap for tensor x if this layer is selected. `layer` = int index or prefix."""
    if not _LAYERS:
        return
    li = _li(layer)
    if li in _LAYERS and isinstance(x, torch.Tensor):
        torch.ops.vllm.glm52_tap(x, f"L{li}.{name}")
