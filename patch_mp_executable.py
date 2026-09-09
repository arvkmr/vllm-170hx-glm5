#!/usr/bin/env python3
"""GLM52_MP_EXECUTABLE=<path>: make vLLM's multiprocessing context spawn workers with that
executable (a wrapper that runs the real python under compute-sanitizer). Requires
VLLM_WORKER_MULTIPROC_METHOD=spawn. Inert unless the env is set."""
import os, sys, py_compile, vllm
F = os.path.join(os.path.dirname(vllm.__file__), "utils/system_utils.py")
s = open(F).read()
old = "def get_mp_context():\n"
assert s.count(old) == 1
i = s.index(old); j = s.index("\n", s.index("return", i))   # end of the first 'return ...' line inside
body = s[i:j+1]
if "GLM52_MP_EXECUTABLE" in s:
    print("= already")
else:
    new_body = body.replace("    return ", "    _ctx = ", 1) + "    if os.environ.get(\"GLM52_MP_EXECUTABLE\"):\n        _ctx.set_executable(os.environ[\"GLM52_MP_EXECUTABLE\"])\n    return _ctx\n"
    s = s[:i] + new_body + s[j+1:]
    open(F, "w").write(s); print("+ patched get_mp_context")
py_compile.compile(F, doraise=True); print("compiles")
