#!/usr/bin/env python3
"""Authenticating, context-enforcing gateway in front of the local GLM-5.3 vLLM.

Enforces the client contract as fixed per-model context windows, so the
client's agent harness always knows its window and can auto-compact:

    glm-5.3-512k   512K-token window, 1 concurrent request   (main agent)
    glm-5.3-256k   256K-token window, 2 concurrent requests  (subagents)

Both are the same served model; the name only selects the window and the
concurrency pool. /v1/models reports each window (max_model_len,
context_length, context_window) for harnesses that discover it.

Prompt tokens are counted exactly by vLLM (/tokenize or
/v1/messages/count_tokens: chat template and tool schemas included). A prompt
that leaves less than MIN_OUTPUT_TOKENS of room gets the standard 400
context-length error in the caller's API dialect ("context_length_exceeded"
for OpenAI, "prompt is too long" for Anthropic): the signal harnesses compact
on. Otherwise max_tokens is clamped so generation cannot run past the window.
A request whose pool is busy waits up to QUEUE_TIMEOUT_S, then gets 429.

Only /v1/models, /v1/chat/completions, /v1/completions, /v1/messages,
/v1/messages/count_tokens and /v1/slots are exposed; every other vLLM route
(/tokenize, /metrics, /scale_elastic_ep, ...) is unreachable from outside.

Keys are stored as SHA-256 hashes only:
    python3 glm_gateway.py add-key <name>     # prints the key once
    python3 glm_gateway.py serve

Requests from the clients named in REQUEST_LOG_CLIENTS are written, as sent
to vLLM, one JSON object per line to REQUEST_LOG_FILE (size-rotated):
    python3 glm_gateway.py tail -f            # readable live view
"""
import argparse
import asyncio
import collections
import hashlib
import hmac
import json
import logging
import logging.handlers
import os
import secrets
import sys
import time

from aiohttp import ClientConnectionError, ClientSession, ClientTimeout, web

UPSTREAM = os.environ.get("UPSTREAM_URL", "http://127.0.0.1:8002")
UPSTREAM_KEY = os.environ.get("UPSTREAM_API_KEY", "")
SERVED_MODEL = os.environ.get("SERVED_MODEL", "glm-5.3")
KEYS_FILE = os.environ.get("KEYS_FILE", os.path.expanduser("~/.config/glm-gateway/keys.json"))
LISTEN_HOST = os.environ.get("LISTEN_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "8100"))
# name:window_tokens:concurrency, comma separated.
MODELS = os.environ.get("MODELS", "glm-5.3-512k:524288:1,glm-5.3-256k:262144:2")
# A prompt must leave at least this much room for the reply, else it is
# reported as too long (so the harness compacts instead of getting a
# truncated one-line answer).
MIN_OUTPUT_TOKENS = int(os.environ.get("MIN_OUTPUT_TOKENS", "4096"))
QUEUE_TIMEOUT_S = float(os.environ.get("QUEUE_TIMEOUT_S", "120"))
MAX_WAITING = int(os.environ.get("MAX_WAITING", "6"))
# A 512K-token prompt is a few MB of JSON; anything far beyond is abuse.
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", str(48 * 1024 * 1024)))
# Comma-separated key names whose requests are logged in full; empty = none.
REQUEST_LOG_CLIENTS = {c.strip() for c in os.environ.get("REQUEST_LOG_CLIENTS", "").split(",") if c.strip()}
# tmpfs, so the log never touches the SSD (and is lost on reboot).
REQUEST_LOG_FILE = os.environ.get("REQUEST_LOG_FILE", "/run/glm-gateway/requests.jsonl")
# Each record carries the whole conversation, so an agent session at 80K
# tokens writes ~300 KB per turn. 128 MiB x 4 caps the RAM use at 512 MiB.
REQUEST_LOG_MAX_BYTES = int(os.environ.get("REQUEST_LOG_MAX_BYTES", str(128 * 1024 * 1024)))
REQUEST_LOG_BACKUPS = int(os.environ.get("REQUEST_LOG_BACKUPS", "3"))

# Fields that would multiply sequences, bypass the accounting, or let the
# client inject server-side templates/processors.
DENIED_FIELDS = {
    "best_of", "use_beam_search", "prompt_logprobs", "chat_template",
    "kv_transfer_params", "vllm_xargs", "logits_processors",
    "mm_processor_kwargs", "priority", "cache_salt",
}

log = logging.getLogger("glm-gateway")
reqlog = logging.getLogger("glm-gateway.requests")
reqlog.propagate = False  # bodies go to the file only, never the journal


def is_anthropic(path):
    return path.startswith("/v1/messages")


def err(path, status, message, etype="invalid_request_error", code=None, headers=None):
    if is_anthropic(path):
        body = {"type": "error", "error": {"type": etype, "message": message}}
    else:
        body = {"error": {"message": message, "type": etype, "param": None, "code": code}}
    return web.json_response(body, status=status, headers=headers)


class UpstreamUnavailable(Exception):
    """vLLM is down or restarting (connection refused, or a 5xx from it)."""


def unavailable(path):
    # 503 + Retry-After, never 400: a 400 tells the client's harness its
    # request is bad, so it gives up instead of retrying once GLM is back.
    etype = "api_error" if is_anthropic(path) else "server_error"
    return err(path, 503, "the model server is temporarily unavailable (restarting); retry shortly",
               etype, code="upstream_unavailable", headers={"Retry-After": "30"})


def context_error(path, window, prompt_tokens):
    limit = window - MIN_OUTPUT_TOKENS
    if is_anthropic(path):
        # Claude-style harnesses match "prompt is too long".
        msg = f"prompt is too long: {prompt_tokens} tokens > {limit} maximum"
    else:
        msg = (f"This model's maximum context length is {window} tokens. However, your "
               f"messages resulted in {prompt_tokens} tokens, leaving less than "
               f"{MIN_OUTPUT_TOKENS} for the completion. Please reduce the length of the messages.")
    return err(path, 400, msg, code="context_length_exceeded")


class Pool:
    """A fixed number of concurrent requests sharing one context window."""

    def __init__(self, name, window, size):
        self.name, self.window = name, window
        self.busy = [None] * size
        self.cond = asyncio.Condition()
        self.waiting = 0

    def _free(self):
        return next((i for i, b in enumerate(self.busy) if b is None), None)

    async def acquire(self, holder, timeout):
        async with self.cond:
            if self._free() is None:
                if self.waiting >= MAX_WAITING:
                    return None
                self.waiting += 1
                try:
                    await asyncio.wait_for(self.cond.wait_for(lambda: self._free() is not None), timeout)
                except asyncio.TimeoutError:
                    return None
                finally:
                    self.waiting -= 1
            idx = self._free()
            self.busy[idx] = holder
            return idx

    async def release(self, idx):
        async with self.cond:
            self.busy[idx] = None
            self.cond.notify_all()

    def snapshot(self):
        return {"model": self.name, "context_window": self.window, "waiting": self.waiting,
                "slots": [{"busy": b is not None, **(b or {})} for b in self.busy]}


def parse_models(spec):
    pools = {}
    for item in spec.split(","):
        name, window, size = item.strip().split(":")
        pools[name] = Pool(name, int(window), int(size))
    return pools


def load_keys():
    with open(KEYS_FILE) as f:
        return json.load(f)  # {"<sha256 hex>": "<name>"}


def save_keys(keys):
    # Atomic, so the running gateway never reads a half-written file.
    fd = os.open(KEYS_FILE + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(keys, f, indent=1)
    os.replace(KEYS_FILE + ".tmp", KEYS_FILE)


class KeyStore:
    """keys.json, reloaded whenever it changes: add-key and revoke-key take
    effect on the next request, no restart needed."""

    def __init__(self):
        self.mtime = os.stat(KEYS_FILE).st_mtime_ns
        self.keys = load_keys()

    def current(self):
        try:
            mtime = os.stat(KEYS_FILE).st_mtime_ns
        except OSError as e:
            log.error("keys file unreadable, keeping %d loaded key(s): %s", len(self.keys), e)
            return self.keys
        if mtime != self.mtime:
            try:
                self.keys, self.mtime = load_keys(), mtime
                log.info("reloaded keys: %s", ", ".join(sorted(set(self.keys.values()))) or "none")
            except (OSError, ValueError) as e:
                log.error("keys reload failed, keeping previous set: %s", e)
        return self.keys


def authenticate(request, keys):
    auth = request.headers.get("Authorization", "")
    token = auth[7:] if auth.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
    if not token:
        return None
    digest = hashlib.sha256(token.encode()).hexdigest()
    for known, name in keys.items():
        if hmac.compare_digest(known, digest):
            return name
    return None


@web.middleware
async def auth_mw(request, handler):
    name = authenticate(request, request.app["keystore"].current())
    if name is None:
        await asyncio.sleep(0.5)  # blunt online guessing
        return err(request.path, 401, "invalid or missing API key", "authentication_error")
    request["client"] = name
    return await handler(request)


async def count_prompt(app, path, body):
    if path == "/v1/chat/completions":
        url = "/tokenize"
        req = {"model": SERVED_MODEL, "messages": body.get("messages") or [],
               "add_generation_prompt": body.get("add_generation_prompt", True),
               "continue_final_message": body.get("continue_final_message", False)}
        for k in ("tools", "chat_template_kwargs"):
            if body.get(k) is not None:
                req[k] = body[k]
    elif path == "/v1/completions":
        url, req = "/tokenize", {"model": SERVED_MODEL, "prompt": body.get("prompt", "")}
    else:
        url, req = "/v1/messages/count_tokens", dict(body, model=SERVED_MODEL)
        req.pop("stream", None)
    async with app["tok_sem"]:
        try:
            r = await app["http"].post(UPSTREAM + url, json=req, headers=app["up_headers"])
        except ClientConnectionError as e:
            raise UpstreamUnavailable(str(e)) from e
        async with r:
            if r.status >= 500:
                raise UpstreamUnavailable(f"tokenizer returned {r.status}")
            data = await r.json(content_type=None)
            if r.status != 200:
                e = data.get("error")
                raise ValueError((e.get("message") if isinstance(e, dict) else e)
                                 or data.get("message") or str(data))
            return int(data["count"] if "count" in data else data["input_tokens"])


def resolve_pool(request, body):
    name = body.get("model")
    pool = request.app["pools"].get(name)
    if pool is None:
        names = ", ".join(request.app["pools"])
        return None, err(request.path, 404 if name else 400,
                         f"model {name!r} not found; available: {names}",
                         "not_found_error", code="model_not_found")
    return pool, None


async def read_body(request):
    try:
        body = await request.json()
        assert isinstance(body, dict)
        return body, None
    except Exception:
        return None, err(request.path, 400, "request body must be a JSON object")


async def handle_models(request):
    now = int(time.time())
    data = [{"id": p.name, "object": "model", "created": now, "owned_by": "glm-gateway",
             "type": "model", "display_name": p.name,
             "max_model_len": p.window, "context_length": p.window, "context_window": p.window,
             "max_concurrency": len(p.busy)}
            for p in request.app["pools"].values()]
    return web.json_response({"object": "list", "data": data})


async def handle_slots(request):
    return web.json_response({"models": [p.snapshot() for p in request.app["pools"].values()]})


async def handle_count_tokens(request):
    body, e = await read_body(request)
    if e:
        return e
    pool, e = resolve_pool(request, body)
    if e:
        return e
    try:
        n = await count_prompt(request.app, request.path, body)
    except UpstreamUnavailable as ex:
        log.error("upstream unavailable during count_tokens: %s", ex)
        return unavailable(request.path)
    except Exception as ex:
        return err(request.path, 400, f"could not tokenize request: {ex}")
    return web.json_response({"input_tokens": n})


async def handle_generate(request):
    app, path, client = request.app, request.path, request["client"]
    body, e = await read_body(request)
    if e:
        return e
    pool, e = resolve_pool(request, body)
    if e:
        return e

    bad = sorted(DENIED_FIELDS & body.keys())
    if bad:
        return err(path, 400, f"unsupported field(s): {', '.join(bad)}")
    if body.get("n", 1) != 1:
        return err(path, 400, "n must be 1")
    if path == "/v1/completions" and not isinstance(body.get("prompt"), str):
        return err(path, 400, "prompt must be a single string")
    out_key = "max_completion_tokens" if body.get("max_completion_tokens") is not None else "max_tokens"
    requested = body.get(out_key)
    if requested is not None and (not isinstance(requested, int) or requested < 1):
        return err(path, 400, f"{out_key} must be a positive integer")

    body["model"] = SERVED_MODEL
    try:
        prompt_tokens = await count_prompt(app, path, body)
    except UpstreamUnavailable as ex:
        log.error("client=%s upstream unavailable: %s", client, ex)
        return unavailable(path)
    except Exception as ex:
        return err(path, 400, f"could not tokenize request: {ex}")
    if prompt_tokens + MIN_OUTPUT_TOKENS > pool.window:
        log.info("client=%s model=%s prompt=%d -> context_length_exceeded", client, pool.name, prompt_tokens)
        return context_error(path, pool.window, prompt_tokens)
    # The completion may use at most what is left in the window.
    body.pop("max_completion_tokens", None)
    body["max_tokens"] = min(requested or pool.window, pool.window - prompt_tokens)

    holder = {"client": client, "prompt_tokens": prompt_tokens, "since": int(time.time())}
    idx = await pool.acquire(holder, QUEUE_TIMEOUT_S)
    if idx is None:
        return err(path, 429, f"all {len(pool.busy)} {pool.name} slots are busy; retry shortly",
                   "rate_limit_error", headers={"Retry-After": "10"})

    if client in REQUEST_LOG_CLIENTS and reqlog.handlers:
        try:
            reqlog.info(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                    "client": client, "model": pool.name, "path": path,
                                    "prompt_tokens": prompt_tokens, "body": body},
                                   ensure_ascii=False))
        except Exception as ex:  # never fail a request over its log line
            log.error("request log write failed: %s", ex)

    t0 = time.monotonic()
    status = 0
    resp = None
    try:
        async with app["http"].post(UPSTREAM + path, json=body, headers=app["up_headers"]) as up:
            status = up.status
            resp = web.StreamResponse(status=up.status)
            resp.content_type = up.content_type
            await resp.prepare(request)
            async for chunk in up.content.iter_any():
                await resp.write(chunk)
            await resp.write_eof()
            return resp
    except ClientConnectionError as ex:
        if resp is not None:
            raise  # mid-stream: the status is already sent, just drop the connection
        log.error("client=%s upstream unavailable: %s", client, ex)
        status = 503
        return unavailable(path)
    finally:
        # Also reached on client disconnect (CancelledError): leaving the
        # upstream context manager closes the connection, which makes vLLM
        # abort the request and free its KV blocks.
        await pool.release(idx)
        log.info("client=%s model=%s path=%s prompt=%d max_tokens=%d status=%s %.1fs",
                 client, pool.name, path, prompt_tokens, body["max_tokens"], status,
                 time.monotonic() - t0)


async def on_startup(app):
    app["http"] = ClientSession(timeout=ClientTimeout(total=None, sock_connect=10))
    app["pools"] = parse_models(MODELS)
    app["tok_sem"] = asyncio.Semaphore(2)


async def on_cleanup(app):
    await app["http"].close()


def make_app():
    app = web.Application(middlewares=[auth_mw], client_max_size=MAX_BODY_BYTES)
    app["keystore"] = KeyStore()
    if not app["keystore"].keys:
        sys.exit(f"no keys in {KEYS_FILE}; run: {sys.argv[0]} add-key <name>")
    app["up_headers"] = {"Authorization": f"Bearer {UPSTREAM_KEY}"} if UPSTREAM_KEY else {}
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/v1/slots", handle_slots)
    app.router.add_post("/v1/chat/completions", handle_generate)
    app.router.add_post("/v1/completions", handle_generate)
    app.router.add_post("/v1/messages", handle_generate)
    app.router.add_post("/v1/messages/count_tokens", handle_count_tokens)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def open_request_log():
    if not REQUEST_LOG_CLIENTS:
        return
    try:
        h = logging.handlers.RotatingFileHandler(REQUEST_LOG_FILE, maxBytes=REQUEST_LOG_MAX_BYTES,
                                                 backupCount=REQUEST_LOG_BACKUPS, encoding="utf-8")
    except OSError as e:
        # Serving matters more than the log: run without it.
        log.error("request log disabled, cannot open %s: %s", REQUEST_LOG_FILE, e)
        return
    h.setFormatter(logging.Formatter("%(message)s"))
    reqlog.addHandler(h)
    reqlog.setLevel(logging.INFO)
    log.info("logging requests from %s to %s", ", ".join(sorted(REQUEST_LOG_CLIENTS)), REQUEST_LOG_FILE)


def text_of(content):
    """Flatten an OpenAI or Anthropic message content to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if not isinstance(p, dict):
                parts.append(str(p))
            elif p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif p.get("type") == "tool_result":
                parts.append(text_of(p.get("content")))
            elif p.get("type") == "tool_use":
                parts.append(f"[tool_use {p.get('name')}] {json.dumps(p.get('input'), ensure_ascii=False)}")
            else:
                parts.append(f"[{p.get('type')}]")
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False)


def describe(rec, full, width):
    """Header line plus the input that is new in this request: every message
    after the last assistant turn (the user's turn, or the tool results)."""
    body = rec.get("body", {})
    out = [f"=== {rec.get('ts')} {rec.get('client')} {rec.get('model')} {rec.get('path')} "
           f"prompt={rec.get('prompt_tokens')} max_tokens={body.get('max_tokens')} "
           f"msgs={len(body.get('messages') or [])} tools={len(body.get('tools') or [])}"]
    msgs = body.get("messages")
    if msgs is None:  # /v1/completions
        msgs = [{"role": "prompt", "content": body.get("prompt", "")}]
    if not full:
        last = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=-1)
        msgs = msgs[last:] if last >= 0 else msgs
    for m in msgs:
        text = text_of(m.get("content"))
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            text += f"\n[tool_call {fn.get('name')}] {fn.get('arguments')}"
        text = text.strip()
        if width and len(text) > width:
            text = text[:width] + f" ... [+{len(text) - width} chars]"
        out.append(f"--- {m.get('role')}: {text}")
    return "\n".join(out)


def tail(a):
    def show(line):
        try:
            rec = json.loads(line)
        except ValueError:
            return
        print(line.rstrip("\n") if a.raw else describe(rec, a.full, a.width) + "\n", flush=True)

    try:
        f = open(a.file, "rb")
    except OSError as e:
        sys.exit(f"cannot open {a.file}: {e}")
    try:
        if a.n > 0:
            for line in collections.deque(f, maxlen=a.n):
                show(line.decode("utf-8", "replace"))
        else:
            f.seek(0, os.SEEK_END)
        if not a.follow:
            return
        ino, buf = os.fstat(f.fileno()).st_ino, b""
        while True:
            chunk = f.read()
            if chunk:
                buf += chunk
                *lines, buf = buf.split(b"\n")  # keep a partly written record
                for line in lines:
                    show(line.decode("utf-8", "replace"))
                continue
            time.sleep(0.5)
            try:
                if os.stat(a.file).st_ino != ino:  # rotated
                    f.close()
                    f, buf = open(a.file, "rb"), b""
                    ino = os.fstat(f.fileno()).st_ino
            except FileNotFoundError:
                pass
    except KeyboardInterrupt:
        pass
    finally:
        f.close()


def add_key(name):
    os.makedirs(os.path.dirname(KEYS_FILE), mode=0o700, exist_ok=True)
    keys = load_keys() if os.path.exists(KEYS_FILE) else {}
    token = "glm_" + secrets.token_urlsafe(32)
    keys[hashlib.sha256(token.encode()).hexdigest()] = name
    save_keys(keys)
    print(f"key for {name!r} (shown once; active immediately):\n{token}")


def revoke_key(name):
    keys = load_keys()
    kept = {h: n for h, n in keys.items() if n != name}
    if len(kept) == len(keys):
        sys.exit(f"no key named {name!r}")
    save_keys(kept)
    print(f"revoked {len(keys) - len(kept)} key(s) named {name!r}; effective immediately")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve")
    sub.add_parser("add-key").add_argument("name")
    sub.add_parser("revoke-key").add_argument("name")
    t = sub.add_parser("tail", help="show logged requests (REQUEST_LOG_CLIENTS)")
    t.add_argument("-f", "--follow", action="store_true", help="keep watching for new requests")
    t.add_argument("-n", type=int, default=5, help="show the last N requests first (default 5)")
    t.add_argument("--full", action="store_true", help="every message, not just the new input")
    t.add_argument("--width", type=int, default=600, help="truncate each message (0 = never)")
    t.add_argument("--raw", action="store_true", help="print the JSON lines unchanged")
    t.add_argument("--file", default=REQUEST_LOG_FILE)
    a = p.parse_args()
    if a.cmd == "add-key":
        return add_key(a.name)
    if a.cmd == "revoke-key":
        return revoke_key(a.name)
    if a.cmd == "tail":
        return tail(a)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("models=%s upstream=%s listen=%s:%d", MODELS, UPSTREAM, LISTEN_HOST, LISTEN_PORT)
    open_request_log()
    web.run_app(make_app(), host=LISTEN_HOST, port=LISTEN_PORT, access_log=None)


if __name__ == "__main__":
    main()
