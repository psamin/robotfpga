"""Tiny Jupyter client for the KR260 (PYNQ Jupyter, default password): upload files, run code.
Runs on the laptop; needs `websockets`. Board address: BOARD_JUPYTER (default 192.168.137.35:9090).
The kernel runs as root on PYNQ, which the overlay needs.

usage:
  python jb.py ls [path]
  python jb.py put LOCAL REMOTE_PATH        # any size: sent in 24 MB chunks, joined on the board
  python jb.py run FILE_OR_CODE [--kernel ID] [--timeout S]
  python jb.py restart
"""

import asyncio
import base64
import http.cookiejar
import json
import os
import re
import sys
import urllib.parse
import urllib.request
import uuid

import websockets

HOST = os.environ.get("BOARD_JUPYTER", "192.168.137.35:9090")
BASE = f"http://{HOST}"
STATE = os.path.join(os.path.expanduser("~"), ".kr260_jb_kernel")
CHUNK = 24 << 20

jar = http.cookiejar.CookieJar()
op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


def xsrf():
    return next((c.value for c in jar if c.name == "_xsrf"), "")


def login():
    op.open(f"{BASE}/login").read()  # sets the _xsrf cookie
    data = urllib.parse.urlencode({"password": "xilinx", "_xsrf": xsrf()}).encode()
    op.open(urllib.request.Request(f"{BASE}/login?next=%2F", data=data))
    if not any(c.name.startswith("username") for c in jar):
        sys.exit("login failed")


def api(method, path, body=None):
    req = urllib.request.Request(
        BASE + "/api/" + path,
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"X-XSRFToken": xsrf(), "Content-Type": "application/json"},
    )
    r = op.open(req, timeout=600).read()
    return json.loads(r) if r else None


def kernel(kid=None):
    if kid is None and os.path.exists(STATE):
        kid = open(STATE).read().strip()
    if kid:
        try:
            api("GET", f"kernels/{kid}")
            return kid
        except Exception:
            pass
    kid = api("POST", "kernels", {"name": "python3"})["id"]
    open(STATE, "w").write(kid)
    return kid


async def execute(kid, code, timeout, quiet=False, sink=None):
    cookie = "; ".join(f"{c.name}={c.value}" for c in jar)
    url = f"ws://{HOST}/api/kernels/{kid}/channels?session_id={uuid.uuid4().hex}"
    async with websockets.connect(
        url, additional_headers={"Cookie": cookie, "X-XSRFToken": xsrf()}, max_size=None
    ) as ws:
        mid = uuid.uuid4().hex
        await ws.send(
            json.dumps(
                {
                    "header": {
                        "msg_id": mid,
                        "username": "",
                        "session": uuid.uuid4().hex,
                        "msg_type": "execute_request",
                        "version": "5.3",
                    },
                    "parent_header": {},
                    "metadata": {},
                    "channel": "shell",
                    "content": {
                        "code": code,
                        "silent": False,
                        "store_history": False,
                        "user_expressions": {},
                        "allow_stdin": False,
                        "stop_on_error": True,
                    },
                }
            )
        )
        ok = True
        while True:
            m = json.loads(await asyncio.wait_for(ws.recv(), timeout))
            if m.get("parent_header", {}).get("msg_id") != mid:
                continue
            t, c = m["msg_type"], m["content"]
            if t == "stream":
                if sink is not None:
                    sink.append(c["text"])
                if not quiet:
                    sys.stdout.write(c["text"])
                    sys.stdout.flush()
            elif t in ("execute_result", "display_data") and not quiet:
                print(c["data"].get("text/plain", ""))
            elif t == "error":
                ok = False
                print("\n".join(re.sub(r"\x1b\[[0-9;]*m", "", x) for x in c["traceback"]))
            elif t == "status" and c["execution_state"] == "idle":
                return ok


def run(code, timeout=600, quiet=False, sink=None):
    """Run Python code on the board's kernel; True if it raised nothing. `sink` collects its stdout."""
    return asyncio.run(execute(kernel(), code, timeout, quiet, sink))


def mkdir(remote):
    parts = remote.strip("/").split("/")
    for i in range(1, len(parts) + 1):
        try:
            api("PUT", "contents/" + "/".join(parts[:i]), {"type": "directory"})
        except Exception:
            pass


def _file_body(raw):
    return {"type": "file", "format": "base64", "content": base64.b64encode(raw).decode()}


def put(local, remote):
    """Upload a file of any size: chunks through the contents API, joined by the kernel."""
    size = os.path.getsize(local)
    mkdir(os.path.dirname(remote))
    if size <= CHUNK:
        raw = open(local, "rb").read()
        api("PUT", "contents/" + remote, _file_body(raw))
        return size
    parts = []
    with open(local, "rb") as f:
        for i, raw in enumerate(iter(lambda: f.read(CHUNK), b"")):
            part = f"{remote}.part{i:04d}"
            api("PUT", "contents/" + part, _file_body(raw))
            parts.append(part)
            sys.stdout.write(f"\r  {local}: {min((i + 1) * CHUNK, size) >> 20} / {size >> 20} MB")
            sys.stdout.flush()
    print()
    root = api("GET", "contents/")["path"] or ""
    code = (
        "import os\n"
        f"root = os.path.join('/root/jupyter_notebooks', {root!r})\n"
        f"parts = {parts!r}\n"
        f"with open(os.path.join(root, {remote!r}), 'wb') as out:\n"
        "    for p in parts:\n"
        "        with open(os.path.join(root, p), 'rb') as f: out.write(f.read())\n"
        "        os.remove(os.path.join(root, p))\n"
        f"print('joined', {remote!r}, os.path.getsize(os.path.join(root, {remote!r})))\n"
    )
    if not run(code, timeout=600):
        sys.exit(f"joining {remote} failed")
    return size


def main():
    a = sys.argv[1:]
    login()
    if a[0] == "ls":
        d = api("GET", "contents/" + (a[1] if len(a) > 1 else ""))
        for x in d["content"]:
            print(x["type"][0], x["size"], x["path"])
    elif a[0] == "put":
        print(f"uploaded {put(a[1], a[2])} bytes -> {a[2]}")
    elif a[0] == "run":
        timeout = float(a[a.index("--timeout") + 1]) if "--timeout" in a else 600
        kid = kernel(a[a.index("--kernel") + 1] if "--kernel" in a else None)
        code = open(a[1]).read() if os.path.exists(a[1]) else a[1]
        sys.exit(0 if asyncio.run(execute(kid, code, timeout)) else 1)
    elif a[0] == "restart":
        kid = kernel()
        api("POST", f"kernels/{kid}/restart")
        print("restarted", kid)


if __name__ == "__main__":
    main()
