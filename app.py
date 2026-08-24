import os
import platform
import re
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BIN_DIR = ROOT / "bin"
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "Qwen3-1.7B-Q3_K_L.gguf"
HF_MODEL_URL = "https://huggingface.co/exebr/novelforge-qwen3-1.7b-q3/resolve/main/Qwen3-1.7B-Q3_K_L.gguf?download=true"
# 4K benchmark: env N_CTX can override this without another code change.
CONTEXT = os.getenv("N_CTX", "4096")
PORT = os.getenv("PORT") or os.getenv("SERVER_PORT") or os.getenv("APP_PORT") or os.getenv("P_SERVER_PORT") or "8080"
HOST = "0.0.0.0"
PUBLIC_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com", re.I)


def log(msg: str) -> None:
    print(f"[novelforge] {msg}", flush=True)


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    log(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "NovelForge-Writer-Server/1.0"})
    with urllib.request.urlopen(req, timeout=180) as src, open(tmp, "wb") as out:
        shutil.copyfileobj(src, out)
    tmp.replace(dest)


def detect_arch() -> str:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "x64"
    if machine in {"aarch64", "arm64"}:
        return "arm64"
    raise RuntimeError(f"Unsupported architecture: {machine}")


def find_runtime_server() -> Path | None:
    candidates = list(BIN_DIR.rglob("llama-server")) if BIN_DIR.exists() else []
    for candidate in candidates:
        if list(candidate.parent.glob("*.so*")):
            candidate.chmod(candidate.stat().st_mode | stat.S_IEXEC)
            return candidate
    return None


def ensure_llama_server() -> Path:
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    existing = find_runtime_server()
    if existing:
        log(f"llama.cpp runtime already present: {existing}")
        return existing

    stale = BIN_DIR / "llama-server"
    if stale.exists():
        stale.unlink()

    arch = detect_arch()
    release = os.getenv("LLAMA_CPP_RELEASE", "b10612")
    asset = f"llama-{release}-bin-ubuntu-{arch}.tar.gz"
    url = f"https://github.com/ggml-org/llama.cpp/releases/download/{release}/{asset}"
    archive = ROOT / asset
    runtime_dir = BIN_DIR / release
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    try:
        download(url, archive)
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(runtime_dir)
    finally:
        if archive.exists():
            archive.unlink()

    candidates = list(runtime_dir.rglob("llama-server"))
    if not candidates:
        raise RuntimeError("llama-server was not found after extracting the release asset")
    server = candidates[0]
    server.chmod(server.stat().st_mode | stat.S_IEXEC)
    log(f"llama.cpp runtime extracted: {server}")
    return server


def runtime_env(server: Path) -> dict[str, str]:
    env = os.environ.copy()
    lib_dirs = {str(server.parent)}
    for so in BIN_DIR.rglob("*.so*"):
        lib_dirs.add(str(so.parent))
    old = env.get("LD_LIBRARY_PATH", "")
    if old:
        lib_dirs.add(old)
    env["LD_LIBRARY_PATH"] = ":".join(sorted(lib_dirs))
    return env


def ensure_model() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_FILE.exists() and MODEL_FILE.stat().st_size > 500_000_000:
        log(f"model already present: {MODEL_FILE.name}")
        return
    download(HF_MODEL_URL, MODEL_FILE)


def ensure_cloudflared() -> Path:
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    binary = BIN_DIR / "cloudflared"
    if binary.exists() and binary.stat().st_size > 5_000_000:
        binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
        log(f"cloudflared already present: {binary}")
        return binary

    arch = detect_arch()
    asset_arch = "amd64" if arch == "x64" else "arm64"
    url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-{asset_arch}"
    download(url, binary)
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    log(f"cloudflared installed: {binary}")
    return binary


def wait_for_local_server(port: int, process: subprocess.Popen, timeout: int = 120) -> None:
    log(f"waiting for llama-server on 127.0.0.1:{port}")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                log("llama-server is reachable locally")
                return
        except OSError:
            time.sleep(1)
    raise RuntimeError("Timed out waiting for llama-server to become reachable")


def pipe_output(process: subprocess.Popen, prefix: str, url_event: threading.Event | None = None) -> None:
    assert process.stdout is not None
    for raw in process.stdout:
        line = raw.rstrip("\r\n")
        print(f"[{prefix}] {line}", flush=True)
        if url_event is not None:
            match = PUBLIC_URL_RE.search(line)
            if match:
                public_url = match.group(0)
                log("=" * 72)
                log(f"PUBLIC WRITER URL: {public_url}")
                log(f"OPENAI API: {public_url}/v1/chat/completions")
                log("Copy PUBLIC WRITER URL into NovelForge Writer Lab.")
                log("=" * 72)
                url_event.set()


def main() -> None:
    log(f"python={sys.version.split()[0]} arch={platform.machine()} port={PORT} ctx={CONTEXT}")
    log("benchmark profile: 1 slot / 2 threads / prompt cache enabled / 4K context")
    ensure_model()
    server = ensure_llama_server()
    cloudflared = ensure_cloudflared()

    llama_cmd = [
        str(server),
        "-m", str(MODEL_FILE),
        "-c", CONTEXT,
        "--host", HOST,
        "--port", PORT,
        "--parallel", "1",
        "--threads", os.getenv("LLAMA_THREADS", "2"),
        "--threads-batch", os.getenv("LLAMA_THREADS_BATCH", "2"),
        "--cache-reuse", os.getenv("LLAMA_CACHE_REUSE", "256"),
    ]
    log("starting llama-server: " + " ".join(llama_cmd))
    llama = subprocess.Popen(
        llama_cmd,
        env=runtime_env(server),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    threading.Thread(target=pipe_output, args=(llama, "llama"), daemon=True).start()

    tunnel = None
    try:
        wait_for_local_server(int(PORT), llama)
        tunnel_cmd = [
            str(cloudflared),
            "tunnel",
            "--no-autoupdate",
            "--url", f"http://127.0.0.1:{PORT}",
        ]
        log("starting cloudflared Quick Tunnel")
        tunnel = subprocess.Popen(
            tunnel_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        url_event = threading.Event()
        threading.Thread(target=pipe_output, args=(tunnel, "cloudflared", url_event), daemon=True).start()

        if not url_event.wait(timeout=60):
            log("WARNING: no trycloudflare.com URL detected within 60 seconds; tunnel may still be connecting")

        while True:
            llama_code = llama.poll()
            tunnel_code = tunnel.poll()
            if llama_code is not None:
                raise RuntimeError(f"llama-server stopped with code {llama_code}")
            if tunnel_code is not None:
                raise RuntimeError(f"cloudflared stopped with code {tunnel_code}")
            time.sleep(2)
    finally:
        if tunnel is not None and tunnel.poll() is None:
            tunnel.terminate()
        if llama.poll() is None:
            llama.terminate()


if __name__ == "__main__":
    main()
