import os
import platform
import shutil
import stat
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BIN_DIR = ROOT / "bin"
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "Qwen3-1.7B-Q3_K_L.gguf"
HF_MODEL_URL = "https://huggingface.co/exebr/novelforge-qwen3-1.7b-q3/resolve/main/Qwen3-1.7B-Q3_K_L.gguf?download=true"
CONTEXT = os.getenv("N_CTX", "1024")
PORT = os.getenv("PORT") or os.getenv("SERVER_PORT") or os.getenv("APP_PORT") or os.getenv("P_SERVER_PORT") or "8080"
HOST = "0.0.0.0"


def log(msg: str) -> None:
    print(f"[novelforge] {msg}", flush=True)


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    log(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as src, open(tmp, "wb") as out:
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
    # Prefer a llama-server that still lives beside the .so files shipped in
    # the official archive. A previous bootstrap copied only the executable
    # to bin/, which breaks dynamic linking on newer llama.cpp builds.
    candidates = list(BIN_DIR.rglob("llama-server")) if BIN_DIR.exists() else []
    for candidate in candidates:
        if candidate.parent != BIN_DIR and list(candidate.parent.glob("*.so*")):
            candidate.chmod(candidate.stat().st_mode | stat.S_IEXEC)
            return candidate
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

    # Remove the stale standalone executable created by the old bootstrap.
    stale = BIN_DIR / "llama-server"
    if stale.exists():
        log("removing stale standalone llama-server (shared libraries missing)")
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
    log(f"LD_LIBRARY_PATH={env['LD_LIBRARY_PATH']}")
    return env


def ensure_model() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_FILE.exists() and MODEL_FILE.stat().st_size > 500_000_000:
        log(f"model already present: {MODEL_FILE.name}")
        return
    download(HF_MODEL_URL, MODEL_FILE)


def main() -> None:
    log(f"python={sys.version.split()[0]} arch={platform.machine()} port={PORT} ctx={CONTEXT}")
    ensure_model()
    server = ensure_llama_server()
    cmd = [
        str(server),
        "-m", str(MODEL_FILE),
        "-c", CONTEXT,
        "--host", HOST,
        "--port", PORT,
        "--parallel", "1",
        "--threads", os.getenv("LLAMA_THREADS", "2"),
        "--threads-batch", os.getenv("LLAMA_THREADS_BATCH", "2"),
    ]
    log("starting llama-server: " + " ".join(cmd))
    os.execve(str(server), cmd, runtime_env(server))


if __name__ == "__main__":
    main()
