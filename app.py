import json
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

# NovelForge personal-use speed profile.
# The 1.7B model remains the target/final writer. A tiny Qwen3 model can act as
# a speculative draft so the target verifies several likely tokens per batch.
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "Qwen3-1.7B-Q4_K_M.gguf")
MODEL_FILE = MODEL_DIR / Path(MODEL_FILENAME).name
HF_MODEL_URL = os.getenv(
    "HF_MODEL_URL",
    "https://huggingface.co/ggml-org/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf?download=true",
)

SPEC_DRAFT_ENABLED = os.getenv("LLAMA_SPEC_DRAFT", "1").strip().lower() not in {"0", "false", "off", "no"}
DRAFT_MODEL_FILENAME = os.getenv("DRAFT_MODEL_FILENAME", "Qwen3-0.6B-Q4_0.gguf")
DRAFT_MODEL_FILE = MODEL_DIR / Path(DRAFT_MODEL_FILENAME).name
DRAFT_HF_MODEL_URL = os.getenv(
    "DRAFT_HF_MODEL_URL",
    "https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_0.gguf?download=true",
)
SPEC_DRAFT_N_MAX = os.getenv("LLAMA_SPEC_DRAFT_N_MAX", "4")
SPEC_DRAFT_P_MIN = os.getenv("LLAMA_SPEC_DRAFT_P_MIN", "0.10")

CONTEXT = os.getenv("N_CTX", "4096")
BATCH_SIZE = os.getenv("LLAMA_BATCH", "512")
UBATCH_SIZE = os.getenv("LLAMA_UBATCH", "512")
CACHE_REUSE = os.getenv("LLAMA_CACHE_REUSE", "512")
PORT = os.getenv("PORT") or os.getenv("SERVER_PORT") or os.getenv("APP_PORT") or os.getenv("P_SERVER_PORT") or "8080"
HOST = "0.0.0.0"
PUBLIC_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com", re.I)

REGISTRY_SUPABASE_URL = os.getenv("WRITER_REGISTRY_SUPABASE_URL", "https://zlysbimnzsaovkrgckgk.supabase.co")
REGISTRY_PUBLISHABLE_KEY = os.getenv("WRITER_REGISTRY_PUBLISHABLE_KEY", "sb_publishable_XPv2geOFiYwbkUxEZyqPiQ_1vBn4jmY")
REGISTRY_TOKEN = os.getenv("WRITER_REGISTRY_TOKEN", "LRx1B_jPsdMe-LUx4f6DtMT9_Vxx8GB_0LLGalIVCtQ")

MEMORY_RESTART_MB = int(os.getenv("WATCHDOG_MEMORY_MB", "3000"))
MEMORY_REARM_MB = int(os.getenv("WATCHDOG_MEMORY_REARM_MB", "2450"))
MEMORY_EMERGENCY_MB = int(os.getenv("WATCHDOG_EMERGENCY_MB", "3250"))
MAX_UPTIME_SECONDS = int(os.getenv("WATCHDOG_MAX_UPTIME_SECONDS", str(6 * 60 * 60)))
WATCHDOG_INTERVAL_SECONDS = int(os.getenv("WATCHDOG_INTERVAL_SECONDS", "5"))
IDLE_GRACE_SECONDS = int(os.getenv("WATCHDOG_IDLE_GRACE_SECONDS", "12"))
RESTART_COOLDOWN_SECONDS = int(os.getenv("WATCHDOG_RESTART_COOLDOWN_SECONDS", "120"))
STARTUP_MEMORY_GRACE_SECONDS = int(os.getenv("WATCHDOG_STARTUP_GRACE_SECONDS", "180"))

llama_busy = threading.Event()
activity_lock = threading.Lock()
last_activity_at = time.monotonic()


def log(msg: str) -> None:
    print(f"[novelforge] {msg}", flush=True)


def mark_activity(busy: bool) -> None:
    global last_activity_at
    llama_busy.set() if busy else llama_busy.clear()
    with activity_lock:
        last_activity_at = time.monotonic()


def seconds_since_activity() -> float:
    with activity_lock:
        return time.monotonic() - last_activity_at


def _read_cgroup_value(paths: list[Path]) -> str | None:
    for path in paths:
        try:
            raw = path.read_text().strip()
            if raw:
                return raw
        except OSError:
            pass
    return None


def current_container_memory_mb() -> float | None:
    raw = _read_cgroup_value([
        Path("/sys/fs/cgroup/memory.current"),
        Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ])
    try:
        return int(raw) / (1024 * 1024) if raw and raw != "max" else None
    except ValueError:
        return None


def container_memory_limit_mb() -> float | None:
    raw = _read_cgroup_value([
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ])
    if not raw or raw == "max":
        return None
    try:
        value = int(raw)
        if value >= (1 << 60):
            return None
        return value / (1024 * 1024)
    except ValueError:
        return None


def effective_cpu_count() -> int:
    """Respect cgroup CPU quota; never blindly oversubscribe a tiny host."""
    host = max(1, os.cpu_count() or 1)
    quota_count = host

    raw = _read_cgroup_value([Path("/sys/fs/cgroup/cpu.max")])
    if raw:
        parts = raw.split()
        if len(parts) >= 2 and parts[0] != "max":
            try:
                quota, period = int(parts[0]), int(parts[1])
                if quota > 0 and period > 0:
                    quota_count = max(1, (quota + period - 1) // period)
            except ValueError:
                pass
    else:
        q = _read_cgroup_value([Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")])
        p = _read_cgroup_value([Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")])
        try:
            if q and p and int(q) > 0 and int(p) > 0:
                quota_count = max(1, (int(q) + int(p) - 1) // int(p))
        except ValueError:
            pass

    return max(1, min(host, quota_count))


def selected_threads() -> int:
    override = os.getenv("LLAMA_THREADS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    return min(4, effective_cpu_count())


def selected_batch_threads() -> int:
    override = os.getenv("LLAMA_THREADS_BATCH")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    return selected_threads()


def effective_emergency_mb() -> int:
    limit = container_memory_limit_mb()
    if limit is None:
        return MEMORY_EMERGENCY_MB
    return min(MEMORY_EMERGENCY_MB, max(512, int(limit * .90)))


def log_resources() -> None:
    current = current_container_memory_mb()
    limit = container_memory_limit_mb()
    emergency = effective_emergency_mb()
    current_text = f"{current:.0f} MB" if current is not None else "unknown"
    limit_text = f"{limit:.0f} MB ({limit / 1024:.2f} GiB)" if limit is not None else "UNLIMITED / host-managed"
    log(f"kernel cgroup memory: current={current_text}, hard_limit={limit_text}")
    if limit is not None:
        log(f"kernel headroom: {max(0, limit - (current or 0)):.0f} MB; effective emergency={emergency} MB ({emergency / limit * 100:.0f}% of hard limit)")
    log(f"cpu: host={os.cpu_count() or 1}, effective={effective_cpu_count()}, llama_threads={selected_threads()}, batch_threads={selected_batch_threads()}")
    log(f"speculative draft: {'ON' if SPEC_DRAFT_ENABLED else 'OFF'}" + (f", model={DRAFT_MODEL_FILE.name}, n_max={SPEC_DRAFT_N_MAX}, p_min={SPEC_DRAFT_P_MIN}" if SPEC_DRAFT_ENABLED else ""))


def register_public_url(public_url: str) -> None:
    try:
        endpoint = f"{REGISTRY_SUPABASE_URL.rstrip('/')}/rest/v1/rpc/register_writer_endpoint"
        body = json.dumps({"p_token": REGISTRY_TOKEN, "p_endpoint_url": public_url}).encode()
        req = urllib.request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "apikey": REGISTRY_PUBLISHABLE_KEY,
                "Authorization": f"Bearer {REGISTRY_PUBLISHABLE_KEY}",
                "User-Agent": "NovelForge-Writer-Registry/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=20) as response:
            response.read()
        log(f"Storyforge registry updated: {public_url}")
    except Exception as exc:
        log(f"WARNING: could not update Storyforge Writer registry: {exc}")


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.unlink(missing_ok=True)
    log(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "NovelForge-Writer-Server/1.0"})
    with urllib.request.urlopen(req, timeout=300) as src, open(tmp, "wb") as out:
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
    for candidate in list(BIN_DIR.rglob("llama-server")) if BIN_DIR.exists() else []:
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
    stale.unlink(missing_ok=True)
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
        archive.unlink(missing_ok=True)

    candidates = list(runtime_dir.rglob("llama-server"))
    if not candidates:
        raise RuntimeError("llama-server was not found after extracting the release asset")
    server = candidates[0]
    server.chmod(server.stat().st_mode | stat.S_IEXEC)
    return server


def runtime_env(server: Path) -> dict[str, str]:
    env = os.environ.copy()
    dirs = {str(server.parent)}
    for so in BIN_DIR.rglob("*.so*"):
        dirs.add(str(so.parent))
    if env.get("LD_LIBRARY_PATH"):
        dirs.add(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(sorted(dirs))
    return env


def ensure_model_file(path: Path, url: str, min_bytes: int, label: str) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > min_bytes:
        log(f"{label} already present: {path.name} ({path.stat().st_size / (1024 ** 3):.2f} GiB)")
        return
    log(f"selected {label}: {path.name}")
    download(url, path)
    log(f"{label} download complete: {path.name} ({path.stat().st_size / (1024 ** 3):.2f} GiB)")


def ensure_model() -> None:
    ensure_model_file(MODEL_FILE, HF_MODEL_URL, 500_000_000, "target model")
    if SPEC_DRAFT_ENABLED:
        ensure_model_file(DRAFT_MODEL_FILE, DRAFT_HF_MODEL_URL, 300_000_000, "draft model")


def ensure_cloudflared() -> Path:
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    binary = BIN_DIR / "cloudflared"
    if binary.exists() and binary.stat().st_size > 5_000_000:
        binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
        return binary
    arch = detect_arch()
    asset_arch = "amd64" if arch == "x64" else "arm64"
    download(f"https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-{asset_arch}", binary)
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
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


def pipe_output(process: subprocess.Popen, prefix: str, url_event: threading.Event | None = None, track_llama: bool = False) -> None:
    assert process.stdout is not None
    for raw in process.stdout:
        line = raw.rstrip("\r\n")
        print(f"[{prefix}] {line}", flush=True)
        if track_llama:
            lower = line.lower()
            if "launch_slot" in lower or "processing task" in lower:
                mark_activity(True)
            elif "slot release" in lower or "stop processing" in lower:
                mark_activity(False)
        if url_event is not None:
            match = PUBLIC_URL_RE.search(line)
            if match:
                public_url = match.group(0)
                log("=" * 72)
                log(f"PUBLIC WRITER URL: {public_url}")
                log(f"OPENAI API: {public_url}/v1/chat/completions")
                log("=" * 72)
                threading.Thread(target=register_public_url, args=(public_url,), daemon=True).start()
                url_event.set()


def llama_command(server: Path) -> list[str]:
    cmd = [
        str(server),
        "-m", str(MODEL_FILE),
        "-c", CONTEXT,
        "--host", HOST,
        "--port", PORT,
        "--parallel", "1",
        "--threads", str(selected_threads()),
        "--threads-batch", str(selected_batch_threads()),
        "--batch-size", BATCH_SIZE,
        "--ubatch-size", UBATCH_SIZE,
        "--cache-reuse", CACHE_REUSE,
    ]
    if SPEC_DRAFT_ENABLED:
        cmd += [
            "--spec-type", "draft-simple",
            "--spec-draft-model", str(DRAFT_MODEL_FILE),
            "--spec-draft-n-max", SPEC_DRAFT_N_MAX,
            "--spec-draft-p-min", SPEC_DRAFT_P_MIN,
            "--spec-draft-threads", str(selected_threads()),
            "--spec-draft-threads-batch", str(selected_batch_threads()),
        ]
    return cmd


def start_llama(server: Path) -> subprocess.Popen:
    mark_activity(False)
    cmd = llama_command(server)
    log("starting llama-server: " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        env=runtime_env(server),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    threading.Thread(target=pipe_output, args=(proc, "llama", None, True), daemon=True).start()
    wait_for_local_server(int(PORT), proc)
    return proc


def stop_process(process: subprocess.Popen, name: str, timeout: int = 20) -> None:
    if process.poll() is not None:
        return
    log(f"stopping {name} cleanly")
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        log(f"{name} did not stop in time; killing it")
        process.kill()
        process.wait(timeout=5)


def main() -> None:
    emergency = effective_emergency_mb()
    log(
        f"python={sys.version.split()[0]} arch={platform.machine()} port={PORT} "
        f"model={MODEL_FILE.name} ctx={CONTEXT} batch={BATCH_SIZE} ubatch={UBATCH_SIZE} cache_reuse={CACHE_REUSE}"
    )
    log_resources()
    log(
        f"watchdog: normal >= {MEMORY_RESTART_MB} MB, emergency >= {emergency} MB, "
        f"re-arm <= {MEMORY_REARM_MB} MB, startup grace {STARTUP_MEMORY_GRACE_SECONDS}s"
    )

    ensure_model()
    server = ensure_llama_server()
    cloudflared = ensure_cloudflared()
    llama = start_llama(server)
    llama_started_at = time.monotonic()
    last_restart_at = llama_started_at
    memory_armed = True
    log_resources()

    tunnel = subprocess.Popen(
        [str(cloudflared), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{PORT}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    url_event = threading.Event()
    threading.Thread(target=pipe_output, args=(tunnel, "cloudflared", url_event), daemon=True).start()
    url_event.wait(timeout=60)
    last_defer_log = 0.0

    try:
        while True:
            if tunnel.poll() is not None:
                raise RuntimeError(f"cloudflared stopped with code {tunnel.returncode}")

            if llama.poll() is not None:
                log(f"llama-server exited with code {llama.returncode}; restarting while keeping tunnel alive")
                time.sleep(2)
                llama = start_llama(server)
                llama_started_at = time.monotonic()
                last_restart_at = llama_started_at
                memory_armed = True
                log_resources()
                continue

            mem_mb = current_container_memory_mb()
            now = time.monotonic()
            uptime = now - llama_started_at
            emergency = effective_emergency_mb()

            if mem_mb is not None and mem_mb <= MEMORY_REARM_MB and not memory_armed:
                memory_armed = True
                log(f"watchdog memory re-armed at {mem_mb:.0f} MB")

            past_startup_grace = uptime >= STARTUP_MEMORY_GRACE_SECONDS
            emergency_due = past_startup_grace and mem_mb is not None and mem_mb >= emergency
            memory_due = (
                memory_armed
                and past_startup_grace
                and mem_mb is not None
                and mem_mb >= MEMORY_RESTART_MB
                and now - last_restart_at >= RESTART_COOLDOWN_SECONDS
            )
            uptime_due = uptime >= MAX_UPTIME_SECONDS and now - last_restart_at >= RESTART_COOLDOWN_SECONDS

            if emergency_due or memory_due or uptime_due:
                if emergency_due and mem_mb is not None:
                    reason = f"EMERGENCY memory {mem_mb:.0f} MB >= {emergency} MB"
                elif memory_due and mem_mb is not None:
                    reason = f"memory {mem_mb:.0f} MB >= {MEMORY_RESTART_MB} MB"
                else:
                    reason = f"uptime {uptime / 3600:.1f}h"

                idle_for = seconds_since_activity()
                if not emergency_due and (llama_busy.is_set() or idle_for < IDLE_GRACE_SECONDS):
                    if now - last_defer_log >= 20:
                        log(f"watchdog wants recycle ({reason}) but Writer is busy/recently active; deferring")
                        last_defer_log = now
                else:
                    log(
                        f"watchdog {reason}; aborting active generation to prevent container OOM"
                        if emergency_due and llama_busy.is_set()
                        else f"watchdog recycle triggered: {reason}"
                    )
                    memory_armed = False
                    stop_process(llama, "llama-server")
                    time.sleep(3)
                    llama = start_llama(server)
                    llama_started_at = time.monotonic()
                    last_restart_at = llama_started_at
                    log("watchdog recycle complete")
                    log_resources()

            time.sleep(WATCHDOG_INTERVAL_SECONDS)
    finally:
        stop_process(llama, "llama-server")
        stop_process(tunnel, "cloudflared")


if __name__ == "__main__":
    main()
