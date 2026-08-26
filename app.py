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

# NovelForge personal-use speed/quality tournament profile.
# Candidate D: Ektome Qwen2.5 1.5B PristinelyUncensored, Q4_K_M (~986 MB).
# Distinct uncensored fine-tune with corrected GGUF chat template; chosen before Phi due host speed budget.
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "Ektome-Qwen2.5-1.5Bi-Q4_K_M.gguf")
MODEL_FILE = MODEL_DIR / Path(MODEL_FILENAME).name
HF_MODEL_URL = os.getenv("HF_MODEL_URL", "https://huggingface.co/Zynerji/Ektome-Qwen2.5-1.5Bi-PristinelyUncensored/resolve/main/Ektome-Qwen2.5-1.5Bi-Q4_K_M.gguf?download=true")

SPEC_DRAFT_ENABLED = os.getenv("LLAMA_SPEC_DRAFT", "0").strip().lower() not in {"0", "false", "off", "no"}
DRAFT_MODEL_FILENAME = os.getenv("DRAFT_MODEL_FILENAME", "Qwen3-0.6B-Q4_0.gguf")
DRAFT_MODEL_FILE = MODEL_DIR / Path(DRAFT_MODEL_FILENAME).name
DRAFT_HF_MODEL_URL = os.getenv("DRAFT_HF_MODEL_URL", "https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_0.gguf?download=true")
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

llama_busy = threading.Event()
activity_lock = threading.Lock()
last_activity_at = time.monotonic()

def log(msg): print(f"[novelforge] {msg}", flush=True)
def mark_activity(busy):
    global last_activity_at
    llama_busy.set() if busy else llama_busy.clear()
    with activity_lock: last_activity_at = time.monotonic()
def seconds_since_activity():
    with activity_lock: return time.monotonic() - last_activity_at

def _read_cgroup_value(paths):
    for path in paths:
        try:
            raw = path.read_text().strip()
            if raw: return raw
        except OSError: pass
    return None

def current_container_memory_mb():
    raw = _read_cgroup_value([Path("/sys/fs/cgroup/memory.current"), Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")])
    try: return int(raw)/(1024*1024) if raw and raw != "max" else None
    except ValueError: return None

def container_memory_limit_mb():
    raw = _read_cgroup_value([Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")])
    if not raw or raw == "max": return None
    try:
        value=int(raw); return None if value >= (1<<60) else value/(1024*1024)
    except ValueError: return None

def effective_cpu_count():
    host=max(1,os.cpu_count() or 1); quota_count=host
    raw=_read_cgroup_value([Path("/sys/fs/cgroup/cpu.max")])
    if raw:
        parts=raw.split()
        if len(parts)>=2 and parts[0] != "max":
            try:
                q,p=int(parts[0]),int(parts[1]); quota_count=max(1,(q+p-1)//p) if q>0 and p>0 else quota_count
            except ValueError: pass
    return max(1,min(host,quota_count))
def selected_threads():
    try: return max(1,int(os.getenv("LLAMA_THREADS"))) if os.getenv("LLAMA_THREADS") else min(4,effective_cpu_count())
    except ValueError: return min(4,effective_cpu_count())
def selected_batch_threads():
    try: return max(1,int(os.getenv("LLAMA_THREADS_BATCH"))) if os.getenv("LLAMA_THREADS_BATCH") else selected_threads()
    except ValueError: return selected_threads()
def effective_emergency_mb():
    limit=container_memory_limit_mb(); return MEMORY_EMERGENCY_MB if limit is None else min(MEMORY_EMERGENCY_MB,max(512,int(limit*.90)))
def log_resources():
    current=current_container_memory_mb(); limit=container_memory_limit_mb(); emergency=effective_emergency_mb()
    log(f"kernel cgroup memory: current={current:.0f} MB" if current is not None else "kernel cgroup memory: current=unknown")
    log(f"cpu: host={os.cpu_count() or 1}, effective={effective_cpu_count()}, llama_threads={selected_threads()}, batch_threads={selected_batch_threads()}")
    log(f"speculative draft: {'ON' if SPEC_DRAFT_ENABLED else 'OFF'}" + (f", model={DRAFT_MODEL_FILE.name}, n_max={SPEC_DRAFT_N_MAX}" if SPEC_DRAFT_ENABLED else ""))

def register_public_url(public_url):
    try:
        endpoint=f"{REGISTRY_SUPABASE_URL.rstrip('/')}/rest/v1/rpc/register_writer_endpoint"
        body=json.dumps({"p_token":REGISTRY_TOKEN,"p_endpoint_url":public_url}).encode()
        req=urllib.request.Request(endpoint,data=body,method="POST",headers={"Content-Type":"application/json","apikey":REGISTRY_PUBLISHABLE_KEY,"Authorization":f"Bearer {REGISTRY_PUBLISHABLE_KEY}","User-Agent":"NovelForge-Writer-Registry/1.0"})
        with urllib.request.urlopen(req,timeout=20) as response: response.read()
        log(f"Storyforge registry updated: {public_url}")
    except Exception as exc: log(f"WARNING: could not update Storyforge Writer registry: {exc}")
def download(url,dest):
    dest.parent.mkdir(parents=True,exist_ok=True); tmp=dest.with_suffix(dest.suffix+".part"); tmp.unlink(missing_ok=True); log(f"downloading {url}")
    req=urllib.request.Request(url,headers={"User-Agent":"NovelForge-Writer-Server/1.0"})
    with urllib.request.urlopen(req,timeout=300) as src, open(tmp,"wb") as out: shutil.copyfileobj(src,out)
    tmp.replace(dest)
def detect_arch():
    m=platform.machine().lower()
    if m in {"x86_64","amd64"}: return "x64"
    if m in {"aarch64","arm64"}: return "arm64"
    raise RuntimeError(f"Unsupported architecture: {m}")
def find_runtime_server():
    for c in list(BIN_DIR.rglob("llama-server")) if BIN_DIR.exists() else []:
        if list(c.parent.glob("*.so*")): c.chmod(c.stat().st_mode|stat.S_IEXEC); return c
    return None
def ensure_llama_server():
    BIN_DIR.mkdir(parents=True,exist_ok=True); existing=find_runtime_server()
    if existing: log(f"llama.cpp runtime already present: {existing}"); return existing
    arch=detect_arch(); release=os.getenv("LLAMA_CPP_RELEASE","b10612"); asset=f"llama-{release}-bin-ubuntu-{arch}.tar.gz"; url=f"https://github.com/ggml-org/llama.cpp/releases/download/{release}/{asset}"; archive=ROOT/asset; runtime_dir=BIN_DIR/release
    if runtime_dir.exists(): shutil.rmtree(runtime_dir)
    runtime_dir.mkdir(parents=True,exist_ok=True)
    try:
        download(url,archive)
        with tarfile.open(archive,"r:gz") as tf: tf.extractall(runtime_dir)
    finally: archive.unlink(missing_ok=True)
    candidates=list(runtime_dir.rglob("llama-server"))
    if not candidates: raise RuntimeError("llama-server was not found after extracting release")
    server=candidates[0]; server.chmod(server.stat().st_mode|stat.S_IEXEC); return server
def runtime_env(server):
    env=os.environ.copy(); dirs={str(server.parent)}
    for so in BIN_DIR.rglob("*.so*"): dirs.add(str(so.parent))
    if env.get("LD_LIBRARY_PATH"): dirs.add(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"]=":".join(sorted(dirs)); return env
def ensure_model_file(path,url,min_bytes,label):
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size>min_bytes: log(f"{label} already present: {path.name} ({path.stat().st_size/(1024**3):.2f} GiB)"); return
    log(f"selected {label}: {path.name}"); download(url,path); log(f"{label} download complete: {path.name}")
def ensure_model():
    ensure_model_file(MODEL_FILE,HF_MODEL_URL,500_000_000,"target model")
    if SPEC_DRAFT_ENABLED: ensure_model_file(DRAFT_MODEL_FILE,DRAFT_HF_MODEL_URL,300_000_000,"draft model")
def ensure_cloudflared():
    BIN_DIR.mkdir(parents=True,exist_ok=True); binary=BIN_DIR/"cloudflared"
    if binary.exists() and binary.stat().st_size>5_000_000: binary.chmod(binary.stat().st_mode|stat.S_IEXEC); return binary
    aa="amd64" if detect_arch()=="x64" else "arm64"; download(f"https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-{aa}",binary); binary.chmod(binary.stat().st_mode|stat.S_IEXEC); return binary
def wait_for_local_server(port,process,timeout=120):
    deadline=time.time()+timeout; log(f"waiting for llama-server on 127.0.0.1:{port}")
    while time.time()<deadline:
        if process.poll() is not None: raise RuntimeError(f"llama-server exited early with code {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1",port),timeout=1): log("llama-server is reachable locally"); return
        except OSError: time.sleep(1)
    raise RuntimeError("Timed out waiting for llama-server")
def pipe_output(process,prefix,url_event=None,track_llama=False):
    assert process.stdout is not None
    for raw in process.stdout:
        line=raw.rstrip("\r\n"); print(f"[{prefix}] {line}",flush=True)
        if track_llama:
            low=line.lower()
            if "launch_slot" in low or "processing task" in low: mark_activity(True)
            elif "slot release" in low or "stop processing" in low: mark_activity(False)
        if url_event is not None:
            match=PUBLIC_URL_RE.search(line)
            if match:
                public_url=match.group(0); log(f"PUBLIC WRITER URL: {public_url}"); threading.Thread(target=register_public_url,args=(public_url,),daemon=True).start(); url_event.set()
def llama_command(server):
    cmd=[str(server),"-m",str(MODEL_FILE),"-c",CONTEXT,"--host",HOST,"--port",PORT,"--parallel","1","--threads",str(selected_threads()),"--threads-batch",str(selected_batch_threads()),"--batch-size",BATCH_SIZE,"--ubatch-size",UBATCH_SIZE,"--cache-reuse",CACHE_REUSE]
    if SPEC_DRAFT_ENABLED: cmd += ["--spec-type","draft-simple","--spec-draft-model",str(DRAFT_MODEL_FILE),"--spec-draft-n-max",SPEC_DRAFT_N_MAX,"--spec-draft-p-min",SPEC_DRAFT_P_MIN]
    return cmd
def start_llama(server):
    cmd=llama_command(server); log("starting llama-server: "+" ".join(cmd)); return subprocess.Popen(cmd,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=runtime_env(server))
def start_cloudflared(binary,port):
    cmd=[str(binary),"tunnel","--no-autoupdate","--url",f"http://127.0.0.1:{port}"]; log("starting cloudflared quick tunnel"); p=subprocess.Popen(cmd,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1); e=threading.Event(); threading.Thread(target=pipe_output,args=(p,"cloudflared",e,False),daemon=True).start(); return p,e
def watchdog(process):
    over=None; emergency=effective_emergency_mb()
    while process.poll() is None:
        time.sleep(WATCHDOG_INTERVAL_SECONDS); memory=current_container_memory_mb(); now=time.monotonic()
        if memory is not None and memory>=emergency:
            if over is None: over=now; log(f"WARNING: memory high ({memory:.0f} MB >= {emergency} MB)")
            if not llama_busy.is_set() and seconds_since_activity()>=IDLE_GRACE_SECONDS and now-over>=10: log("watchdog requesting restart: memory pressure while idle"); process.terminate(); return
        elif memory is not None and memory<=MEMORY_REARM_MB: over=None
        if now-last_activity_at>MAX_UPTIME_SECONDS and not llama_busy.is_set(): log("watchdog requesting restart: max uptime"); process.terminate(); return
def main():
    log("NovelForge Writer bootstrap starting"); log_resources(); ensure_model(); server=ensure_llama_server(); cloudflared=ensure_cloudflared(); port=int(PORT)
    while True:
        llama=start_llama(server); threading.Thread(target=pipe_output,args=(llama,"llama",None,True),daemon=True).start()
        try:
            wait_for_local_server(port,llama); tunnel,_=start_cloudflared(cloudflared,port); threading.Thread(target=watchdog,args=(llama,),daemon=True).start(); code=llama.wait(); log(f"llama-server exited with code {code}")
            if tunnel.poll() is None: tunnel.terminate()
        except KeyboardInterrupt:
            if llama.poll() is None: llama.terminate()
            raise
        except Exception as exc:
            log(f"runtime error: {exc}")
            if llama.poll() is None: llama.terminate()
        log(f"restarting runtime after {RESTART_COOLDOWN_SECONDS}s cooldown"); time.sleep(RESTART_COOLDOWN_SECONDS)
if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: log("shutdown requested")
