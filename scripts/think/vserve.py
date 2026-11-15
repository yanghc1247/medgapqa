"""D130: start a vLLM server (same arguments as think_run.Server + optional VLLM_BATCH_INVARIANT, max_model_len and tensor-parallel size). Started with setsid; the PID is written to a file.
python scripts/think/vserve.py --model qwen3-14b --gpus 7 --port 8851 --max-len 40960 --batch-invariant --log runs/p2/d130_checks/srv_q14.log
"""
import argparse, os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/medgapqa"), str(ROOT / "scripts"), str(ROOT / "scripts" / "p2"), str(ROOT / "scripts" / "think")]
import p2_models as P2M  # noqa: E402
import think_run as TR  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--gpus", required=True); ap.add_argument("--port", type=int, required=True)
ap.add_argument("--max-len", type=int, required=True); ap.add_argument("--batch-invariant", action="store_true"); ap.add_argument("--log", required=True)
ap.add_argument("--mem", default="0.90"); ap.add_argument("--extra", default="")
a = ap.parse_args()
s = P2M.SPECS.get(a.model) or {"medgemma-4b": {"repo": "google/medgemma-4b-it", "revision": "290cda5eeccbee130f987c4ad74a59ae6f196408"}}[a.model]
path = s["path"] if "path" in s else str(Path(P2M.CACHE) / f"models--{s['repo'].replace('/', '--')}" / "snapshots" / s["revision"])
tp = len(a.gpus.split(","))
env = {**os.environ, "CUDA_VISIBLE_DEVICES": a.gpus, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_CACHE_ROOT": "$PROJECT_ROOT/.cache/vllm",
       "TMPDIR": "$PROJECT_ROOT/.cache/tmp_vllm", "HF_HOME": "$PROJECT_ROOT/.cache/hf_vllm", "XDG_CACHE_HOME": "$PROJECT_ROOT/.cache",
       "VLLM_BATCH_INVARIANT": "1" if a.batch_invariant else "0"}
cmd = [str(TR.VENV / "bin" / "vllm"), "serve", path, "--served-model-name", "m", "--port", str(a.port), "--dtype", "bfloat16", "--max-model-len", str(a.max_len),
       "--gpu-memory-utilization", a.mem, "--seed", "0", "--generation-config", "vllm", "--logprobs-mode", "processed_logprobs", "--max-logprobs", "20",
       "--tensor-parallel-size", str(tp)] + (a.extra.split() if a.extra else [])
p = subprocess.Popen(cmd, env=env, stdout=open(a.log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
Path(a.log + ".pid").write_text(str(p.pid)); print(p.pid, " ".join(cmd))
