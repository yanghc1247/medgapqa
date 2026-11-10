"""Smoke test of the release (CPU only; no model weights, items or trajectories needed).
1. every .py file compiles; 2. registered prompt hashes; 3. registered-hash guards (scripts/p2/run_pilot.py and the frozen manifests) match the files in this release.
python tests/test_smoke.py   (or: python -m pytest tests/test_smoke.py)
"""
import hashlib, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC, SCRIPTS = ROOT / "src" / "medgapqa", ROOT / "scripts"
for p in (SRC, SCRIPTS, SCRIPTS / "p2", SCRIPTS / "d56", SCRIPTS / "d58"):
    sys.path.insert(0, str(p))


def py_files():
    return sorted(list(SRC.rglob("*.py")) + list(SCRIPTS.rglob("*.py")))


def test_compile():
    for p in py_files():
        compile(p.read_text(encoding="utf-8"), str(p), "exec")


def test_prompts():
    import agent, pipeline_k, prompts_c4f, prompts_p6, simulator_k
    cfg = agent.load_config(allow_provisional=False)
    fc, fp = (json.load(open(ROOT / "prompts" / f)) for f in ("FROZEN_C4F_v3.json", "FROZEN_P6_v1.json"))
    nq, ns, sch = pipeline_k.SCHEMAS["Q3"]
    assert simulator_k.prompt_sha() == "68f99cd48cf56d7f2533404e1f913a7ee82d7a23025eb41d392c68bd3445e002"
    assert prompts_c4f.prompt_sha(cfg) == fc["system_prompt_sha256"]
    assert prompts_p6.sha(pipeline_k.GEN_SYSTEM) == fp["gen_prompt_sha256"]
    assert hashlib.sha256(pipeline_k.GEN_SYSTEM.format(NQ=nq, NS=ns, SCHEMA=sch).encode()).hexdigest() == fp["p1b_gen_prompt_sha256"]


def test_guards():
    sha = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
    pairs = re.findall(r'"((?:src/medgapqa|scripts|configs)/[\w./]+)": "([0-9a-f]{64})"', (SCRIPTS / "p2" / "run_pilot.py").read_text(encoding="utf-8"))
    for f in ("FROZEN_C4F_v3.json", "FROZEN_P6_v1.json"):
        pairs += list(json.load(open(ROOT / "prompts" / f))["files_sha256"].items())
    assert pairs and all(sha(p) == h for p, h in pairs), [p for p, h in pairs if sha(p) != h]


if __name__ == "__main__":
    for t in (test_compile, test_prompts, test_guards):
        t(); print("ok", t.__name__)
