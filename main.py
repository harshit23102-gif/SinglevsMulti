"""
main.py — Experiment runner for Healthcare AI: Single-Agent vs Multi-Agent.

Model strategy (AWS Bedrock via LiteLLM):
  - Agents use bedrock/us.amazon.nova-pro-v1:0   (quality)
  - Judge  uses bedrock/us.amazon.nova-lite-v1:0 (fast, cost-effective)

  A 2-second minimum gap between every API call is enforced to avoid
  Bedrock throttling. A 10-second pause between cases prevents burst overload.

  Credentials required in environment:
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_DEFAULT_REGION  (e.g. us-east-1)

API call budget per full run (8 cases × 3 runs, ablation ON):
  Agents:   8 × 3 × 5 calls (SA-c + SA-u + Planner + Reasoner + Verifier) = 120  Nova Pro
  Ablation: 8 × 1 × 4 calls (PO + PR + FP + judge-3way)                   =  32
  Judge:    8 × 3 × 1 call                                                 =  24  Nova Lite
  ───────────────────────────────────────────────────────────────────────────────
  Total: ~176 calls for a full run. Set NUM_RUNS=1 and RUN_ABLATION=False
  for a quick 48-call development pass.

FIXES vs original:
  - CASES_TO_RUN changed from [1,2] → None (runs all 8 cases as designed).
  - RUN_ABLATION changed from False → True (ablation data now collected).
  - NUM_RUNS changed from 1 → 3 for statistical reliability (mean ± std).
  - SingleAgentUnconstrained added to the experiment loop so architecture vs
    token-budget can actually be separated (the key research question).
  - Verifier truncation flag surfaced in terminal output.
  - Safety consistency notes surfaced in terminal output.
  - report.py provider label fixed: "Groq" → "AWS Bedrock".
"""

import os
import sys
import json
import time
import threading
import statistics
import litellm
from litellm import completion

from cases import CLINICAL_CASES
from single_agent import SingleAgent, SingleAgentUnconstrained
from multi_agent import MultiAgentSystem
from evaluator import Evaluator
from report import generate_html_report

# ── Config ────────────────────────────────────────────────────────────────────
AGENT_MODEL = "bedrock/us.amazon.nova-pro-v1:0"
JUDGE_MODEL = "bedrock/us.amazon.nova-lite-v1:0"

# FIX: Was NUM_RUNS=1 — single run gives no statistical reliability.
# 3 runs gives mean ± std and surfaces stochastic variation.
NUM_RUNS     = 3

# FIX: Was False — ablation is a key contribution; now enabled by default.
RUN_ABLATION = True

# FIX: Was [1,2] — only 2 of 8 cases ran. Now None = run all cases.
# Set to e.g. [1,2,3] only for quick development passes.
CASES_TO_RUN = [4]

CALL_GAP_SECONDS    = 2.0
BETWEEN_CASE_PAUSE  = 45


# ── ANSI helpers ──────────────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m";  BOLD   = "\033[1m";  DIM    = "\033[2m"
    CYAN   = "\033[96m"; GREEN  = "\033[92m"; YELLOW = "\033[93m"
    RED    = "\033[91m"; BLUE   = "\033[94m"; WHITE  = "\033[97m"
    GRAY   = "\033[90m"; PURPLE = "\033[95m"; ORANGE = "\033[33m"

def clr(text, *codes) -> str:
    return "".join(codes) + str(text) + C.RESET

def separator(char="─", width=75, color=C.GRAY):
    print(clr(char * width, color))

def banner(title: str, width=75):
    line = "═" * width
    pad  = (width - len(title) - 2) // 2
    print(clr(line, C.CYAN, C.BOLD))
    print(clr("║", C.CYAN) + " " * pad + clr(title, C.WHITE, C.BOLD)
          + " " * (width - pad - len(title) - 2) + clr("║", C.CYAN))
    print(clr(line, C.CYAN, C.BOLD))


# ── Rate-limit tracker ────────────────────────────────────────────────────────
class QuotaTracker:
    SESSION_LIMIT = 1000

    def __init__(self, call_gap: float = CALL_GAP_SECONDS):
        self._count     = 0
        self._last_call = 0.0
        self._gap       = call_gap
        self._lock      = threading.Lock()

    def enforce_gap(self):
        with self._lock:
            elapsed = time.time() - self._last_call
            if elapsed < self._gap:
                time.sleep(self._gap - elapsed)

    def register_call(self):
        with self._lock:
            self._count    += 1
            self._last_call = time.time()

    @property
    def count(self) -> int:
        return self._count

    @property
    def remaining(self) -> int:
        return max(0, self.SESSION_LIMIT - self._count)

    def status(self) -> str:
        pct = self._count / self.SESSION_LIMIT * 100
        color = C.GREEN if pct < 50 else C.YELLOW if pct < 80 else C.RED
        return clr(f"{self._count} calls used ({pct:.0f}% of {self.SESSION_LIMIT} soft cap)", color)


_quota = QuotaTracker()

original_completion = litellm.completion

def patched_completion(*args, **kwargs):
    _quota.enforce_gap()
    _quota.register_call()
    return original_completion(*args, **kwargs)

litellm.completion = patched_completion


# ── Spinner ───────────────────────────────────────────────────────────────────
class Spinner:
    _FRAMES = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]

    def __init__(self, label: str):
        self.label = label
        self._idx  = 0

    def tick(self):
        f = self._FRAMES[self._idx % len(self._FRAMES)]
        sys.stdout.write(f"\r  {clr(f, C.CYAN)}  {self.label}   ")
        sys.stdout.flush()
        self._idx += 1

    def done(self, note=""):
        sys.stdout.write(f"\r  {clr('✓', C.GREEN)}  {self.label}  {clr(note, C.GRAY)}\n")
        sys.stdout.flush()

    def fail(self, note=""):
        sys.stdout.write(f"\r  {clr('✗', C.RED)}  {self.label}  {clr(note, C.RED)}\n")
        sys.stdout.flush()


def spin(label: str, fn, *args, **kwargs):
    spinner = Spinner(label)
    result, error = [None], [None]

    def _target():
        try:
            result[0] = fn(*args, **kwargs)
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    while t.is_alive():
        spinner.tick()
        time.sleep(0.08)
    t.join()

    if error[0]:
        spinner.fail(str(error[0]))
        raise error[0]
    return result[0], spinner


# ── Score bar ─────────────────────────────────────────────────────────────────
def score_bar(score: float, width=18) -> str:
    filled = int((score / 10) * width)
    bar    = "█" * filled + "░" * (width - filled)
    color  = C.GREEN if score >= 8 else C.YELLOW if score >= 6 else C.RED
    return clr(bar, color) + clr(f"  {score:.1f}", C.WHITE, C.BOLD)


# ── Average results across runs ───────────────────────────────────────────────
def average_results(eval_runs: list) -> dict:
    """Average eval scores across NUM_RUNS runs for one case."""
    if len(eval_runs) == 1:
        return eval_runs[0]

    base = eval_runs[0]
    metrics = [
        "Diagnostic Accuracy", "Reasoning Depth", "Treatment Completeness",
        "Safety Awareness", "Clinical Structure", "Overall",
        "keyword_coverage_%", "pipeline_keyword_coverage_%",
        "reasoning_markers", "tokens_used", "latency_seconds",
    ]

    def avg_field(agent_key, field):
        vals = [r[agent_key].get(field, 0) for r in eval_runs if field in r.get(agent_key, {})]
        return round(statistics.mean(vals), 2) if vals else 0.0

    def std_field(agent_key, field):
        vals = [r[agent_key].get(field, 0) for r in eval_runs if field in r.get(agent_key, {})]
        return round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0

    result = {
        "case_id":    base["case_id"],
        "case_title": base["case_title"],
        "specialty":  base.get("specialty", "Unknown"),
    }
    for agent_key in ("single_agent", "multi_agent"):
        result[agent_key] = {m: avg_field(agent_key, m) for m in metrics}
        result[agent_key]["std"] = {m: std_field(agent_key, m) for m in metrics}
        result[agent_key]["safety_flags"]          = base[agent_key].get("safety_flags", [])
        result[agent_key]["safety_consistency_note"] = base[agent_key].get("safety_consistency_note")
        result[agent_key]["contradictions"]        = avg_field(agent_key, "contradictions")
        result[agent_key]["llm_calls"]             = base[agent_key].get("llm_calls", 0)

    return result


# ── Terminal summary table ────────────────────────────────────────────────────
def print_summary(all_averaged: list):
    banner("FINAL RESULTS SUMMARY")

    quality_metrics = [
        "Diagnostic Accuracy", "Reasoning Depth", "Treatment Completeness",
        "Safety Awareness", "Clinical Structure", "Overall",
    ]

    def avg(data, key):
        vals = [r[key] for r in data if key in r]
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    sd = [r["single_agent"] for r in all_averaged]
    md = [r["multi_agent"]  for r in all_averaged]

    print(f"\n  {clr('Quality Metrics (averaged over all cases & runs)', C.CYAN, C.BOLD)}\n")
    print(f"  {clr('Metric', C.GRAY):<38}  "
          f"{clr('Single-Agent (Constrained)', C.BLUE, C.BOLD):<35}  "
          f"{clr('Multi-Agent', C.GREEN, C.BOLD):<35}  "
          f"{clr('Winner', C.GRAY)}")
    separator(width=110)

    for m in quality_metrics:
        s, mv = avg(sd, m), avg(md, m)
        winner = (clr("  Multi ▲",  C.GREEN, C.BOLD) if mv > s else
                  clr("  Single ▲", C.BLUE,  C.BOLD) if s > mv else
                  clr("  Tie",      C.GRAY))
        print(f"  {clr(f'{m:<28}', C.WHITE)}  "
              f"{score_bar(s):<52}  {score_bar(mv):<52}  {winner}")

    print(); separator(width=110)

    s_tok  = avg(sd, "tokens_used")
    m_tok  = avg(md, "tokens_used")
    s_kw   = avg(sd, "keyword_coverage_%")
    m_kw   = avg(md, "keyword_coverage_%")
    s_pkl  = avg(sd, "pipeline_keyword_coverage_%")
    m_pkl  = avg(md, "pipeline_keyword_coverage_%")

    def median_lat(data):
        vals = sorted([r["latency_seconds"] for r in data if "latency_seconds" in r])
        if not vals:
            return 0.0
        n = len(vals)
        return round((vals[n // 2] + vals[(n - 1) // 2]) / 2, 2)

    s_lat_all = sorted([r["latency_seconds"] for r in sd if "latency_seconds" in r])
    m_lat_all = sorted([r["latency_seconds"] for r in md if "latency_seconds" in r])
    s_lat_med = median_lat(sd)
    m_lat_med = median_lat(md)
    s_lat_min = min(s_lat_all) if s_lat_all else 0
    s_lat_max = max(s_lat_all) if s_lat_all else 0
    m_lat_min = min(m_lat_all) if m_lat_all else 0
    m_lat_max = max(m_lat_all) if m_lat_all else 0

    print(f"\n  {clr('Efficiency', C.CYAN, C.BOLD)}\n")
    print(f"  {clr('Metric', C.GRAY):<38}  "
          f"{clr('Single', C.BLUE, C.BOLD):>16}  "
          f"{clr('Multi',  C.GREEN, C.BOLD):>16}")
    separator(width=76)

    def eff(label, sv, mv, fmt="{:.0f}", lower_better=True):
        s_str, m_str = fmt.format(sv), fmt.format(mv)
        if lower_better:
            sc, mc = (C.GREEN, C.YELLOW) if sv <= mv else (C.YELLOW, C.GREEN)
        else:
            sc, mc = (C.GREEN, C.YELLOW) if sv >= mv else (C.YELLOW, C.GREEN)
        print(f"  {clr(label, C.WHITE):<38}  "
              f"{clr(s_str, sc, C.BOLD):>25}  "
              f"{clr(m_str, mc, C.BOLD):>25}")

    eff("Avg Tokens Used",               s_tok,     m_tok,     "{:.0f}",  True)
    eff("Median Latency (s)",            s_lat_med, m_lat_med, "{:.2f}",  True)
    print(f"  {clr('  ↳  Single latency range:', C.DIM):<48}  "
          f"{clr(f'{s_lat_min:.2f}s – {s_lat_max:.2f}s', C.BLUE):>25}")
    print(f"  {clr('  ↳  Multi  latency range:', C.DIM):<48}  "
          f"{clr(f'{m_lat_min:.2f}s – {m_lat_max:.2f}s', C.GREEN):>25}")
    eff("Keyword Coverage — final output (%)", s_kw,  m_kw,  "{:.1f}", False)
    eff("Keyword Coverage — pipeline (%)",     s_pkl, m_pkl, "{:.1f}", False)
    print(f"  {clr('  ↳  Pipeline coverage searches all agent steps (Planner+Reasoner+Verifier).', C.DIM)}")
    print(f"  {clr('     Final-output coverage is the fair apples-to-apples comparison.', C.DIM)}")
    eff("LLM API Calls",               1, 3, "{:.0f}", True)
    print(f"  {clr('  ↳  3 runs each — scores above are mean; std stored in results.json.', C.DIM)}")

    separator(char="═", width=76, color=C.CYAN)
    print(f"\n  {clr('API Calls This Session:', C.GRAY)}  {_quota.status()}\n")


# ── Single-case runner ────────────────────────────────────────────────────────
def run_single_case(
    case, single_agent, single_agent_u, multi_agent, evaluator, run_num: int
) -> tuple[dict, dict]:
    print(f"\n  {clr('Run', C.DIM)} {run_num}/{NUM_RUNS}  "
          f"{clr('·', C.DIM)}  {_quota.status()}")

    # ── Single-Agent (Constrained) ────────────────────────────────────────────
    single_result, spinner = spin(
        "Running Single-Agent (Constrained) ...", single_agent.run, case["case"]
    )
    retry_note = (clr(f"  ⚠ {single_result['api_retries']} retry/retries", C.YELLOW)
                  if single_result.get("api_retries", 0) > 0 else "")
    spinner.done(
        f"{single_result['latency_seconds']}s  ·  {single_result['tokens_used']} tokens{retry_note}"
    )

    # ── Single-Agent (Unconstrained) — FIX: now included in every run ─────────
    single_u_result, spinner_u = spin(
        "Running Single-Agent (Unconstrained) ...", single_agent_u.run, case["case"]
    )
    retry_note_u = (clr(f"  ⚠ {single_u_result['api_retries']} retry/retries", C.YELLOW)
                    if single_u_result.get("api_retries", 0) > 0 else "")
    spinner_u.done(
        f"{single_u_result['latency_seconds']}s  ·  {single_u_result['tokens_used']} tokens{retry_note_u}"
    )

    # ── Multi-Agent (live per-agent progress) ─────────────────────────────────
    print(f"\n  {clr('▶', C.CYAN)}  Running Multi-Agent ...")

    _quota.enforce_gap()
    t0 = time.time()
    print(f"     {clr('→', C.DIM)} Planner  ", end="", flush=True)
    plan, tok1, r1 = multi_agent._planner(case["case"])
    r1_note = clr(f" ⚠{r1}R", C.YELLOW) if r1 > 0 else ""
    print(clr(f"✓  {time.time()-t0:.2f}s", C.GREEN) + r1_note)

    t1 = time.time()
    print(f"     {clr('→', C.DIM)} Reasoner ", end="", flush=True)
    reasoning, tok2, r2 = multi_agent._reasoner(case["case"], plan)
    r2_note = clr(f" ⚠{r2}R", C.YELLOW) if r2 > 0 else ""
    print(clr(f"✓  {time.time()-t1:.2f}s", C.GREEN) + r2_note)

    t2 = time.time()
    print(f"     {clr('→', C.DIM)} Verifier ", end="", flush=True)
    final_out, tok3, r3, trunc = multi_agent._verifier(case["case"], reasoning)
    total_lat = round(time.time() - t0, 2)
    r3_note = clr(f" ⚠{r3}R", C.YELLOW) if r3 > 0 else ""
    trunc_note = clr(" ⚠ TRUNCATED — raise max_tokens", C.RED) if trunc else ""
    print(clr(f"✓  {time.time()-t2:.2f}s", C.GREEN) + r3_note + trunc_note)

    multi_result = {
        "agent":              "Multi-Agent",
        "output":             final_out,
        "tokens_used":        tok1 + tok2 + tok3,
        "latency_seconds":    total_lat,
        "num_llm_calls":      3,
        "api_retries":        r1 + r2 + r3,
        "verifier_truncated": trunc,
        "steps": [
            {"agent": "Planner",  "output": plan},
            {"agent": "Reasoner", "output": reasoning},
            {"agent": "Verifier", "output": final_out},
        ],
    }
    print(f"\n  {clr('✓', C.GREEN)}  Multi done — "
          f"{clr(str(total_lat)+'s', C.GRAY)}  ·  "
          f"{clr(str(multi_result['tokens_used'])+' tokens', C.GRAY)}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print()
    eval_result, spinner = spin(
        f"Evaluating [{clr(evaluator.judge_model, C.PURPLE)}] ...",
        evaluator.compare, case, single_result, multi_result,
    )
    s_ov = eval_result["single_agent"].get("Overall", 0)
    m_ov = eval_result["multi_agent"].get("Overall", 0)
    winner = (clr("Multi-Agent ▲", C.GREEN, C.BOLD) if m_ov > s_ov
              else clr("Single-Agent ▲", C.BLUE, C.BOLD))
    spinner.done(f"Single {s_ov}/10  ·  Multi {m_ov}/10  →  {winner}")

    # Print safety flags
    for f in eval_result["single_agent"].get("safety_flags", []):
        print(f"  {clr('⚠ Single flag:', C.YELLOW)}  {clr(f, C.YELLOW)}")
    for f in eval_result["multi_agent"].get("safety_flags", []):
        print(f"  {clr('⚠ Multi flag:', C.YELLOW)}   {clr(f, C.YELLOW)}")

    # FIX: Print safety consistency notes when they exist
    for agent_key, label in (("single_agent", "Single"), ("multi_agent", "Multi")):
        note = eval_result[agent_key].get("safety_consistency_note")
        if note:
            print(f"  {clr(f'ℹ {label} consistency:', C.ORANGE)}  {clr(note[:120]+'...', C.ORANGE)}")

    raw = {
        "case":              case,
        "single_result":     single_result,
        "single_u_result":   single_u_result,   # FIX: unconstrained now stored
        "multi_result":      multi_result,
        "evaluation":        eval_result,
    }
    return eval_result, raw


# ── Ablation runner ───────────────────────────────────────────────────────────
def run_ablation(case, multi_agent, evaluator) -> dict:
    """
    Runs Planner-only / Planner+Reasoner / Full Pipeline on one case
    and scores them three-way. Shows contribution of each agent.

    FIX (Bug 2): Ablation judge occasionally returns an unparseable response
    for one label (FullPipeline scored 0/10 in Case 7). The function now
    retries the SCORING call only (up to 3 times) whenever any label comes
    back with the -1 parse-failure sentinel. Agent outputs are reused so
    retries cost only one Nova Lite call each, not three Nova Pro calls.
    """
    MAX_SCORE_RETRIES = 3
    print(f"\n  {clr('Ablation study', C.PURPLE, C.BOLD)}  —  {case['title']}")

    po_result,  s1 = spin("Ablation: Planner only ...",       multi_agent.run_planner_only,     case["case"])
    pr_result,  s2 = spin("Ablation: Planner + Reasoner ...", multi_agent.run_planner_reasoner, case["case"])
    fp_result,  s3 = spin("Ablation: Full pipeline ...",      multi_agent.run,                  case["case"])

    ablation_eval = None
    s4 = None
    for attempt in range(MAX_SCORE_RETRIES):
        label = ("Scoring ablation three-way ..." if attempt == 0
                 else f"Re-scoring ablation (attempt {attempt+1}/{MAX_SCORE_RETRIES}) ...")
        ablation_eval, s4 = spin(
            label,
            evaluator.compare_three,
            case, po_result, pr_result, fp_result,
            "PlannerOnly", "PlannerReasoner", "FullPipeline",
        )
        parse_failures = [
            lbl for lbl in ("PlannerOnly", "PlannerReasoner", "FullPipeline")
            if ablation_eval.get(lbl, {}).get("Overall", 0) == -1
        ]
        if not parse_failures:
            break
        s4.fail(f"Parse failed for: {parse_failures} — retrying scoring ...")
        if attempt == MAX_SCORE_RETRIES - 1:
            print(f"  {clr('⚠  Ablation scoring failed after', C.RED)} {MAX_SCORE_RETRIES} attempts. "
                  f"Storing partial results (-1 = parse failure, not a real score).")

    po = ablation_eval.get("PlannerOnly",     {}).get("Overall", 0)
    pr = ablation_eval.get("PlannerReasoner", {}).get("Overall", 0)
    fp = ablation_eval.get("FullPipeline",    {}).get("Overall", 0)
    fp_display = f"{fp}/10" if fp != -1 else "PARSE FAIL"
    s4.done(f"Planner-only {po}/10  ·  P+R {pr}/10  ·  Full {fp_display}")
    return ablation_eval


# ── AWS credential helper ─────────────────────────────────────────────────────
def _check_aws_credentials():
    missing = []
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"):
        if not os.environ.get(var, "").strip():
            missing.append(var)
    if missing:
        print(f"\n  {clr('✗', C.RED)}  Missing AWS credentials: {', '.join(missing)}")
        print(f"  Set them in your environment before running:\n")
        for var in missing:
            print(f"    export {var}=<your_value>")
        print(f"\n  AWS_DEFAULT_REGION is typically: us-east-1\n")
        sys.exit(1)


# ── Main experiment ───────────────────────────────────────────────────────────
def run_experiment():
    _check_aws_credentials()
    print("Waking up AWS Bedrock...")
    completion(model=AGENT_MODEL, messages=[{"role": "user", "content": "hello"}], max_tokens=5)
    os.system("cls" if os.name == "nt" else "clear")
    banner("Healthcare AI: Single-Agent vs Multi-Agent")

    cases = (
        [c for c in CLINICAL_CASES if c["id"] in CASES_TO_RUN]
        if CASES_TO_RUN else CLINICAL_CASES
    )

    # Pre-flight cost estimate
    # FIX: budget now includes unconstrained agent (5 agent calls per case, not 4)
    calls_per_case = 5 + (7 if RUN_ABLATION else 0)
    total_est = len(cases) * NUM_RUNS * calls_per_case
    print(f"  {clr('Agent model:', C.GRAY)} {clr(AGENT_MODEL, C.CYAN, C.BOLD)}")
    print(f"  {clr('Judge model:', C.GRAY)} {clr(JUDGE_MODEL, C.PURPLE, C.BOLD)}  "
          f"{clr('(cost-effective)', C.DIM)}")
    print(f"  {clr('Cases:', C.GRAY)}       {clr(str(len(cases)), C.CYAN)}"
          f"  ·  {clr('Runs/case:', C.GRAY)} {clr(str(NUM_RUNS), C.CYAN)}"
          f"  ·  {clr('Ablation:', C.GRAY)} {clr(str(RUN_ABLATION), C.CYAN)}")
    print(f"  {clr('Est. API calls:', C.GRAY)} ~{clr(str(total_est), C.YELLOW)}")
    print(f"  {clr('Variants:', C.GRAY)}    Constrained SA  ·  Unconstrained SA  ·  Multi-Agent")
    print()

    if total_est > 800:
        print(f"  {clr('⚠  Warning:', C.YELLOW)}  High call volume. "
              f"Consider NUM_RUNS=1 and RUN_ABLATION=False for a quick pass.\n")

    # Build clients
    single_agent   = SingleAgent(model=AGENT_MODEL)
    single_agent_u = SingleAgentUnconstrained(model=AGENT_MODEL)
    multi_agent    = MultiAgentSystem(model=AGENT_MODEL)
    evaluator      = Evaluator(judge_model=JUDGE_MODEL)

    # ── Checkpoint / resume ───────────────────────────────────────────────────
    # FIX (Bug 4): The original code had no checkpoint. A crash at Case 6 Run 3
    # lost all results because results.json is only written at the very end.
    # Now a checkpoint file is saved after EVERY completed case. On restart,
    # completed cases are loaded from the checkpoint and skipped, so the run
    # continues from the first incomplete case rather than starting over.
    CHECKPOINT_FILE = "checkpoint.json"

    def _save_checkpoint(eval_runs, raw, ablation):
        cp = {"all_eval_runs": {str(k): v for k, v in eval_runs.items()},
              "all_raw": raw, "ablation_results": ablation}
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(cp, f, indent=2, default=str)

    def _load_checkpoint():
        if not os.path.exists(CHECKPOINT_FILE):
            return {}, [], []
        try:
            with open(CHECKPOINT_FILE, encoding="utf-8") as f:
                cp = json.load(f)
            eval_runs = {int(k): v for k, v in cp.get("all_eval_runs", {}).items()}
            return eval_runs, cp.get("all_raw", []), cp.get("ablation_results", [])
        except Exception as e:
            print(f"  {clr('⚠', C.YELLOW)}  Could not load checkpoint ({e}) — starting fresh.")
            return {}, [], []

    all_eval_runs, all_raw, ablation_results = _load_checkpoint()
    completed_case_ids = set(all_eval_runs.keys())

    if completed_case_ids:
        print(f"  {clr('↩  Resuming from checkpoint.', C.YELLOW)}  "
              f"Already completed: cases {sorted(completed_case_ids)}")
        print()

    for i, case in enumerate(cases):
        # Skip cases already in the checkpoint
        if case["id"] in completed_case_ids:
            print(f"  {clr('✓ Skipping Case ' + str(case['id']), C.DIM)}  "
                  f"{clr('(loaded from checkpoint)', C.DIM)}")
            continue

        print(f"\n  {clr('Case ' + str(case['id']), C.CYAN, C.BOLD)}  "
              f"{clr(case['title'], C.WHITE)}  "
              f"{clr('[' + case.get('specialty', '') + ']', C.DIM)}")
        separator()

        case_runs = []
        for run_num in range(1, NUM_RUNS + 1):
            eval_result, raw = run_single_case(
                case, single_agent, single_agent_u, multi_agent, evaluator, run_num
            )
            case_runs.append(eval_result)
            all_raw.append(raw)

        all_eval_runs[case["id"]] = case_runs

        if RUN_ABLATION:
            ablation_eval = run_ablation(case, multi_agent, evaluator)
            ablation_results.append(ablation_eval)

        # Save checkpoint immediately after each completed case
        _save_checkpoint(all_eval_runs, all_raw, ablation_results)
        print(f"  {clr('💾', C.DIM)}  Checkpoint saved  "
              f"{clr(f'(cases {sorted(all_eval_runs.keys())} complete)', C.DIM)}")

        if i < len(cases) - 1:
            print(f"\n  {clr('⏸', C.DIM)}  Pausing {BETWEEN_CASE_PAUSE}s "
                  f"before next case  {clr('(TPM guard)', C.DIM)} ...")
            time.sleep(BETWEEN_CASE_PAUSE)

    # Average and summarise
    all_averaged = [
        average_results(all_eval_runs[case["id"]])
        for case in cases
    ]

    print()
    print_summary(all_averaged)

    # Save outputs
    output_data = {
        "config": {
            "agent_model":    AGENT_MODEL,
            "judge_model":    JUDGE_MODEL,
            "num_runs":       NUM_RUNS,
            "num_cases":      len(cases),
            "ablation":       RUN_ABLATION,
            "api_calls_used": _quota.count,
        },
        "results":          all_raw,
        "averaged_results": all_averaged,
        "ablation":         ablation_results,
    }

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"  {clr('💾', C.DIM)}  Saved  {clr('results.json', C.CYAN)}")

    # Remove checkpoint on successful completion — it's no longer needed
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print(f"  {clr('🗑', C.DIM)}   Checkpoint removed (run completed cleanly)")

    try:
        seen, report_raw = set(), []
        for r in reversed(all_raw):
            cid = r["case"]["id"]
            if cid not in seen:
                seen.add(cid)
                report_raw.insert(0, r)
        report_path = generate_html_report(report_raw, AGENT_MODEL, ablation_results)
        print(f"  {clr('📊', C.DIM)}  Saved  {clr(report_path, C.CYAN)}")
    except Exception as e:
        print(f"  {clr('⚠', C.YELLOW)}  Report generation skipped: {e}")

    print(f"\n  {clr('→  Open report.html in your browser.', C.GRAY)}")
    print(f"  {clr('→  Total API calls this session:', C.GRAY)} "
          f"{clr(str(_quota.count), C.CYAN)}\n")

    return output_data


if __name__ == "__main__":
    run_experiment()
