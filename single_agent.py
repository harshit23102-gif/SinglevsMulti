"""
single_agent.py — Single-Agent clinical reasoning system.

Two variants are provided to enable a fair architectural comparison:

  SingleAgent              — Constrained (600 tokens). Simulates realistic
                             single-pass deployment under response-time limits.

  SingleAgentUnconstrained — Budget-matched (2300 tokens, same total as the
                             multi-agent pipeline). Isolates architecture from
                             token budget in the comparison.

Both variants use exponential-backoff retry on transient API failures.

Model: AWS Bedrock — amazon.nova-pro-v1:0 (via LiteLLM bedrock/ prefix).
Credentials are read from the environment:
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION

FIXES vs original:
  - Imports call_with_retry from utils.py (no more copy-paste).
  - SingleAgentUnconstrained.run() now includes the same warm-up ping that
    SingleAgent.run() has. Without it, the unconstrained variant absorbed the
    cold-start latency penalty that the constrained variant avoided, making
    latency comparisons unfair.
"""

import time
import logging
import litellm

from utils import call_with_retry

logger = logging.getLogger(__name__)


# ── Constrained Single-Agent ──────────────────────────────────────────────────
class SingleAgent:
    """
    Constrained Single-Agent (600 tokens).

    One LLM call handles the entire clinical reasoning process end-to-end.
    Token budget is intentionally limited to simulate realistic single-pass
    deployment constraints (response-time budgets, API cost management).
    """

    MAX_TOKENS   = 600
    TEMPERATURE  = 0.3

    SYSTEM_PROMPT = """You are a clinical AI assistant. Given a patient case, provide a clinical assessment:
1. Primary Diagnosis
2. Differential Diagnoses (2-3)
3. Immediate Treatment Plan
4. Key Investigations
5. Disposition

Be concise and direct."""

    def __init__(self, model: str = "bedrock/us.amazon.nova-pro-v1:0"):
        self.model = model
        self.name  = "Single-Agent (Constrained)"

    def _call(self, patient_case: str) -> tuple[str, int, int]:
        def _api():
            return litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Patient Case:\n{patient_case}"},
                ],
                temperature=self.TEMPERATURE,
                max_tokens=self.MAX_TOKENS,
            )
        response, retries = call_with_retry(_api)
        return response.choices[0].message.content, response.usage.total_tokens, retries

    def run(self, patient_case: str) -> dict:
        """
        Run the single-agent on a clinical case. Returns a result dict.

        Fairness measures vs multi-agent:
        1. Warm-up ping  — fires a max_tokens=1 call before the timed block so
           Bedrock's endpoint is warm, matching the multi-agent whose Planner
           naturally warms the endpoint for Reasoner and Verifier.
        2. Gap sleep outside timer — the QuotaTracker's call-gap sleep is
           triggered by the warm-up ping, so it does not inflate the measured
           latency of the real call.
        """
        # Warm-up: wake the Bedrock endpoint so cold-start cost is not charged
        # to the single-agent's latency. max_tokens=1 keeps this near-instant.
        try:
            litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": "ready"}],
                max_tokens=1,
                temperature=0.0,
            )
        except Exception:
            pass  # warm-up failure is non-fatal; proceed with the real call

        t0 = time.time()
        output, tokens, retries = self._call(patient_case)
        return {
            "agent":           self.name,
            "output":          output,
            "tokens_used":     tokens,
            "latency_seconds": round(time.time() - t0, 2),
            "num_llm_calls":   1,
            "api_retries":     retries,
            "steps":           [{"agent": "Single-Agent", "output": output}],
        }


# ── Budget-Matched (Unconstrained) Single-Agent ───────────────────────────────
class SingleAgentUnconstrained:
    """
    Budget-Matched Single-Agent (2300 tokens — same total as multi-agent pipeline).

    Uses the same base LLM and approximately the same token budget as the
    multi-agent system (Planner 600 + Reasoner 900 + Verifier 1200 = 2700 max,
    ~2300 in practice), but with a richer chain-of-thought system prompt and
    no agent decomposition.

    Use this variant to determine whether the multi-agent quality advantage
    is attributable to architectural decomposition or simply to a larger
    token budget. This is the fairest direct comparison.

    FIX: Added the same warm-up ping that SingleAgent.run() uses. Without it,
    this variant unfairly absorbed Bedrock cold-start latency that the
    constrained variant and multi-agent avoided, inflating its measured latency.
    """

    MAX_TOKENS  = 2300
    TEMPERATURE = 0.3

    SYSTEM_PROMPT = """You are an expert clinical AI assistant performing a comprehensive clinical assessment.

Work through the case STEP BY STEP:

STEP 1 — CRITICAL FINDINGS
  List the 3-5 most important findings. Flag any AMBIGUOUS or MISLEADING findings.

STEP 2 — DIFFERENTIAL DIAGNOSIS
  For each of your top 3 competing diagnoses, state:
    • Evidence FOR this diagnosis
    • Evidence AGAINST this diagnosis
  Then rank by likelihood with a confidence percentage.

STEP 3 — PRIMARY DIAGNOSIS
  State your primary diagnosis with supporting evidence.
  Explicitly note which life-threatening alternatives you are ruling out and why.

STEP 4 — IMMEDIATE TREATMENT PLAN
  Prioritise interventions:
    • First 30 minutes (immediate)
    • Next 2 hours (urgent)
  Include specific drug names, doses, and routes where appropriate.

STEP 5 — INVESTIGATIONS
  List urgently needed investigations in priority order.

STEP 6 — SAFETY CHECK
  Review the patient's medications and comorbidities. Flag any:
    • Drug interactions
    • Contraindications to planned treatments
    • Patient-specific safety concerns

STEP 7 — DISPOSITION
  Recommend appropriate care level (ward / HDU / ICU) with justification.

End with a brief SUMMARY. Do not leave any section incomplete."""

    def __init__(self, model: str = "bedrock/us.amazon.nova-pro-v1:0"):
        self.model = model
        self.name  = "Single-Agent (Unconstrained)"

    def _call(self, patient_case: str) -> tuple[str, int, float, int]:
        _latency = [0.0]

        def _api():
            t0 = time.time()
            resp = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Patient Case:\n{patient_case}"},
                ],
                temperature=self.TEMPERATURE,
                max_tokens=self.MAX_TOKENS,
            )
            _latency[0] = round(time.time() - t0, 2)
            return resp

        response, retries = call_with_retry(_api)
        return response.choices[0].message.content, response.usage.total_tokens, _latency[0], retries

    def run(self, patient_case: str) -> dict:
        """
        Run the unconstrained single-agent on a clinical case. Returns a result dict.

        FIX: Warm-up ping added (same as SingleAgent.run()) so cold-start latency
        is not charged to this variant's measured latency, making the three-way
        latency comparison (Constrained / Unconstrained / Multi-Agent) fair.
        """
        # Warm-up ping — equalises cold-start latency across all three variants.
        try:
            litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": "ready"}],
                max_tokens=1,
                temperature=0.0,
            )
        except Exception:
            pass  # non-fatal

        output, tokens, latency, retries = self._call(patient_case)
        return {
            "agent":           self.name,
            "output":          output,
            "tokens_used":     tokens,
            "latency_seconds": latency,
            "num_llm_calls":   1,
            "api_retries":     retries,
            "steps":           [{"agent": "Single-Agent (Unconstrained)", "output": output}],
        }
