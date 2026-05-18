"""
multi_agent.py — Multi-Agent clinical reasoning system.

Three specialized agents run in sequence:
  1. Planner  — flags ambiguities, red herrings, and competing diagnoses
  2. Reasoner — argues FOR / AGAINST each differential step by step
  3. Verifier — challenges the reasoning, catches safety gaps, finalizes output

Ablation variants expose sub-pipeline configurations so the contribution of
each agent can be measured independently:
  run_planner_only()       → Planner output only (no diagnosis)
  run_planner_reasoner()   → Planner + Reasoner (no Verifier safety check)
  run()                    → Full pipeline (Planner + Reasoner + Verifier)

main.py calls _planner / _reasoner / _verifier directly for live terminal
progress. The run() and ablation methods are available for programmatic use.

Model: AWS Bedrock — amazon.nova-pro-v1:0 (via LiteLLM bedrock/ prefix).
Credentials are read from the environment:
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION

FIXES vs original:
  - Imports call_with_retry from utils.py (no more copy-paste).
  - Verifier max_tokens raised 800 → 1200 → 1500 to prevent mid-sentence truncation.
    The original cap silently cut off the Verifier's final output (e.g.
    "Piperacillin-tazobactam 3.375g IV or Meropen" in Case 2).
  - is_truncated() check added — truncated outputs are flagged in the result
    dict so downstream code (evaluator, report) can surface the warning.
"""

import time
import logging
import litellm

from utils import call_with_retry, is_truncated

logger = logging.getLogger(__name__)


class MultiAgentSystem:

    def __init__(self, model: str = "bedrock/us.amazon.nova-pro-v1:0"):
        self.model = model
        self.name  = "Multi-Agent"

    # ── LLM call helper ───────────────────────────────────────────────────────
    def _call(self, system: str, user: str, max_tokens: int = 700) -> tuple[str, int, int]:
        """Single LLM call with retry. Returns (content, total_tokens, retry_count)."""
        def _api():
            return litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0.3,
                max_tokens=max_tokens,
            )
        response, retries = call_with_retry(_api)
        return response.choices[0].message.content, response.usage.total_tokens, retries

    # ── Agent 1: Planner ──────────────────────────────────────────────────────
    def _planner(self, patient_case: str) -> tuple[str, int, int]:
        """
        Decomposes the case BEFORE diagnosis.
        Flags ambiguities, red herrings, and top competing diagnoses.
        Produces a structured roadmap for the Reasoner.
        """
        system = """You are a Clinical Case Planner AI. Analyze the case BEFORE diagnosing.

Output a structured plan covering:
1. The 3-5 most critical clinical findings
2. AMBIGUOUS or MISLEADING findings that could lead to a wrong diagnosis
3. Potential RED HERRINGS — findings that distract from the real problem
4. Top 3 competing diagnoses to investigate (do NOT diagnose yet)
5. Specific sub-tasks for the Reasoner (what must be explicitly reasoned through?)
6. Patient-specific SAFETY CONCERNS (medications, comorbidities, contraindications)

This is a planning document — no diagnosis yet. Be concise and sharp."""

        return self._call(
            system,
            f"Create a reasoning plan for this case:\n{patient_case}",
            max_tokens=600,
        )

    # ── Agent 2: Reasoner ─────────────────────────────────────────────────────
    def _reasoner(self, patient_case: str, plan: str) -> tuple[str, int, int]:
        """
        Follows the plan. Explicitly argues FOR and AGAINST each differential
        before reaching a conclusion.
        """
        system = """You are a Clinical Reasoning AI. You receive a case and a Planner's roadmap.

You MUST:
1. Address each AMBIGUOUS finding — explain your interpretation
2. For each competing diagnosis: state evidence FOR and AGAINST
3. Conclude with Primary Diagnosis (confidence %) and clearest supporting evidence
4. List 2-3 Differentials ruled out — explain WHY each was deprioritized
5. Propose IMMEDIATE treatment with priority order (first 30 min vs next 2 hours)
6. List urgent investigations needed
7. Address all safety concerns flagged by the Planner

Show ALL reasoning steps. This output will be reviewed by a senior clinician."""

        return self._call(
            system,
            (
                f"Patient Case:\n{patient_case}\n\n"
                f"---\nPlanner's Roadmap:\n{plan}\n\n"
                f"---\nPerform step-by-step clinical reasoning:"
            ),
            max_tokens=900,
        )

    # ── Agent 3: Verifier ─────────────────────────────────────────────────────
    def _verifier(self, patient_case: str, reasoning: str) -> tuple[str, int, int, bool]:
        """
        Senior clinician review. Actively looks for errors, omissions, and
        safety gaps before producing the final verified assessment.

        FIX: max_tokens raised 800 → 1200 → 1500.
        The original 800-token cap silently truncated the Verifier mid-sentence
        on complex cases (e.g. Case 2: "...or Meropen"). The Verifier produces
        the most critical output in the pipeline (the final safety-checked
        assessment) — it must not be silently clipped.

        Returns (content, total_tokens, retry_count, was_truncated).
        """
        system = """You are a Senior Clinical Verifier AI — a specialist consultant actively looking for errors.

Your job:
1. CHALLENGE the primary diagnosis — is the evidence sufficient? What could still be missed?
2. Check treatment plan: correct priority order? Appropriate doses/routes?
3. Identify any MISSED safety concern, drug interaction, or contraindication
4. Check if any life-threatening alternative was dismissed too quickly
5. Add any critical intervention or monitoring parameter that was omitted
6. Produce the FINAL verified clinical assessment — complete, safe, and well-organized

Be a tough reviewer. Fix gaps in the final output rather than just flagging them.
Always end with a clear, complete SUMMARY section — do not leave it mid-sentence."""

        content, tokens, retries = self._call(
            system,
            (
                f"Patient Case:\n{patient_case}\n\n"
                f"---\nReasoner's Assessment:\n{reasoning}\n\n"
                f"---\nVerify and produce the final clinical assessment:"
            ),
            max_tokens=1500,  # FIX: raised 800→1200→1500; Case 2 still truncated at 1200
        )
        truncated = is_truncated(content)
        if truncated:
            logger.warning(
                "Verifier output appears truncated (does not end with a sentence "
                "terminator). Consider raising max_tokens further or splitting the output."
            )
        return content, tokens, retries, truncated

    # ── Full pipeline ─────────────────────────────────────────────────────────
    def run(self, patient_case: str) -> dict:
        """
        Runs the full Planner → Reasoner → Verifier pipeline.
        Note: main.py calls each agent separately for live terminal progress.
        """
        t0 = time.time()
        plan,      tok1, r1          = self._planner(patient_case)
        reasoning, tok2, r2          = self._reasoner(patient_case, plan)
        final_out, tok3, r3, trunc   = self._verifier(patient_case, reasoning)
        return {
            "agent":              self.name,
            "output":             final_out,
            "tokens_used":        tok1 + tok2 + tok3,
            "latency_seconds":    round(time.time() - t0, 2),
            "num_llm_calls":      3,
            "api_retries":        r1 + r2 + r3,
            "verifier_truncated": trunc,       # FIX: surfaces truncation flag
            "steps": [
                {"agent": "Planner",  "output": plan},
                {"agent": "Reasoner", "output": reasoning},
                {"agent": "Verifier", "output": final_out},
            ],
        }

    # ── Ablation: Planner only ────────────────────────────────────────────────
    def run_planner_only(self, patient_case: str) -> dict:
        t0 = time.time()
        plan, tok1, r1 = self._planner(patient_case)
        return {
            "agent":           "Multi-Agent (Planner Only)",
            "output":          plan,
            "tokens_used":     tok1,
            "latency_seconds": round(time.time() - t0, 2),
            "num_llm_calls":   1,
            "api_retries":     r1,
            "steps": [
                {"agent": "Planner", "output": plan},
            ],
        }

    # ── Ablation: Planner + Reasoner ──────────────────────────────────────────
    def run_planner_reasoner(self, patient_case: str) -> dict:
        t0 = time.time()
        plan,      tok1, r1 = self._planner(patient_case)
        reasoning, tok2, r2 = self._reasoner(patient_case, plan)
        return {
            "agent":           "Multi-Agent (Planner + Reasoner)",
            "output":          reasoning,
            "tokens_used":     tok1 + tok2,
            "latency_seconds": round(time.time() - t0, 2),
            "num_llm_calls":   2,
            "api_retries":     r1 + r2,
            "steps": [
                {"agent": "Planner",  "output": plan},
                {"agent": "Reasoner", "output": reasoning},
            ],
        }
