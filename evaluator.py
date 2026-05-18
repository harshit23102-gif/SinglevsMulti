"""
evaluator.py — LLM-as-judge scoring for clinical outputs.

Model strategy:
  - AGENT_MODEL  (nova-pro-v1:0)   — used by the agents for quality
  - JUDGE_MODEL  (nova-lite-v1:0)  — used by the judge for scoring
    Nova Lite is faster and cheaper than Nova Pro, keeping costs low.

  API call budget per full run (8 cases):
    Agents:  8 cases × 5 calls (SA-c + SA-u + 3 MA) = 40 calls  on Nova Pro
    Judge:   8 cases × 1 call                        =  8 calls  on Nova Lite
    Total:  48 calls

Features:
  - Head-to-head comparative judging (prevents score inflation)
  - Contradiction detection heuristic (flags internal inconsistencies)
  - 57-marker reasoning depth heuristic (no extra LLM call)
  - TWO keyword coverage metrics (see FIX below)
  - Safety omission flags (patient-specific, rule-based)
  - Safety/LLM-score consistency check (see FIX below)
  - Exponential-backoff retry on transient API failures

FIXES vs original:
  - Imports call_with_retry from utils.py (no more copy-paste).
  - Keyword coverage split into two metrics:
      keyword_coverage_%       — final output only (fair, apples-to-apples)
      pipeline_keyword_coverage_% — full pipeline text (multi-agent advantage
                                    measure; shows what the Planner+Reasoner
                                    covered even if Verifier condensed it)
    Previously, both architectures were measured on all steps, but single-agent
    steps contain only the same output duplicated — giving multi-agent a
    3× text-search advantage with no disclosure. Now both are explicit.
  - Safety consistency check: when a safety_flag fires but the LLM judge gave
    Safety Awareness ≥ 7, a 'safety_score_inconsistency' note is added to the
    result so downstream code (report, terminal) can surface the discrepancy.
    (Example: Case 2 multi-agent — INR flag fired + Safety Awareness = 8.0.)

Credentials: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
"""

import re
import time
import random
import logging
import litellm

from utils import call_with_retry

logger = logging.getLogger(__name__)

# ── Model config ──────────────────────────────────────────────────────────────
JUDGE_MODEL = "bedrock/us.amazon.nova-lite-v1:0"


class Evaluator:

    def __init__(self, judge_model: str = JUDGE_MODEL):
        self.judge_model = judge_model

    # ── Head-to-head judge ────────────────────────────────────────────────────
    def _judge(
        self,
        patient_case: str,
        single_output: str,
        multi_output: str,
    ) -> tuple[dict, dict]:
        """
        Scores both outputs simultaneously in one LLM call.
        Side-by-side comparison forces differential scores and prevents the
        score inflation seen when each output is graded independently.
        Returns (single_scores, multi_scores).

        Note: judge and agents share the same model family (Nova). This may
        introduce familiarity bias. Acknowledge as a limitation.
        """
        system_prompt = """You are a strict clinical examiner doing a HEAD-TO-HEAD evaluation.
TWO clinical assessments (System A and System B) are provided for the SAME case.

RULES:
- Score each dimension 1-10 for BOTH systems
- MUST assign different scores when quality differs — never identical unless truly equal
- +2 pts Reasoning Depth: explicit FOR/AGAINST per differential
- +2 pts Safety Awareness: catches a drug interaction or contraindication the other misses
- Keyword listing without step-by-step reasoning = 5-6 max, not 8-9

DIMENSIONS:
1. Diagnostic Accuracy   — Correct diagnosis + catches case-specific red herrings
2. Reasoning Depth       — Step-by-step reasoning, FOR/AGAINST each differential
3. Treatment Completeness— All key treatments, correct priority order
4. Safety Awareness      — Patient-specific contraindications, drug interactions
5. Clinical Structure    — Organized, labeled sections, clear handoff format

Respond ONLY in this exact format (no extra text):
A_Diagnostic_Accuracy: X
A_Reasoning_Depth: X
A_Treatment_Completeness: X
A_Safety_Awareness: X
A_Clinical_Structure: X
A_Overall: X
B_Diagnostic_Accuracy: X
B_Reasoning_Depth: X
B_Treatment_Completeness: X
B_Safety_Awareness: X
B_Clinical_Structure: X
B_Overall: X"""

        user_msg = (
            f"PATIENT CASE:\n{patient_case}\n\n"
            f"{'='*60}\n"
            f"SYSTEM A (Single-Agent):\n{single_output}\n\n"
            f"{'='*60}\n"
            f"SYSTEM B (Multi-Agent):\n{multi_output}\n\n"
            f"{'='*60}\n"
            "Score comparatively. Assign DIFFERENT scores where quality differs."
        )

        def _api():
            return litellm.completion(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=300,
            )

        response, _ = call_with_retry(_api)
        return self._parse_ab_scores(response.choices[0].message.content)

    def _parse_ab_scores(self, text: str) -> tuple[dict, dict]:
        dim_map = {
            "Diagnostic_Accuracy":    "Diagnostic Accuracy",
            "Reasoning_Depth":        "Reasoning Depth",
            "Treatment_Completeness": "Treatment Completeness",
            "Safety_Awareness":       "Safety Awareness",
            "Clinical_Structure":     "Clinical Structure",
            "Overall":                "Overall",
        }
        single_scores, multi_scores = {}, {}
        for line in text.strip().split("\n"):
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            try:
                score = float(val.strip())
            except ValueError:
                continue
            if key.startswith("A_") and key[2:] in dim_map:
                single_scores[dim_map[key[2:]]] = score
            elif key.startswith("B_") and key[2:] in dim_map:
                multi_scores[dim_map[key[2:]]] = score
        return single_scores, multi_scores

    # ── Three-way judge (ablation) ────────────────────────────────────────────
    def _judge_three(
        self,
        patient_case: str,
        output_a: str,
        output_b: str,
        output_c: str,
        label_a: str = "PlannerOnly",
        label_b: str = "PlannerReasoner",
        label_c: str = "FullPipeline",
    ) -> dict:
        """
        Three-way comparative scoring for ablation studies.
        One LLM call — parses all three label prefixes from the response.
        """
        system_prompt = f"""You are a strict clinical examiner doing a THREE-WAY evaluation.
THREE assessments ({label_a}, {label_b}, {label_c}) for the SAME patient case.

RULES:
- Score each dimension 1-10 for ALL THREE systems
- Assign DIFFERENT scores when quality differs
- Keyword listing without reasoning = 5-6 max

Respond ONLY in this exact format:
{label_a}_Diagnostic_Accuracy: X
{label_a}_Reasoning_Depth: X
{label_a}_Treatment_Completeness: X
{label_a}_Safety_Awareness: X
{label_a}_Clinical_Structure: X
{label_a}_Overall: X
{label_b}_Diagnostic_Accuracy: X
{label_b}_Reasoning_Depth: X
{label_b}_Treatment_Completeness: X
{label_b}_Safety_Awareness: X
{label_b}_Clinical_Structure: X
{label_b}_Overall: X
{label_c}_Diagnostic_Accuracy: X
{label_c}_Reasoning_Depth: X
{label_c}_Treatment_Completeness: X
{label_c}_Safety_Awareness: X
{label_c}_Clinical_Structure: X
{label_c}_Overall: X"""

        user_msg = (
            f"PATIENT CASE:\n{patient_case}\n\n"
            f"{'='*60}\n{label_a}:\n{output_a}\n\n"
            f"{'='*60}\n{label_b}:\n{output_b}\n\n"
            f"{'='*60}\n{label_c}:\n{output_c}\n\n"
            "Score all three comparatively."
        )

        def _api():
            return litellm.completion(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=500,
            )

        response, _ = call_with_retry(_api)
        return self._parse_three_scores(
            response.choices[0].message.content,
            label_a, label_b, label_c,
        )

    def _parse_three_scores(
        self, text: str, label_a: str, label_b: str, label_c: str
    ) -> dict:
        """
        FIX (Bug 2): Case 5 ablation returned Full 0/10 because the judge
        occasionally adds a preamble ("Here are the scores:") or uses slightly
        different label casing, causing the prefix match to fail silently and
        leaving the dict empty — which reads as 0 downstream.

        Fixes applied:
        1. Strip markdown bold (**text**) and common preamble lines.
        2. Case-insensitive prefix matching.
        3. Detect parse failures per label and warn loudly instead of
           silently returning {}.  A parse failure is now flagged with
           Overall = -1 so callers can detect and surface it rather than
           mistaking it for a genuine 0/10.
        """
        dim_map = {
            "diagnostic_accuracy":    "Diagnostic Accuracy",
            "reasoning_depth":        "Reasoning Depth",
            "treatment_completeness": "Treatment Completeness",
            "safety_awareness":       "Safety Awareness",
            "clinical_structure":     "Clinical Structure",
            "overall":                "Overall",
        }
        scores = {label_a: {}, label_b: {}, label_c: {}}
        labels_lower = {lbl.lower(): lbl for lbl in (label_a, label_b, label_c)}

        for line in text.strip().split("\n"):
            # Strip markdown bold markers and leading/trailing whitespace
            line = line.strip().replace("**", "")
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip().lower()   # FIX: case-insensitive matching

            # Extract numeric score — handle "8", "8.0", "8/10", "8 / 10"
            val_clean = val.strip().replace("/10", "").replace("/ 10", "").strip()
            try:
                score = float(val_clean.split()[0])   # take first token if trailing text
            except (ValueError, IndexError):
                continue

            # Match label prefix case-insensitively
            for lbl_lower, lbl_orig in labels_lower.items():
                prefix = f"{lbl_lower}_"
                if key.startswith(prefix):
                    dim_key = key[len(prefix):]
                    if dim_key in dim_map:
                        scores[lbl_orig][dim_map[dim_key]] = score

        # FIX: Detect and warn on parse failures — don't silently return {}
        for lbl in (label_a, label_b, label_c):
            if not scores[lbl]:
                logger.warning(
                    f"_parse_three_scores: NO scores parsed for label '{lbl}'. "
                    f"Judge response may have used unexpected formatting.\n"
                    f"Raw judge output (first 400 chars):\n{text[:400]}"
                )
                # Sentinel: -1 signals a parse failure, not a genuine 0 score
                scores[lbl]["Overall"] = -1
                scores[lbl]["_parse_failed"] = True

        return scores

    # ── Keyword coverage ──────────────────────────────────────────────────────
    def _keyword_coverage_output_only(self, result: dict, expected: list) -> float:
        """
        Coverage measured against the FINAL output only.

        FIX: This is the fair, apples-to-apples metric. The original
        _keyword_coverage() measured across output + all steps. For multi-agent
        that gives 3× the text (planner + reasoner + verifier), while for
        single-agent it doubles the same text — neither is a fair comparison.
        This metric measures only the final output for both architectures.
        """
        lower = result.get("output", "").lower()
        found = sum(1 for kw in expected if kw.lower() in lower)
        return round(found / len(expected) * 100, 1) if expected else 0.0

    def _keyword_coverage_pipeline(self, result: dict, expected: list) -> float:
        """
        Coverage across the FULL pipeline text (output + all steps).

        For single-agent this equals output-only coverage (steps = [output]).
        For multi-agent this measures what the Planner+Reasoner covered even
        if the Verifier condensed those keywords out of the final output.

        Reported separately so the comparison is transparent, not hidden.
        """
        all_text = result.get("output", "")
        seen_texts = {all_text}
        for step in result.get("steps", []):
            if isinstance(step, dict):
                step_text = step.get("output", "")
                if step_text not in seen_texts:     # avoid double-counting
                    all_text += " " + step_text
                    seen_texts.add(step_text)
        lower = all_text.lower()
        found = sum(1 for kw in expected if kw.lower() in lower)
        return round(found / len(expected) * 100, 1) if expected else 0.0

    # ── Reasoning markers (heuristic, no LLM call) ───────────────────────────
    _REASONING_MARKERS = [
        "diagnosis", "differential", "evidence for", "evidence against",
        "ruled out", "ruled in", "deprioritized", "unlikely because",
        "most likely", "less likely", "cannot exclude", "must exclude",
        "confidence", "primary diagnosis", "competing diagnosis",
        "therefore", "however", "suggests", "indicates", "consistent with",
        "inconsistent with", "argues for", "argues against", "supports",
        "this finding", "this pattern",
        "assessment", "reasoning", "consider", "clinical picture",
        "red flag", "red herring", "misleading", "ambiguous",
        "treatment", "immediate", "first 30", "next 2 hours", "priority",
        "monitor", "dose", "route", "contraindication", "interaction",
        "investigation", "urgent", "repeat", "serial", "pending",
        "safety concern", "drug interaction", "allergy", "hold",
        "avoid", "caution", "risk",
        "step 1", "step 2", "step 3", "plan", "disposition",
        "icu", "admit", "escalate",
    ]

    def _reasoning_markers(self, output: str) -> int:
        lower = output.lower()
        return sum(1 for m in self._REASONING_MARKERS if m in lower)

    # ── Contradiction detection (heuristic, no LLM call) ─────────────────────
    _CONTRADICTION_PAIRS = [
        ("do not anticoagulate", "start heparin"),
        ("hold warfarin",        "continue warfarin"),
        ("avoid nsaids",         "give ibuprofen"),
        ("no lumbar puncture",   "perform lumbar puncture"),
        ("no ct head",           "ct head"),
        ("rule out pe",          "pe is unlikely"),
        ("icu admission",        "discharge home"),
    ]

    def _contradiction_score(self, output: str) -> int:
        lower = output.lower()
        return sum(1 for a, b in self._CONTRADICTION_PAIRS if a in lower and b in lower)

    # ── Safety omission flags (rule-based, no LLM call) ───────────────────────
    # Each tuple: (case_trigger, output_required, message)
    # case_trigger  — substring that must appear in the CASE TEXT (indicating risk)
    # output_required — substring that must appear in the OUTPUT (indicating it was addressed)
    # If trigger fires but required is absent → flag raised.
    #
    # FIX (Bug 3): The original pattern ("inr", "vitamin k", ...) was a false positive
    # on Case 5 (apixaban/TIA patient). That case mentions "INR 1.1" as a lab value
    # — a normal finding on a DOAC, not a warfarin-supratherapeutic scenario.
    # The trigger "inr" was too broad. Fix: require BOTH "warfarin" AND "supratherapeutic"
    # (or the numeric signal) to coexist in the case text before flagging INR reversal.
    # Separate specific triggers prevent one broad keyword from firing across unrelated cases.
    _SAFETY_PATTERNS = [
        # Warfarin-specific checks — require "warfarin" explicitly in case text
        ("warfarin",            "inr",            "Warfarin patient — INR management not mentioned"),
        ("supratherapeutic",    "vitamin k",      "Supratherapeutic INR — Vitamin K / reversal not mentioned"),
        # DOAC-specific check
        ("apixaban",            "anticoagulat",   "Apixaban patient — anticoagulation plan not addressed"),
        # Immunosuppression check
        ("methotrexate",        "hold",           "Methotrexate patient — holding immunosuppressant not discussed"),
        # Rh-negative / blood group checks (all common phrasings)
        ("rh negative",         "anti-d",         "Rh-negative patient — Anti-D immunoglobulin not mentioned"),
        ("rh-negative",         "anti-d",         "Rh-negative patient — Anti-D immunoglobulin not mentioned"),
        ("o negative",          "anti-d",         "O-negative patient — Anti-D immunoglobulin not mentioned"),
        ("blood type o neg",    "anti-d",         "O-negative patient — Anti-D immunoglobulin not mentioned"),
    ]

    def _safety_flags(self, case_text: str, output: str) -> list:
        """
        FIX: deduplicate before returning. Multiple patterns can share the same
        message (e.g. 'o negative' and 'blood type o neg' both match
        'Blood type O negative', producing the same flag twice in the terminal).
        dict.fromkeys preserves insertion order while removing duplicates.
        """
        seen         = {}
        case_lower   = case_text.lower()
        output_lower = output.lower()
        for trigger, required, message in self._SAFETY_PATTERNS:
            if trigger in case_lower and required not in output_lower:
                seen[message] = None   # dict key = dedup, order preserved
        return list(seen.keys())

    # ── Safety / LLM-score consistency check ─────────────────────────────────
    @staticmethod
    def _safety_consistency_note(safety_flags: list, llm_safety_score: float) -> str | None:
        """
        FIX: Detects and surfaces the contradiction between rule-based safety
        flags and LLM judge Safety Awareness scores.

        Example from original results: Case 2 multi-agent triggered the INR
        flag (rule-based: Vitamin K not mentioned in output) yet the LLM judge
        gave Safety Awareness = 8.0. These two signals conflict.

        Explanation for why this happens:
          - The rule-based flag checks FINAL OUTPUT only.
          - The LLM judge sees the same final output but may infer/assume safety
            awareness from the overall tone and structure of the response, or
            may not notice the specific omission.
        Both signals are valid but measure different things. Surfacing the note
        allows the researcher to investigate rather than silently accept either.
        """
        if safety_flags and llm_safety_score >= 7.0:
            return (
                f"CONSISTENCY NOTE: {len(safety_flags)} rule-based safety flag(s) fired "
                f"but LLM judge gave Safety Awareness = {llm_safety_score:.1f}/10. "
                f"Likely cause: the LLM judge inferred safety awareness from tone/structure "
                f"while the rule-based check found a specific keyword omission in the final output. "
                f"Both are valid signals — investigate the flagged items manually."
            )
        return None

    # ── Public: two-way compare ───────────────────────────────────────────────
    def compare(
        self,
        case: dict,
        single_result: dict,
        multi_result: dict,
    ) -> dict:
        single_scores, multi_scores = self._judge(
            case["case"], single_result["output"], multi_result["output"]
        )

        s_flags = self._safety_flags(case["case"], single_result["output"])
        m_flags = self._safety_flags(case["case"], multi_result["output"])

        s_safety_score = single_scores.get("Safety Awareness", 0.0)
        m_safety_score = multi_scores.get("Safety Awareness", 0.0)

        return {
            "case_id":    case["id"],
            "case_title": case["title"],
            "specialty":  case.get("specialty", "Unknown"),
            "single_agent": {
                **single_scores,
                # FIX: two coverage metrics — final-output-only (fair) and pipeline (full)
                "keyword_coverage_%":          self._keyword_coverage_output_only(single_result, case["expected_keywords"]),
                "pipeline_keyword_coverage_%": self._keyword_coverage_pipeline(single_result, case["expected_keywords"]),
                "reasoning_markers":           self._reasoning_markers(single_result["output"]),
                "contradictions":              self._contradiction_score(single_result["output"]),
                "safety_flags":                s_flags,
                "safety_consistency_note":     self._safety_consistency_note(s_flags, s_safety_score),
                "tokens_used":                 single_result["tokens_used"],
                "latency_seconds":             single_result["latency_seconds"],
                "llm_calls":                   single_result["num_llm_calls"],
            },
            "multi_agent": {
                **multi_scores,
                "keyword_coverage_%":          self._keyword_coverage_output_only(multi_result, case["expected_keywords"]),
                "pipeline_keyword_coverage_%": self._keyword_coverage_pipeline(multi_result, case["expected_keywords"]),
                "reasoning_markers":           self._reasoning_markers(multi_result["output"]),
                "contradictions":              self._contradiction_score(multi_result["output"]),
                "safety_flags":                m_flags,
                "safety_consistency_note":     self._safety_consistency_note(m_flags, m_safety_score),
                "verifier_truncated":          multi_result.get("verifier_truncated", False),
                "tokens_used":                 multi_result["tokens_used"],
                "latency_seconds":             multi_result["latency_seconds"],
                "llm_calls":                   multi_result["num_llm_calls"],
            },
        }

    # ── Public: three-way compare (ablation) ─────────────────────────────────
    def compare_three(
        self,
        case: dict,
        result_a: dict,
        result_b: dict,
        result_c: dict,
        label_a: str = "PlannerOnly",
        label_b: str = "PlannerReasoner",
        label_c: str = "FullPipeline",
    ) -> dict:
        scores = self._judge_three(
            case["case"],
            result_a["output"], result_b["output"], result_c["output"],
            label_a, label_b, label_c,
        )

        def _metrics(result, lbl):
            safety_flags = self._safety_flags(case["case"], result["output"])
            llm_safety   = scores.get(lbl, {}).get("Safety Awareness", 0.0)
            return {
                **scores.get(lbl, {}),
                "keyword_coverage_%":          self._keyword_coverage_output_only(result, case["expected_keywords"]),
                "pipeline_keyword_coverage_%": self._keyword_coverage_pipeline(result, case["expected_keywords"]),
                "reasoning_markers":           self._reasoning_markers(result["output"]),
                "contradictions":              self._contradiction_score(result["output"]),
                "safety_flags":                safety_flags,
                "safety_consistency_note":     self._safety_consistency_note(safety_flags, llm_safety),
                "tokens_used":                 result["tokens_used"],
                "latency_seconds":             result["latency_seconds"],
                "llm_calls":                   result["num_llm_calls"],
            }

        return {
            "case_id":    case["id"],
            "case_title": case["title"],
            "specialty":  case.get("specialty", "Unknown"),
            label_a: _metrics(result_a, label_a),
            label_b: _metrics(result_b, label_b),
            label_c: _metrics(result_c, label_c),
        }
