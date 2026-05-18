"""
utils.py — Shared utilities for the Healthcare AI evaluation pipeline.

Extracted from single_agent.py / multi_agent.py / evaluator.py to eliminate
the copy-paste of _call_with_retry across three files. Import from here instead.

FIX (original): DRY — _call_with_retry was copy-pasted identically in 3 modules.
FIX (crash):    424 "Failed Dependency" from Bedrock was classified as non-retryable
                and crashed the run at Case 6 Run 3. Root cause: the retryable check
                only searched exc message text, but "connection error" does not match
                "APIConnectionError" (no space), and "unexpected error" / "424" /
                "failed dependency" were absent from the pattern list entirely.
                Fix: expanded pattern list + also checks the exception class name.
"""

import time
import random
import logging

logger = logging.getLogger(__name__)

# Substrings checked against BOTH the exception message AND the exception class name.
# Add new patterns here whenever a new Bedrock transient error surfaces.
_BEDROCK_RETRYABLE = (
    # Throttling / quota
    "throttlingexception",
    "too many requests",
    "rate exceeded",
    # Service availability
    "serviceunavailable",
    "service unavailable",
    "modeltimeoutexception",
    "model timeout",
    # Server-side transient errors — these are safe to retry per AWS docs
    "internal server error",
    "unexpected error",          # FIX: "The system encountered an unexpected error"
    "failed dependency",         # FIX: 424 Failed Dependency (Bedrock upstream fault)
    "502", "503", "504",         # Gateway / upstream timeout HTTP codes
    # Connection-level errors
    "connection",                # FIX: covers "APIConnectionError", "ConnectionError", etc.
                                 #      (original "connection error" missed "APIConnectionError"
                                 #       because the class name has no space)
    "timeout",
    "read timeout",
    "connect timeout",
)

# Exception class name substrings that are always retryable regardless of message.
_RETRYABLE_EXC_TYPES = (
    "connectionerror",
    "apiconnectionerror",
    "timeout",
    "bedrockexception",          # LiteLLM wraps transient Bedrock faults in this
)


def _is_retryable(exc: Exception) -> bool:
    """
    Returns True if the exception represents a transient error safe to retry.
    Checks both the exception message and the exception class name, because
    some LiteLLM wrappers put the useful signal in the class name, not the msg.
    """
    msg       = str(exc).lower()
    type_name = type(exc).__name__.lower()
    return (
        any(tag in msg       for tag in _BEDROCK_RETRYABLE) or
        any(tag in type_name for tag in _RETRYABLE_EXC_TYPES)
    )


def call_with_retry(fn, max_retries: int = 5, base_delay: float = 5.0):
    """
    Calls fn() with exponential back-off tuned for AWS Bedrock throttling.

    Back-off formula: base_delay * 2^attempt + jitter(0, 1.5)
    max_retries raised 4 → 5 to give one extra chance on 424-class errors.
    base_delay=5 s — Bedrock needs longer recovery windows than Groq.

    Returns (result, retry_count).
    Raises on genuinely non-retryable errors (auth, bad request, etc.).
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            result = fn()
            return result, attempt      # attempt==0 → no retries needed
        except Exception as exc:
            last_exc = exc
            retryable = _is_retryable(exc)

            if retryable and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1.5)
                logger.warning(
                    f"Bedrock transient error (attempt {attempt + 1}/{max_retries}): "
                    f"{type(exc).__name__}: {exc}. Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            elif not retryable:
                logger.error(
                    f"Non-retryable error — will not retry: "
                    f"{type(exc).__name__}: {exc}"
                )
                raise
            else:
                logger.error(
                    f"Retryable error exhausted all {max_retries} attempts: "
                    f"{type(exc).__name__}: {exc}"
                )
                break

    raise last_exc


def is_truncated(text: str) -> bool:
    """
    Heuristic to detect whether an LLM response was cut off by a max_tokens limit.

    A response is considered truncated if it ends without a sentence-terminating
    character, suggesting the model was mid-generation when the token budget ran out.

    FIX: The Verifier output in the original code was visibly cut off mid-word
    ('...Meropen') because max_tokens=800 was insufficient. This check lets callers
    detect and flag that condition rather than silently returning incomplete output.
    """
    stripped = text.strip()
    if not stripped:
        return False
    return stripped[-1] not in {".", "!", "?", ":", "-", "*", "#", "}"}
