import re
from typing import Any
RISK_PATTERNS = [
    r"\brefunds?\b",
    r"\bchargebacks?\b",
    r"\bcharged\s+(?:me\s+)?twice\b",
    r"\bdouble\s+charged\b",
    r"\bpayment\s+dispute\b",
    r"\bcancel\s+(?:my\s+)?(?:order|subscription)\b",
    r"\bdelete\s+(?:my\s+)?account\b",
    r"\baccount\s+deletion\b",
    r"\berase\s+(?:my\s+)?(?:iphone|ipad|device)\b",
    r"\bfactory\s+reset\b",
    r"\blegal\s+action\b",
    r"\blawsuit\b",
    r"\bpolice\b",
    r"\bstolen\b",
    r"\bhacked\b",
    r"\baccount\s+compromised\b",
    r"\bunauthorized\s+(?:purchase|payment|access)\b",
    r"\bsomeone\s+(?:accessed|used|hacked)\s+my\s+account\b",
]
COMPILED_RISK_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in RISK_PATTERNS
]
def contains_risky_request(message: str) -> bool:
    if not isinstance(message, str):
        return False

    text = message.strip()

    if not text:
        return False

    return any(
        pattern.search(text)
        for pattern in COMPILED_RISK_PATTERNS
    )


def decide_escalation(
    intent: str,
    classifier_confidence: float,
    retrieved_cases: list[dict[str, Any]] | None,
    customer_message: str = "",
    min_classifier_confidence: float = 0.80,
    min_retrieval_similarity: float = 0.35,
) -> dict[str, Any]:

    if contains_risky_request(customer_message):
        return {
            "escalate": True,
            "reason": "The request involves an action that requires human review.",
        }

    if intent == "other_unclear":
        return {
            "escalate": True,
            "reason": "The message could not be mapped to a supported intent.",
        }

    try:
        confidence = float(classifier_confidence)
    except (TypeError, ValueError):
        return {
            "escalate": True,
            "reason": "Classifier confidence is invalid.",
        }

    if not 0.0 <= confidence <= 1.0:
        return {
            "escalate": True,
            "reason": "Classifier confidence is invalid.",
        }

    if confidence < min_classifier_confidence:
        return {
            "escalate": True,
            "reason": "Classifier confidence is below the safe handling threshold.",
        }

    if not retrieved_cases:
        return {
            "escalate": True,
            "reason": "No relevant historical support cases were retrieved.",
        }

    try:
        top_similarity = float(retrieved_cases[0]["similarity"])
    except (KeyError, IndexError, TypeError, ValueError):
        return {
            "escalate": True,
            "reason": "The retrieved support evidence is invalid.",
        }

    if not 0.0 <= top_similarity <= 1.0:
        return {
            "escalate": True,
            "reason": "The retrieved support evidence is invalid.",
        }

    if top_similarity < min_retrieval_similarity:
        return {
            "escalate": True,
            "reason": "Retrieved historical cases are not sufficiently similar.",
        }

    return {
        "escalate": False,
        "reason": (
            "The intent is confident, relevant historical evidence is available, "
            "and no human-review trigger was detected."
        ),
    }