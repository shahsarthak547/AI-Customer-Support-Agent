
import re


WORD_PATTERN = re.compile(r"\b[a-z0-9]+\b", re.IGNORECASE)

SENTENCE_PATTERN = re.compile(r"(?<=[.!?:])\s+")

# Sentences containing these terms are treated as potentially
# technical and therefore require grounding.
TECHNICAL_TERMS = {
    "settings",
    "battery",
    "iphone",
    "ipad",
    "ios",
    "app",
    "apps",
    "background",
    "activity",
    "capacity",
    "maximum",
    "charging",
    "power",
    "mode",
    "cellular",
    "signal",
    "notifications",
    "service",
    "recommendation",
    "refresh",
    "toggle",
    "disable",
    "enable",
    "turn",
    "tap",
    "open",
    "select",
    "review",
    "check",
    "reduce",
    "consumption",
    "usage",
}

# Short acknowledgements and conversational phrases do not
# need to match technical knowledge.
NON_TECHNICAL_STARTS = (
    "i understand",
    "i'm sorry",
    "sorry to hear",
    "thanks for",
    "thank you",
    "i can help",
)


def normalize_tokens(text):
    return set(
        WORD_PATTERN.findall(text.lower())
    )


def is_technical_sentence(sentence):
    normalized = sentence.lower().strip()

    if normalized.startswith(NON_TECHNICAL_STARTS):
        return False

    words = normalize_tokens(normalized)

    return bool(words.intersection(TECHNICAL_TERMS))


def calculate_overlap(response_sentence, guidance_sentence):
    response_tokens = normalize_tokens(response_sentence)
    guidance_tokens = normalize_tokens(guidance_sentence)

    if not response_tokens or not guidance_tokens:
        return 0.0

    intersection = response_tokens.intersection(
        guidance_tokens
    )

    # Measures how much of the response wording is supported
    # by the guidance wording.
    return len(intersection) / len(response_tokens)


def validate_grounding(
    response,
    verified_knowledge,
    minimum_overlap=0.35,
):
    if not isinstance(response, str) or not response.strip():
        return {
            "grounded": False,
            "reason": "The response is empty or invalid.",
            "unsupported_sentences": [],
        }

    if not isinstance(verified_knowledge, list):
        return {
            "grounded": False,
            "reason": "Verified knowledge must be a list.",
            "unsupported_sentences": [],
        }

    guidance_statements = []

    for item in verified_knowledge:
        if not isinstance(item, dict):
            continue

        if item.get("verification_status") not in (
            None,
            "verified",
        ):
            continue

        guidance = item.get("guidance", [])

        if not isinstance(guidance, list):
            continue

        for statement in guidance:
            if (
                isinstance(statement, str)
                and statement.strip()
            ):
                guidance_statements.append(
                    statement.strip()
                )

    if not guidance_statements:
        return {
            "grounded": False,
            "reason": "No verified guidance is available.",
            "unsupported_sentences": [],
        }

    # Remove numbered-list markers before splitting sentences.
    cleaned_response = re.sub(
        r"(?<!\w)\d+[.)]\s*",
        "",
        response.strip()
    )

    sentences = [
        sentence.strip()
        for sentence in SENTENCE_PATTERN.split(cleaned_response)
        if sentence.strip()
    ]

    # Ignore introductory headings that end with a colon.
    sentences = [
        sentence
        for sentence in sentences
        if not sentence.endswith(":")
    ]
    unsupported_sentences = []

    for sentence in sentences:
        if not is_technical_sentence(sentence):
            continue

        best_overlap = max(
            calculate_overlap(sentence, guidance)
            for guidance in guidance_statements
        )

        if best_overlap < minimum_overlap:
            unsupported_sentences.append({
                "sentence": sentence,
                "best_overlap": round(best_overlap, 3),
            })

    if unsupported_sentences:
        return {
            "grounded": False,
            "reason": (
                "One or more technical statements could not "
                "be sufficiently matched to verified guidance."
            ),
            "unsupported_sentences": unsupported_sentences,
        }

    return {
        "grounded": True,
        "reason": (
            "Technical statements passed the lexical "
            "grounding check."
        ),
        "unsupported_sentences": [],
    }
