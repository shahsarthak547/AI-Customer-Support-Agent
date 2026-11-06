
import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


KNOWLEDGE_PATH = Path("src/knowledge/knowledge_base.json")

MIN_KNOWLEDGE_SIMILARITY = 0.10
MIN_KEYWORD_MATCH = 0.60
TOP_K_KNOWLEDGE = 3

WORD_PATTERN = re.compile(r"\b[a-z0-9]+\b", re.IGNORECASE)
GENERIC_KEYWORD_TOKENS = {
    "iphone",
    "ipad",
    "battery",
    "app",
    "apps",
}
def normalize_word(word):
    word = word.lower()

    if len(word) > 5 and word.endswith("ing"):
        word = word[:-3]

        # Handle doubled consonants: running -> run
        if len(word) > 2 and word[-1] == word[-2]:
            word = word[:-1]

    elif len(word) > 4 and word.endswith("ed"):
        word = word[:-2]

    elif len(word) > 3 and word.endswith("s"):
        word = word[:-1]

    return word
def tokenize(text):
    words = WORD_PATTERN.findall(text.lower())
    return {normalize_word(word) for word in words}



def calculate_keyword_match(query, keywords):
    """
    Measures overlap using meaningful keyword tokens.
    Generic product words cannot establish topic relevance.
    """
    query_tokens = tokenize(query)

    if not query_tokens:
        return 0.0

    best_score = 0.0

    for keyword in keywords:
        keyword_tokens = tokenize(keyword)

        meaningful_tokens = (
            keyword_tokens - GENERIC_KEYWORD_TOKENS
        )

        if not meaningful_tokens:
            continue

        matched_tokens = query_tokens.intersection(
            meaningful_tokens
        )

        score = len(matched_tokens) / len(meaningful_tokens)

        best_score = max(best_score, score)

    return best_score

def load_knowledge():
    if not KNOWLEDGE_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {KNOWLEDGE_PATH}"
        )

    with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    entries = data.get("entries", [])

    if not isinstance(entries, list):
        raise ValueError("Knowledge entries must be a list.")

    verified_entries = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("verification_status") == "verified"
    ]

    if not verified_entries:
        return {
            "entries": [],
            "vectorizer": None,
            "matrix": None,
        }

    # Only index product, topic and keywords.
    # Do not use troubleshooting guidance to establish relevance.
    documents = []

    for entry in verified_entries:
        keywords = entry.get("keywords", [])

        if not isinstance(keywords, list):
            keywords = []

        document = " ".join([
            str(entry.get("product", "")),
            str(entry.get("topic", "")),
            " ".join(
                keyword for keyword in keywords
                if isinstance(keyword, str)
            ),
        ])

        documents.append(document)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        stop_words="english",
    )

    matrix = vectorizer.fit_transform(documents)

    return {
        "entries": verified_entries,
        "vectorizer": vectorizer,
        "matrix": matrix,
    }


def search_knowledge(query, knowledge):
    if not isinstance(query, str) or not query.strip():
        return []

    if not knowledge["entries"]:
        return []

    query_vector = knowledge["vectorizer"].transform([query])

    similarities = cosine_similarity(
        query_vector,
        knowledge["matrix"],
    )[0]

    ranked_indices = similarities.argsort()[::-1]

    results = []

    for index in ranked_indices:
        similarity = float(similarities[index])

        if similarity < MIN_KNOWLEDGE_SIMILARITY:
            continue

        entry = knowledge["entries"][index]

        keywords = entry.get("keywords", [])

        keyword_match = calculate_keyword_match(
            query,
            keywords if isinstance(keywords, list) else [],
        )

        # A generic similarity score is not enough.
        # Require a meaningful match with an approved keyword.
        if keyword_match < MIN_KEYWORD_MATCH:
            continue

        results.append({
            "id": entry["id"],
            "product": entry["product"],
            "intent": entry["intent"],
            "topic": entry["topic"],
            "keywords": keywords,
            "guidance": entry["guidance"],
            "source": entry["source"],
            "verification_status": entry["verification_status"],
            "similarity": similarity,
            "keyword_match": keyword_match,
        })

        if len(results) >= TOP_K_KNOWLEDGE:
            break

    return results
