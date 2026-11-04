import re


USERNAME_PATTERN = re.compile(r"@\w+")
URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)
VERSION_PATTERN = re.compile(
    r"\b(?:iOS|iPhone|iPad)\s+\d+(?:\.\d+)+\b",
    re.IGNORECASE,
)


def validate_response(response, customer_message):
    if not isinstance(response, str) or not response.strip():
        return {
            "safe": False,
            "reason": "The generated response is empty or invalid.",
        }

    text = response.strip()
    customer_text = (
        customer_message
        if isinstance(customer_message, str)
        else ""
    )

    if USERNAME_PATTERN.search(text):
        return {
            "safe": False,
            "reason": "The response contains a Twitter username artifact.",
        }

    if URL_PATTERN.search(text):
        return {
            "safe": False,
            "reason": "The response contains an unverified URL.",
        }

    response_versions = {
        match.group(0).lower()
        for match in VERSION_PATTERN.finditer(text)
    }

    customer_versions = {
        match.group(0).lower()
        for match in VERSION_PATTERN.finditer(customer_text)
    }

    if not response_versions.issubset(customer_versions):
        return {
            "safe": False,
            "reason": (
                "The response introduces a specific version "
                "not provided by the customer."
            ),
        }

    return {
        "safe": True,
        "reason": (
            "The generated response passed the basic "
            "safety and artifact checks."
        ),
    }