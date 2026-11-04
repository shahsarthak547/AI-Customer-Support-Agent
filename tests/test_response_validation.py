import pytest

from src.agent.response_validator import validate_response


def test_empty_response_is_rejected():
    result = validate_response("", "My battery drains quickly")

    assert result["safe"] is False


def test_whitespace_response_is_rejected():
    result = validate_response("   ", "My battery drains quickly")

    assert result["safe"] is False


def test_invalid_response_type_is_rejected():
    result = validate_response(None, "My battery drains quickly")

    assert result["safe"] is False


def test_twitter_username_is_rejected():
    result = validate_response(
        "Please contact @AppleSupport for help.",
        "My iPhone battery drains quickly",
    )

    assert result["safe"] is False


def test_url_is_rejected():
    result = validate_response(
        "Visit https://example.com for help.",
        "My iPhone battery drains quickly",
    )

    assert result["safe"] is False


def test_unprovided_version_is_rejected():
    result = validate_response(
        "This issue occurs on iOS 18.2.",
        "My phone is running iOS 17.4.",
    )

    assert result["safe"] is False


def test_matching_version_is_allowed():
    result = validate_response(
        "For iOS 17.4, try restarting your device.",
        "My phone is running iOS 17.4.",
    )

    assert result["safe"] is True


def test_general_response_is_allowed():
    result = validate_response(
        "Try restarting your device and checking whether the issue persists.",
        "My iPhone battery drains quickly",
    )

    assert result["safe"] is True


def test_multiple_versions_must_be_supported():
    result = validate_response(
        "This affects iOS 17.4 and iOS 18.2.",
        "My phone is running iOS 17.4.",
    )

    assert result["safe"] is False