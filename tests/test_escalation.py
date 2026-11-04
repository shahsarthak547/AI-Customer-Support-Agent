import pytest

from src.agent.escalation import (
    contains_risky_request,
    decide_escalation,
)


@pytest.mark.parametrize(
    "message",
    [
        "I want a refund",
        "I was charged twice",
        "Please cancel my subscription",
        "Someone hacked my account",
        "My phone was stolen",
        "I want to take legal action",
        "Someone accessed my account",
        "I need to erase my iPhone",
    ],
)
def test_risky_requests_are_detected(message):
    assert contains_risky_request(message)


@pytest.mark.parametrize(
    "message",
    [
        "My iPhone battery drains quickly",
        "How do I connect to Wi-Fi?",
        "My keyboard is not responding",
    ],
)
def test_normal_requests_are_not_risky(message):
    assert not contains_risky_request(message)


def test_risky_message_escalates():
    result = decide_escalation(
        intent="battery_charging",
        classifier_confidence=0.99,
        retrieved_cases=[
            {"similarity": 0.90},
        ],
        customer_message="I want a refund",
    )

    assert result["escalate"] is True


def test_low_confidence_escalates():
    result = decide_escalation(
        intent="battery_charging",
        classifier_confidence=0.60,
        retrieved_cases=[
            {"similarity": 0.90},
        ],
        customer_message="My battery drains quickly",
    )

    assert result["escalate"] is True


def test_low_retrieval_similarity_escalates():
    result = decide_escalation(
        intent="battery_charging",
        classifier_confidence=0.95,
        retrieved_cases=[
            {"similarity": 0.20},
        ],
        customer_message="My battery drains quickly",
    )

    assert result["escalate"] is True


def test_valid_request_can_be_auto_handled():
    result = decide_escalation(
        intent="battery_charging",
        classifier_confidence=0.95,
        retrieved_cases=[
            {"similarity": 0.80},
        ],
        customer_message="My battery drains quickly",
    )

    assert result["escalate"] is False


def test_missing_retrieval_cases_escalates():
    result = decide_escalation(
        intent="battery_charging",
        classifier_confidence=0.95,
        retrieved_cases=[],
        customer_message="My battery drains quickly",
    )

    assert result["escalate"] is True


def test_invalid_confidence_escalates():
    result = decide_escalation(
        intent="battery_charging",
        classifier_confidence="invalid",
        retrieved_cases=[
            {"similarity": 0.80},
        ],
        customer_message="My battery drains quickly",
    )

    assert result["escalate"] is True