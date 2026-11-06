
import json

from src.knowledge.knowledge_search import (
    load_knowledge,
    search_knowledge,
)


def test_unverified_entries_are_excluded(tmp_path, monkeypatch):
    from src.knowledge import knowledge_search

    data = {
        "entries": [
            {
                "id": "TEST-001",
                "product": "iPhone",
                "intent": "battery_charging",
                "topic": "Battery usage",
                "keywords": ["battery drain"],
                "guidance": ["Check battery usage."],
                "source": "https://example.com",
                "verification_status": "pending_review",
            }
        ]
    }

    path = tmp_path / "knowledge_base.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    monkeypatch.setattr(
        knowledge_search,
        "KNOWLEDGE_PATH",
        path,
    )

    knowledge = load_knowledge()

    assert knowledge["entries"] == []
    assert search_knowledge("battery draining", knowledge) == []


def test_verified_entries_can_be_retrieved(tmp_path, monkeypatch):
    from src.knowledge import knowledge_search

    data = {
        "entries": [
            {
                "id": "TEST-002",
                "product": "iPhone",
                "intent": "battery_charging",
                "topic": "Battery usage",
                "keywords": ["battery drain", "battery usage"],
                "guidance": ["Check battery usage by app."],
                "source": "https://example.com",
                "verification_status": "verified",
            }
        ]
    }

    path = tmp_path / "knowledge_base.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    monkeypatch.setattr(
        knowledge_search,
        "KNOWLEDGE_PATH",
        path,
    )

    knowledge = load_knowledge()

    assert len(knowledge["entries"]) == 1

    results = search_knowledge(
        "My iPhone battery is draining",
        knowledge,
    )

    assert len(results) == 1
    assert results[0]["id"] == "TEST-002"
