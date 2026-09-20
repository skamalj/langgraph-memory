"""Engine behaviour on LangGraph's InMemoryStore with a deterministic embedder and rule-based extractor.

No LLM, no network. The same assertions run against real backends in test_backends_e2e.py.
"""
from datetime import datetime, timedelta, timezone

import pytest
from langgraph.store.memory import InMemoryStore
from langgraph_store_core.testing import FakeEmbeddings

from langgraph_memory import ConsolidationDecision, ExtractedFact, MemoryConfig, MemoryEngine, MemoryRecord

DIMS = 256
EMB = FakeEmbeddings(DIMS)


def rule_extractor(text: str):
    """One fact per non-empty line that contains 'likes', 'lives', 'prefers' or 'moved'; importance by keyword."""
    facts = []
    for line in text.splitlines():
        body = line.split(":", 1)[-1].strip()
        if any(k in body.lower() for k in ("likes", "lives", "prefers", "moved", "hates")):
            imp = 0.9 if "never" in body.lower() or "hates" in body.lower() else 0.5
            cats = ["preference"] if any(k in body.lower() for k in ("likes", "prefers", "hates")) else ["personal"]
            facts.append(ExtractedFact(content=body, categories=cats, importance=imp))
    return facts


def rule_consolidator(fact, similar):
    """Update the most similar memory when it shares the subject word, else insert."""
    top, sim = similar[0]
    if fact.content.split()[0].lower() == top.content.split()[0].lower():
        return ConsolidationDecision(action="update", target_key=top.key, content=fact.content)
    return ConsolidationDecision(action="insert")


def make_store():
    return InMemoryStore(index={"dims": DIMS, "embed": EMB, "fields": ["content"]})


@pytest.fixture()
def engine():
    return MemoryEngine(make_store(), extractor=rule_extractor, consolidator=rule_consolidator,
                        config=MemoryConfig(consolidation_threshold=0.6))


NS = ("memories", "kamal")


def test_requires_indexed_store():
    with pytest.raises(ValueError):
        MemoryEngine(InMemoryStore(), extractor=rule_extractor, consolidator=rule_consolidator)


def test_requires_model_or_callables():
    with pytest.raises(ValueError):
        MemoryEngine(make_store())


def test_remember_extracts_and_stores(engine):
    written = engine.remember("user: hi there\nuser: kamal likes sushi\nai: noted\nuser: kamal lives in hanoi", NS)
    assert [w.content for w in written] == ["kamal likes sushi", "kamal lives in hanoi"]
    assert written[0].categories == ["preference"] and written[1].categories == ["personal"]
    stored = engine.list(NS)
    assert {r.content for r in stored} == {"kamal likes sushi", "kamal lives in hanoi"}
    assert all(r.namespace == NS for r in stored)


def test_remember_accepts_messages(engine):
    from langchain_core.messages import AIMessage, HumanMessage
    written = engine.remember([HumanMessage("kamal prefers window seats"), AIMessage("ok"),
                               {"role": "user", "content": "kamal moved to hanoi"}], NS)
    assert {w.content for w in written} == {"kamal prefers window seats", "kamal moved to hanoi"}


def test_remember_verbatim_without_extraction(engine):
    written = engine.remember("raw note about the account", NS, extract=False, categories=["note"], importance=0.7)
    assert len(written) == 1 and written[0].categories == ["note"] and written[0].importance == 0.7


def test_batch_dedupe(engine):
    written = engine.remember("user: kamal likes sushi\nuser: Kamal likes sushi!", NS)
    assert len(written) == 1


def test_consolidation_skip_exact_duplicate(engine):
    engine.remember("user: kamal likes sushi", NS)
    again = engine.remember("user: kamal likes sushi", NS)
    assert again == [] and len(engine.list(NS)) == 1


def test_consolidation_update_same_subject(engine):
    first = engine.remember("user: kamal likes sushi", NS)[0]
    updated = engine.remember("user: kamal likes sushi and ramen", NS)
    assert len(updated) == 1 and updated[0].key == first.key          # updated in place, not duplicated
    assert updated[0].content == "kamal likes sushi and ramen"
    assert len(engine.list(NS)) == 1


def test_consolidation_disabled_inserts_everything():
    e = MemoryEngine(make_store(), extractor=rule_extractor, consolidator=rule_consolidator,
                     config=MemoryConfig(consolidation_threshold=1.0))
    e.remember("user: kamal likes sushi", NS)
    e.remember("user: kamal likes sushi and ramen", NS)
    assert len(e.list(NS)) == 2


def test_recall_ranks_by_similarity_and_sets_reasons(engine):
    engine.remember("user: kamal likes sushi\nuser: kamal lives in hanoi\nuser: kamal prefers late flights", NS)
    matches = engine.recall("sushi japanese food", NS, limit=2)
    assert matches[0].record.content == "kamal likes sushi"
    assert 0.0 <= matches[0].score <= 1.0 and matches[0].similarity > matches[-1].similarity
    assert "semantic" in matches[0].match_reasons and "recency" in matches[0].match_reasons


def test_recall_namespace_isolation(engine):
    engine.remember("user: kamal likes sushi", NS)
    engine.remember("user: priya likes sushi", ("memories", "priya"))
    assert all(m.record.namespace == NS for m in engine.recall("sushi", NS))
    assert engine.recall("sushi", ("memories", "nobody")) == []


def test_recall_categories_and_min_score(engine):
    engine.remember("user: kamal likes sushi\nuser: kamal lives in hanoi", NS)
    assert all("preference" in m.record.categories for m in engine.recall("kamal", NS, categories=["preference"]))
    assert engine.recall("kamal", NS, min_score=0.999) == []


def test_recency_weight_changes_ranking():
    store = make_store()
    e = MemoryEngine(store, extractor=rule_extractor, consolidator=rule_consolidator,
                     config=MemoryConfig(consolidation_threshold=1.0, semantic_weight=0.2, recency_weight=0.8, importance_weight=0.0, recency_half_life_days=1))
    old = e.remember("user: kamal likes sushi", NS)[0]
    old.created_at = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    store.put(NS, old.key, old.to_value(), index=False)
    e.remember("user: kamal likes sushi rolls", NS)
    top = e.recall("kamal likes sushi", NS, limit=2)[0]
    assert top.record.content == "kamal likes sushi rolls"     # newer wins under heavy recency weight
    e2 = MemoryEngine(store, extractor=rule_extractor, consolidator=rule_consolidator,
                      config=MemoryConfig(semantic_weight=1.0, recency_weight=0.0, importance_weight=0.0))
    assert e2.recall("kamal likes sushi", NS, limit=2)[0].record.content == "kamal likes sushi"   # pure semantic


def test_importance_weight_changes_ranking():
    e = MemoryEngine(make_store(), extractor=rule_extractor, consolidator=rule_consolidator,
                     config=MemoryConfig(consolidation_threshold=1.0, semantic_weight=0.1, recency_weight=0.0, importance_weight=0.9))
    e.remember("user: kamal likes tea\nuser: kamal hates tea in the morning", NS)   # 'hates' -> importance 0.9
    assert e.recall("tea", NS)[0].record.importance == 0.9


def test_private_memories_hidden_unless_source_matches(engine):
    engine.remember("user: kamal likes sushi", NS, source="s1", private=True)
    assert engine.recall("sushi", NS) == []
    assert engine.recall("sushi", NS, source="s1") and engine.recall("sushi", NS, include_private=True)


def test_touch_updates_last_accessed_and_keeps_vector(engine):
    rec = engine.remember("user: kamal likes sushi", NS)[0]
    before = rec.last_accessed
    engine.recall("sushi", NS)
    after = engine.list(NS)[0].last_accessed
    assert after >= before
    assert engine.recall("sushi", NS)[0].similarity > 0.5     # still semantically searchable after the metadata-only put


def test_forget_by_key_categories_and_age(engine):
    a = engine.remember("user: kamal likes sushi", NS)[0]
    engine.remember("user: kamal lives in hanoi", NS)
    assert engine.forget(NS, key=a.key) == 1
    assert engine.forget(NS, categories=["personal"]) == 1
    assert engine.list(NS) == []


def test_on_prune_hook_and_namespace_forms(engine):
    engine.on_prune([{"id": "1", "role": "user", "content": "kamal likes sushi"}], "memories/kamal")   # str namespace
    assert engine.recall("sushi", NS)
    with pytest.raises(ValueError):
        engine.on_prune([{"role": "user", "content": "x"}], None)


def test_reducer_integration(engine):
    from agentstate_reducer import MessageReducer, ReducerConfig
    reducer = MessageReducer(config=ReducerConfig(min_messages=2, max_messages=4, preserve_first=False, on_prune=[engine.on_prune]))
    msgs = [{"id": f"m{i}", "role": "user", "content": c} for i, c in enumerate(
        ["kamal likes sushi", "kamal lives in hanoi", "ok", "fine", "kamal prefers late flights"])]
    result = reducer.reduce(existing=msgs, namespace=NS)
    assert len(result.surviving) == 2
    assert {r.content for r in engine.list(NS)} >= {"kamal likes sushi", "kamal lives in hanoi"}


def test_tools_and_recall_node(engine):
    tools = {t.name: t for t in engine.tools()}
    cfg = {"configurable": {"memory_namespace": NS}}
    assert "created" in tools["manage_memory"].invoke({"content": "kamal likes sushi"}, config=cfg)
    assert "sushi" in tools["search_memory"].invoke({"query": "sushi"}, config=cfg)
    node = engine.recall_node(into="memory", limit=3)
    out = node({"messages": [{"role": "user", "content": "what food?"}]}, cfg)
    assert "sushi" in out["memory"]
    assert node({"messages": []}, cfg) == {"memory": ""}


async def test_async_surface(engine):
    await engine.aremember("user: kamal likes sushi", NS)
    assert (await engine.arecall("sushi", NS))[0].record.content == "kamal likes sushi"
