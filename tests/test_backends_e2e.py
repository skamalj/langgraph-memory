"""The engine on real stores: Postgres (pgvector) and DynamoDB (native SearchVectors), plus the full
checkpointer -> reducer -> on_prune -> engine -> store -> recall chain with a real LangGraph graph on DynamoDB.

Deterministic embedder and rule-based extraction (no LLM); backends are real.
"""
import os
import time
import uuid
from typing import Annotated, TypedDict

import pytest
from langgraph_store_core.testing import FakeEmbeddings

from langgraph_memory import MemoryConfig, MemoryEngine
from test_engine import rule_consolidator, rule_extractor

DIMS = 256
EMB = FakeEmbeddings(DIMS)
NS = ("memories", "kamal")


def _engine(store):
    return MemoryEngine(store, extractor=rule_extractor, consolidator=rule_consolidator,
                        config=MemoryConfig(consolidation_threshold=0.6))


def _exercise(engine):
    first = engine.remember("user: kamal likes sushi\nuser: kamal lives in hanoi", NS)
    assert len(first) == 2
    upd = engine.remember("user: kamal likes sushi and ramen", NS)          # consolidation -> update in place
    assert len(upd) == 1 and upd[0].key == [w for w in first if "sushi" in w.content][0].key
    assert len(engine.list(NS)) == 2
    hits = engine.recall("sushi japanese food", NS, limit=2)
    assert hits[0].record.content == "kamal likes sushi and ramen" and hits[0].similarity > 0
    assert engine.recall("sushi", NS, categories=["personal"]) and all("personal" in m.record.categories for m in engine.recall("sushi", NS, categories=["personal"]))
    assert engine.recall("sushi", ("memories", "priya")) == []
    assert engine.forget(NS, categories=["preference"]) == 1
    assert engine.forget(NS) == 1 and engine.list(NS) == []


# ── Postgres / pgvector ───────────────────────────────────────────────────────

PG_URL = os.environ.get("SQL_TEST_URL", "postgresql+psycopg2://postgres:postgres@localhost:5433/postgres")


def _pg_available():
    try:
        from sqlalchemy import create_engine, text
        with create_engine(PG_URL).begin() as c:
            c.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _pg_available(), reason="PostgreSQL+pgvector not reachable")
def test_engine_on_postgres_store():
    from sqlalchemy import text
    from langgraph_store_postgres import PostgresStore
    tname = f"lg_memory_test_{uuid.uuid4().hex[:8]}"
    store = PostgresStore(PG_URL, table_name=tname, index={"dims": DIMS, "embed": EMB, "fields": ["content"]})
    try:
        _exercise(_engine(store))
    finally:
        with store._engine.begin() as c:
            c.execute(text(f'DROP TABLE IF EXISTS "{tname}"'))
        store._engine.dispose()


# ── DynamoDB (native SearchVectors) + full checkpointer chain ────────────────

REGION = os.environ.get("LG_DDB_REGION", "us-east-1")


def _aws_available():
    try:
        import boto3
        boto3.client("sts", region_name=REGION).get_caller_identity()
        return True
    except Exception:
        return False


def _wait_vector_index(name):
    import boto3
    ddb = boto3.client("dynamodb", region_name=REGION)
    for _ in range(60):
        vi = ddb.describe_table(TableName=name)["Table"].get("VectorIndexes") or []
        if vi and all(v.get("IndexStatus", "ACTIVE") == "ACTIVE" for v in vi):
            return
        time.sleep(5)


@pytest.mark.skipif(not _aws_available(), reason="AWS credentials not available")
def test_engine_on_dynamodb_store_and_checkpointer_chain():
    import boto3
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from agentstate_reducer import Background, MessageReducer, ReducerConfig
    from langgraph_dynamodb_checkpoint import DynamoDBSaver
    from langgraph_store_dynamodb import DynamoDBStore

    table = f"lg_memory_test_{uuid.uuid4().hex[:8]}"
    store = DynamoDBStore(table, region_name=REGION, index={"dims": DIMS, "embed": EMB, "fields": ["content"]})
    _wait_vector_index(table)
    try:
        engine = _engine(store)
        _exercise(engine)

        # full chain: graph turns -> DynamoDBSaver -> reducer -> on_prune -> engine -> store
        bg = Background(engine.on_prune)
        reducer = MessageReducer(config=ReducerConfig(min_messages=2, max_messages=4, preserve_first=False, on_prune=[bg]))
        saver = DynamoDBSaver(os.environ.get("DDB_TEST_TABLE", "reducer_e2e_test"), reducer=reducer)

        class State(TypedDict):
            messages: Annotated[list, add_messages]

        def echo(state):
            return {"messages": [AIMessage(content="noted")]}

        g = StateGraph(State); g.add_node("echo", echo); g.add_edge(START, "echo"); g.add_edge("echo", END)
        graph = g.compile(checkpointer=saver, store=store)
        cfg = {"configurable": {"thread_id": f"t-{uuid.uuid4().hex[:8]}", "memory_namespace": NS}}
        for turn in ["kamal likes sushi", "kamal lives in hanoi", "kamal prefers late flights", "thanks"]:
            graph.invoke({"messages": [HumanMessage(content=turn)]}, config=cfg)
        bg.close()
        assert bg.dropped == 0
        remembered = {r.content for r in engine.list(NS)}
        assert "kamal likes sushi" in remembered and "kamal lives in hanoi" in remembered
        assert engine.recall("where does the user live", NS, limit=1)[0].record.content == "kamal lives in hanoi"
        saver.delete(cfg)
    finally:
        try:
            boto3.client("dynamodb", region_name=REGION).delete_table(TableName=table)
        except Exception:
            pass
