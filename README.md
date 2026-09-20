# langgraph-memory

A **long-term memory engine for LangGraph** over any `BaseStore`: LLM fact extraction, consolidation against what is already known, and recall ranked by **semantic similarity, recency and importance**. A maintained alternative to LangMem (dormant since late 2025), modelled on CrewAI's unified `Memory` engine and built to pair with the [`agentstate-reducer` `on_prune` hook](https://skamalj.github.io/agentstate-reducer/reducer/long-term-memory/) and the [`langgraph-store-*`](https://skamalj.github.io/agentstate-reducer/langgraph/stores/) backends.

```bash
pip install langgraph-memory
```

```python
from langgraph_memory import MemoryEngine
from langgraph_store_postgres import PostgresStore
from langgraph_store_core import bedrock_titan_embeddings

store = PostgresStore(url, index={"dims": 1024, "embed": bedrock_titan_embeddings(dimensions=1024), "fields": ["content"]})
engine = MemoryEngine(store, "anthropic:claude-sonnet-5")

engine.remember(messages, namespace=("memories", "kamal"))                 # extract → dedupe → consolidate → put
for m in engine.recall("what does the user prefer for travel?", namespace=("memories", "kamal"), limit=5):
    print(m.score, m.record.content, m.match_reasons)
```

## What it does

| Stage | Behaviour |
|---|---|
| **Extract** | a chat model with structured output turns text or messages into discrete facts, each with categories and an importance in 0–1. Any model `init_chat_model` accepts, or a model instance; or pass your own `extractor` callable. |
| **Dedupe** | near-identical facts within one batch are collapsed |
| **Consolidate** | for each fact, `store.search(query=fact)` finds similar existing memories; above `consolidation_threshold` (0.85) the model decides **insert / update / skip**, so memory converges instead of piling up |
| **Store** | one `BaseStore` item per memory: `{content, categories, importance, created_at, last_accessed, source, private, metadata}`; the store embeds `content` on `put` |
| **Recall** | native vector search, oversampled ×3, then re-ranked with CrewAI's formula `0.5·similarity + 0.3·recency_decay + 0.2·importance` (half-life 30 days); filters by `categories`, `min_score`, `source` / `private`; bumps `last_accessed` |

All weights, the half-life, thresholds and oversampling live in `MemoryConfig`.

## Integrations

```python
from agentstate_reducer import MessageReducer, ReducerConfig, Background
reducer = MessageReducer(config=ReducerConfig(max_messages=20, on_prune=[Background(engine.on_prune)]))
saver = DynamoDBSaver("checkpoints", reducer=reducer)      # pruned turns become memories, under memory_namespace
```

```python
agent = create_react_agent(model, tools=engine.tools())     # search_memory / manage_memory, namespace from config["configurable"]["memory_namespace"]
graph.add_node("recall", engine.recall_node(into="memory", limit=5))   # writes formatted memories into state["memory"] before the model call
```

`engine.forget(namespace, key=... | categories=... | older_than=...)` and `engine.list(namespace)` for housekeeping; `aremember` / `arecall` for async graphs.

## Embedding

The engine never embeds anything. The store does, on `put` and on `search(query=...)`, using the `IndexConfig` it was built with — so one embedder serves writes, consolidation lookups and recall, and each backend uses its native vector engine (DynamoDB `SearchVectors`, pgvector, Cosmos `VectorDistance`, Firestore `find_nearest`). The constructor refuses a store without an `IndexConfig`, because recall could not rank on it. Dimensions are fixed at index creation on every backend; changing the embedding model means a new table and re-embedding.

## Testing

Deterministic and offline: LangGraph's `InMemoryStore` with `langgraph_store_core.testing.FakeEmbeddings`, and rule-based extractor/consolidator callables instead of a model. Backend e2e tests run the same suite against `langgraph-store-postgres` (pgvector) and `langgraph-store-dynamodb`, plus a real `DynamoDBSaver` + reducer `on_prune` flow.

Docs: <https://skamalj.github.io/agentstate-reducer/>

## License

MIT
