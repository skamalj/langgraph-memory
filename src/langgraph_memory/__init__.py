"""langgraph-memory: long-term memory engine for LangGraph over any BaseStore.

    from langgraph_memory import MemoryEngine

    engine = MemoryEngine(store, "anthropic:claude-sonnet-5")
    engine.remember(messages, namespace=("memories", user_id))
    engine.recall("what does the user prefer?", namespace=("memories", user_id))
"""

from .engine import (
    Consolidator,
    Extractor,
    LLMConsolidator,
    LLMExtractor,
    MemoryEngine,
    messages_to_text,
)
from .types import (
    ConsolidationDecision,
    ExtractedFact,
    ExtractedFacts,
    MemoryConfig,
    MemoryMatch,
    MemoryRecord,
    composite_score,
)

__all__ = [
    "MemoryEngine",
    "MemoryConfig",
    "MemoryRecord",
    "MemoryMatch",
    "ExtractedFact",
    "ExtractedFacts",
    "ConsolidationDecision",
    "LLMExtractor",
    "LLMConsolidator",
    "Extractor",
    "Consolidator",
    "messages_to_text",
    "composite_score",
]

__version__ = "0.1.0"
