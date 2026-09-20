"""MemoryEngine: LLM extraction, consolidation and ranked recall over any LangGraph BaseStore."""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from .types import (
    KIND,
    ConsolidationDecision,
    ExtractedFact,
    ExtractedFacts,
    MemoryConfig,
    MemoryMatch,
    MemoryRecord,
    composite_score,
    now_iso,
    parse_dt,
)

__all__ = ["MemoryEngine", "LLMExtractor", "LLMConsolidator", "Extractor", "Consolidator", "messages_to_text"]

logger = logging.getLogger(__name__)

Namespace = Union[Tuple[str, ...], List[str], str]
# (text) -> facts
Extractor = Callable[[str], List[ExtractedFact]]
# (new fact, [(record, similarity), ...]) -> decision
Consolidator = Callable[[ExtractedFact, List[Tuple[MemoryRecord, float]]], ConsolidationDecision]

_EXTRACT_PROMPT = (
    "You extract long-term memories from a conversation transcript. Return only durable facts, "
    "preferences, decisions and commitments about the user or the task that a future conversation "
    "would benefit from. Write each as one short, self-contained sentence in the third person. "
    "Skip greetings, transient state, and anything already implied by another fact. "
    "Give short lowercase category tags and an importance from 0 (trivia) to 1 (must never forget)."
)
_CONSOLIDATE_PROMPT = (
    "You maintain a set of long-term memories. Given a NEW fact and the most similar EXISTING memories, decide: "
    '"skip" if the new fact is already captured; "update" (with target_key and merged content) if it corrects or '
    'extends one existing memory; otherwise "insert". Prefer update over insert when the subject is the same.'
)


def _resolve_model(model: Any):
    if isinstance(model, str):
        from langchain.chat_models import init_chat_model

        return init_chat_model(model)
    return model


class LLMExtractor:
    """Default extractor: a chat model with structured output."""

    def __init__(self, model: Any, *, prompt: str = _EXTRACT_PROMPT) -> None:
        self._llm = _resolve_model(model).with_structured_output(ExtractedFacts)
        self._prompt = prompt

    def __call__(self, text: str) -> List[ExtractedFact]:
        out = self._llm.invoke([("system", self._prompt), ("human", text)])
        return list(out.facts) if out else []


class LLMConsolidator:
    """Default consolidator: a chat model decides insert / update / skip against similar memories."""

    def __init__(self, model: Any, *, prompt: str = _CONSOLIDATE_PROMPT) -> None:
        self._llm = _resolve_model(model).with_structured_output(ConsolidationDecision)
        self._prompt = prompt

    def __call__(self, fact: ExtractedFact, similar: List[Tuple[MemoryRecord, float]]) -> ConsolidationDecision:
        existing = "\n".join(f"- key={r.key} (similarity={s:.2f}): {r.content}" for r, s in similar)
        human = f"NEW fact: {fact.content}\n\nEXISTING memories:\n{existing}"
        return self._llm.invoke([("system", self._prompt), ("human", human)])


def messages_to_text(messages: Iterable[Any]) -> str:
    """Render dict or LangChain messages as 'role: content' lines."""
    lines = []
    for m in messages:
        if isinstance(m, dict):
            role, content = m.get("role") or m.get("type") or "user", m.get("content", "")
        else:
            role, content = getattr(m, "type", None) or getattr(m, "role", "user"), getattr(m, "content", "")
        if isinstance(content, list):  # content blocks
            content = " ".join(str(b.get("text", "")) if isinstance(b, dict) else str(b) for b in content)
        if str(content).strip():
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _ns(namespace: Namespace) -> Tuple[str, ...]:
    if isinstance(namespace, str):
        return tuple(p for p in namespace.strip("/").split("/") if p) or (namespace,)
    return tuple(namespace)


_WORD = re.compile(r"[a-z0-9]+")


def _norm(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


class MemoryEngine:
    """Long-term memory over a LangGraph ``BaseStore`` built with an ``IndexConfig``.

    - ``remember`` : text / messages -> LLM extraction -> batch dedupe -> consolidation
      against similar existing memories -> ``store.put``
    - ``recall``   : ``store.search(query=...)`` (native vector search) oversampled, then
      re-ranked by semantic similarity, recency decay and importance (CrewAI's formula)
    - ``on_prune`` : a ready-made hook for ``agentstate-reducer``
    - ``tools()`` / ``recall_node()`` : agent tools and a graph node for injection

    The engine never embeds anything itself; the store does, on ``put`` and on
    ``search(query=...)``. That is why the store **must** be built with an
    ``IndexConfig`` — the constructor refuses one that is not.
    """

    def __init__(
        self,
        store: BaseStore,
        model: Any = None,
        *,
        extractor: Optional[Extractor] = None,
        consolidator: Optional[Consolidator] = None,
        config: Optional[MemoryConfig] = None,
    ) -> None:
        index_config = getattr(store, "index_config", None)
        if not index_config:
            raise ValueError(
                "MemoryEngine needs a store constructed with an IndexConfig "
                '(index={"dims": ..., "embed": ..., "fields": ["content"]}); '
                "without one search(query=...) is filter-only and recall cannot rank."
            )
        if model is None and (extractor is None or consolidator is None):
            raise ValueError("Pass a chat model (name or instance) or explicit extractor/consolidator callables.")
        self.store = store
        self.config = config or MemoryConfig()
        self.config.validate()
        self.extractor: Extractor = extractor or LLMExtractor(model)
        self.consolidator: Consolidator = consolidator or LLMConsolidator(model)

    # ------------------------------------------------------------------ write path
    def extract(self, content: Union[str, Iterable[Any]]) -> List[ExtractedFact]:
        text = content if isinstance(content, str) else messages_to_text(content)
        if not text.strip():
            return []
        return [f for f in self.extractor(text) if f.content and f.content.strip()]

    def remember(
        self,
        content: Union[str, Iterable[Any]],
        namespace: Namespace,
        *,
        categories: Optional[List[str]] = None,
        importance: Optional[float] = None,
        source: Optional[str] = None,
        private: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        extract: bool = True,
    ) -> List[MemoryRecord]:
        """Extract facts from ``content`` (or store it verbatim with ``extract=False``) into ``namespace``."""
        if extract:
            facts = self.extract(content)
        else:
            text = content if isinstance(content, str) else messages_to_text(content)
            facts = [ExtractedFact(content=text, categories=categories or [], importance=importance if importance is not None else self.config.default_importance)] if text.strip() else []
        return self.remember_facts(facts, namespace, categories=categories, importance=importance, source=source, private=private, metadata=metadata)

    def remember_facts(
        self,
        facts: Sequence[Union[ExtractedFact, str]],
        namespace: Namespace,
        *,
        categories: Optional[List[str]] = None,
        importance: Optional[float] = None,
        source: Optional[str] = None,
        private: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryRecord]:
        ns = _ns(namespace)
        # intra-batch dedupe on normalised text
        seen: set = set()
        batch: List[ExtractedFact] = []
        for f in facts:
            fact = f if isinstance(f, ExtractedFact) else ExtractedFact(content=str(f))
            key = _norm(fact.content)
            if not key or key in seen:
                continue
            seen.add(key)
            batch.append(fact)

        written: List[MemoryRecord] = []
        for fact in batch:
            cats = list(fact.categories or categories or [])
            imp = importance if importance is not None else float(fact.importance if fact.importance is not None else self.config.default_importance)
            decision = self._consolidate(fact, ns)
            if decision.action == "skip":
                continue
            if decision.action == "update" and decision.target_key:
                existing = self.store.get(ns, decision.target_key)
                if existing is not None:
                    rec = MemoryRecord.from_item(existing)
                    rec.content = decision.content or fact.content
                    rec.categories = sorted(set(rec.categories) | set(cats))
                    rec.importance = max(rec.importance, imp)
                    rec.last_accessed = now_iso()
                    if metadata:
                        rec.metadata.update(metadata)
                    self.store.put(ns, rec.key, rec.to_value())
                    written.append(rec)
                    continue
            rec = MemoryRecord(
                key=uuid.uuid4().hex, namespace=ns, content=decision.content or fact.content,
                categories=cats, importance=imp, source=source, private=private, metadata=dict(metadata or {}),
            )
            self.store.put(ns, rec.key, rec.to_value())
            written.append(rec)
        return written

    def _consolidate(self, fact: ExtractedFact, ns: Tuple[str, ...]) -> ConsolidationDecision:
        if self.config.consolidation_threshold >= 1.0:
            return ConsolidationDecision(action="insert")
        hits = self.store.search(ns, query=fact.content, filter={"kind": KIND}, limit=self.config.consolidation_limit)
        similar = [(MemoryRecord.from_item(h), float(h.score)) for h in hits if h.score is not None and h.score >= self.config.consolidation_threshold]
        if not similar:
            return ConsolidationDecision(action="insert")
        if any(_norm(r.content) == _norm(fact.content) for r, _ in similar):
            return ConsolidationDecision(action="skip")
        try:
            decision = self.consolidator(fact, similar)
        except Exception as exc:  # never lose a memory because the judge failed
            logger.warning("consolidator failed, inserting: %s", exc)
            return ConsolidationDecision(action="insert")
        if decision.action not in ("insert", "update", "skip"):
            decision.action = "insert"
        return decision

    # ------------------------------------------------------------------ read path
    def recall(
        self,
        query: str,
        namespace: Namespace,
        *,
        categories: Optional[List[str]] = None,
        limit: int = 10,
        min_score: float = 0.0,
        source: Optional[str] = None,
        include_private: bool = False,
        filter: Optional[Dict[str, Any]] = None,
        touch: Optional[bool] = None,
    ) -> List[MemoryMatch]:
        """Semantic search via the store, re-ranked by similarity + recency + importance."""
        if not query or not query.strip():
            return []
        ns = _ns(namespace)
        store_filter = {"kind": KIND, **(filter or {})}
        hits = self.store.search(ns, query=query, filter=store_filter, limit=max(1, limit * self.config.recall_oversample))
        matches: List[MemoryMatch] = []
        for h in hits:
            if h.score is None:
                raise RuntimeError("store returned no similarity score — is it built with an IndexConfig?")
            rec = MemoryRecord.from_item(h)
            if categories and not any(c in rec.categories for c in categories):
                continue
            if rec.private and not include_private and rec.source != source:
                continue
            score, reasons = composite_score(rec, float(h.score), self.config)
            if score >= min_score:
                matches.append(MemoryMatch(record=rec, score=score, similarity=float(h.score), match_reasons=reasons))
        matches.sort(key=lambda m: -m.score)
        matches = matches[:limit]
        if (self.config.touch_on_recall if touch is None else touch) and matches:
            self._touch([m.record for m in matches])
        return matches

    def _touch(self, records: List[MemoryRecord]) -> None:
        stamp = now_iso()
        for rec in records:
            rec.last_accessed = stamp
            try:
                self.store.put(rec.namespace, rec.key, rec.to_value(), index=False)   # metadata-only, keeps the vector
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("touch failed for %s: %s", rec.key, exc)

    def list(self, namespace: Namespace, *, limit: int = 200) -> List[MemoryRecord]:
        return [MemoryRecord.from_item(i) for i in self.store.search(_ns(namespace), filter={"kind": KIND}, limit=limit)]

    def forget(
        self,
        namespace: Namespace,
        *,
        key: Optional[str] = None,
        categories: Optional[List[str]] = None,
        older_than: Optional[datetime] = None,
    ) -> int:
        ns = _ns(namespace)
        if key is not None:
            self.store.delete(ns, key)
            return 1
        n = 0
        for rec in self.list(ns, limit=10_000):
            if categories and not any(c in rec.categories for c in categories):
                continue
            if older_than is not None and parse_dt(rec.created_at) >= (older_than if older_than.tzinfo else older_than.replace(tzinfo=timezone.utc)):
                continue
            self.store.delete(rec.namespace, rec.key)
            n += 1
        return n

    # ------------------------------------------------------------------ integrations
    @property
    def on_prune(self) -> Callable[[list, Any], Any]:
        """A ``RememberFn`` for ``agentstate_reducer``: pruned messages -> extracted memories."""

        def hook(pruned: list, namespace: Any) -> None:
            if namespace is None:
                raise ValueError("on_prune received namespace=None; set memory_namespace in the run config/state")
            self.remember(pruned, namespace)

        return hook

    def tools(self, *, namespace_key: str = "memory_namespace") -> list:
        """LangChain tools ``search_memory`` and ``manage_memory``; namespace from ``config['configurable'][namespace_key]``."""
        from langchain_core.tools import tool

        engine = self

        def _ns_from(config: RunnableConfig) -> Tuple[str, ...]:
            ns = (config or {}).get("configurable", {}).get(namespace_key)
            if ns is None:
                raise ValueError(f"config['configurable']['{namespace_key}'] is required for memory tools")
            return _ns(ns)

        @tool
        def search_memory(query: str, config: RunnableConfig, limit: int = 5) -> str:
            """Search long-term memory for facts relevant to the query."""
            matches = engine.recall(query, _ns_from(config), limit=limit)
            return "\n".join(m.format() for m in matches) or "No relevant memories."

        @tool
        def manage_memory(content: str, config: RunnableConfig, action: str = "create", key: Optional[str] = None) -> str:
            """Create ('create'), update (with key) or delete (with key) a long-term memory."""
            ns = _ns_from(config)
            if action == "delete" and key:
                engine.forget(ns, key=key)
                return f"deleted {key}"
            if action == "update" and key:
                item = engine.store.get(ns, key)
                if item is None:
                    return f"unknown key {key}"
                rec = MemoryRecord.from_item(item)
                rec.content = content
                engine.store.put(ns, rec.key, rec.to_value())
                return f"updated {key}"
            written = engine.remember(content, ns, extract=False)
            return f"created {written[0].key}" if written else "nothing stored"

        return [search_memory, manage_memory]

    def recall_node(self, *, into: str = "memory", limit: int = 5, namespace_key: str = "memory_namespace", messages_key: str = "messages"):
        """A LangGraph node: recall for the last human message and write formatted memories into ``state[into]``."""
        engine = self

        def node(state: dict, config: Any = None) -> dict:
            ns = (config or {}).get("configurable", {}).get(namespace_key)
            if ns is None:
                return {into: ""}
            msgs = state.get(messages_key) or []
            query = ""
            for m in reversed(list(msgs)):
                role = m.get("role") if isinstance(m, dict) else getattr(m, "type", None)
                if role in ("human", "user"):
                    query = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
                    break
            if not query:
                return {into: ""}
            matches = engine.recall(str(query), _ns(ns), limit=limit)
            return {into: "\n".join(m.format() for m in matches)}

        return node

    # ------------------------------------------------------------------ async
    async def aremember(self, content, namespace: Namespace, **kw) -> List[MemoryRecord]:
        return await asyncio.to_thread(self.remember, content, namespace, **kw)

    async def arecall(self, query: str, namespace: Namespace, **kw) -> List[MemoryMatch]:
        return await asyncio.to_thread(self.recall, query, namespace, **kw)
