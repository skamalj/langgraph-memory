"""Data types for langgraph-memory."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

KIND = "memory"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@dataclass
class MemoryConfig:
    """Scoring, consolidation and recall behaviour (mirrors CrewAI's MemoryConfig)."""

    semantic_weight: float = 0.5
    recency_weight: float = 0.3
    importance_weight: float = 0.2
    recency_half_life_days: float = 30.0
    consolidation_threshold: float = 0.85   # cosine at/above which the consolidator decides insert/update/skip; 1.0 disables
    consolidation_limit: int = 5
    recall_oversample: int = 3
    default_importance: float = 0.5
    touch_on_recall: bool = True            # bump last_accessed on returned records (metadata-only put)

    def validate(self) -> None:
        for name in ("semantic_weight", "recency_weight", "importance_weight", "consolidation_threshold", "default_importance"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"MemoryConfig.{name} must be in [0, 1], got {v}")
        if self.recency_half_life_days <= 0 or self.consolidation_limit < 1 or self.recall_oversample < 1:
            raise ValueError("MemoryConfig: half-life, consolidation_limit and recall_oversample must be positive")


@dataclass
class MemoryRecord:
    """One long-term memory as stored in the BaseStore value."""

    key: str
    namespace: Tuple[str, ...]
    content: str
    categories: List[str] = field(default_factory=list)
    importance: float = 0.5
    created_at: str = field(default_factory=now_iso)
    last_accessed: str = field(default_factory=now_iso)
    source: Optional[str] = None
    private: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_value(self) -> dict:
        return {
            "kind": KIND,
            "content": self.content,
            "categories": list(self.categories),
            "importance": float(self.importance),
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "source": self.source,
            "private": bool(self.private),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_item(cls, item: Any) -> "MemoryRecord":
        v = item.value
        return cls(
            key=item.key, namespace=tuple(item.namespace), content=str(v.get("content", "")),
            categories=list(v.get("categories") or []), importance=float(v.get("importance", 0.5)),
            created_at=v.get("created_at") or now_iso(), last_accessed=v.get("last_accessed") or now_iso(),
            source=v.get("source"), private=bool(v.get("private", False)), metadata=dict(v.get("metadata") or {}),
        )


@dataclass
class MemoryMatch:
    record: MemoryRecord
    score: float                       # composite: w_sem*similarity + w_rec*decay + w_imp*importance
    similarity: float                  # raw cosine from the store
    match_reasons: List[str] = field(default_factory=list)

    def format(self) -> str:
        line = f"- (score={self.score:.2f}) {self.record.content}"
        if self.record.categories:
            line += f"  [{', '.join(self.record.categories)}]"
        return line


def composite_score(record: MemoryRecord, similarity: float, config: MemoryConfig) -> Tuple[float, List[str]]:
    """CrewAI's formula: semantic + recency decay + importance, with match reasons."""
    age_days = max((datetime.now(timezone.utc) - parse_dt(record.created_at)).total_seconds() / 86400.0, 0.0)
    decay = 0.5 ** (age_days / config.recency_half_life_days)
    score = config.semantic_weight * similarity + config.recency_weight * decay + config.importance_weight * record.importance
    reasons = ["semantic"]
    if decay > 0.5:
        reasons.append("recency")
    if record.importance > 0.5:
        reasons.append("importance")
    return score, reasons


# ---------------------------------------------------------------- LLM schemas

class ExtractedFact(BaseModel):
    content: str = Field(description="One self-contained fact, preference, or decision worth remembering, in third person.")
    categories: List[str] = Field(default_factory=list, description="Short lowercase tags, e.g. preference, personal, work, decision.")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="0 = trivia, 1 = must never forget.")


class ExtractedFacts(BaseModel):
    facts: List[ExtractedFact] = Field(default_factory=list)


class ConsolidationDecision(BaseModel):
    action: str = Field(description='One of "insert" (new fact), "update" (replace an existing memory), "skip" (already known).')
    target_key: Optional[str] = Field(default=None, description="Key of the existing memory to update, when action is update.")
    content: Optional[str] = Field(default=None, description="Merged content to store, when action is update or insert.")
