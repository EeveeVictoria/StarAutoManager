"""Data models for StarAutoManager."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional


class Confidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def _order(self) -> int:
        return {Confidence.LOW: 0, Confidence.MEDIUM: 1, Confidence.HIGH: 2}[self]

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Confidence):
            return NotImplemented
        return self._order >= other._order

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Confidence):
            return NotImplemented
        return self._order > other._order

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Confidence):
            return NotImplemented
        return self._order <= other._order

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Confidence):
            return NotImplemented
        return self._order < other._order


@dataclass
class Repository:
    """A GitHub repository."""

    node_id: str
    name: str
    full_name: str  # owner/repo
    description: str = ""
    url: str = ""
    language: str = ""
    topics: list[str] = field(default_factory=list)
    stargazer_count: int = 0
    fork_count: int = 0
    is_archived: bool = False
    starred_at: Optional[str] = None
    pushed_at: Optional[str] = None
    updated_at: Optional[str] = None
    license_id: Optional[str] = None
    owner: str = ""
    readme_snippet: Optional[str] = None
    # List memberships (list node IDs this repo belongs to)
    list_ids: list[str] = field(default_factory=list)

    def summary(self, include_readme: bool = False) -> str:
        """Generate a concise summary for LLM consumption."""
        parts = [f"- **{self.full_name}**"]
        if self.description:
            parts.append(f": {self.description}")
        meta = []
        if self.language:
            meta.append(self.language)
        if self.topics:
            meta.append(f"topics: {', '.join(self.topics[:8])}")
        if self.stargazer_count:
            meta.append(f"{self.stargazer_count:,}★")
        if self.is_archived:
            meta.append("ARCHIVED")
        if meta:
            parts.append(f" ({', '.join(meta)})")
        if include_readme and self.readme_snippet:
            snippet = self.readme_snippet[:300].replace("\n", " ")
            parts.append(f"\n  README: {snippet}...")
        return "".join(parts)

    @property
    def days_since_starred(self) -> Optional[int]:
        if not self.starred_at:
            return None
        starred = datetime.fromisoformat(self.starred_at.replace("Z", "+00:00"))
        return (datetime.now(starred.tzinfo) - starred).days

    @property
    def days_since_pushed(self) -> Optional[int]:
        if not self.pushed_at:
            return None
        pushed = datetime.fromisoformat(self.pushed_at.replace("Z", "+00:00"))
        return (datetime.now(pushed.tzinfo) - pushed).days


@dataclass
class StarList:
    """A GitHub Star List."""

    node_id: str
    name: str
    description: str = ""
    slug: str = ""
    repos: list[Repository] = field(default_factory=list)

    @property
    def repo_count(self) -> int:
        return len(self.repos)

    def summary_for_llm(self, max_repos: int = 15) -> str:
        """Generate list summary for LLM context."""
        lines = [f'### "{self.name}"']
        if self.description:
            lines.append(f"Description: {self.description}")
        lines.append(f"Contains {self.repo_count} repos:")
        for repo in self.repos[:max_repos]:
            lines.append(repo.summary())
        if self.repo_count > max_repos:
            lines.append(f"  ... and {self.repo_count - max_repos} more")
        return "\n".join(lines)


@dataclass
class Categorization:
    """A single categorization decision."""

    repo_full_name: str
    repo_node_id: str
    list_name: str
    reason: str = ""
    confidence: Confidence = Confidence.MEDIUM
    is_new_list: bool = False
    new_list_description: str = ""

    def to_dict(self) -> dict:
        return {
            "repo": self.repo_full_name,
            "list": self.list_name,
            "reason": self.reason,
            "confidence": self.confidence.value,
            "new_list": self.is_new_list,
            "new_list_description": self.new_list_description,
        }


@dataclass
class StaleRepo:
    """A repo detected as potentially stale."""

    repo: Repository
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return f"- {self.repo.full_name}: {'; '.join(self.reasons)}"


@dataclass
class DuplicateGroup:
    """A group of repos that may serve similar purposes."""

    description: str
    repos: list[Repository] = field(default_factory=list)


@dataclass
class RunReport:
    """Complete report of a single run."""

    timestamp: str = ""
    total_starred: int = 0
    total_lists: int = 0
    total_uncategorized: int = 0
    categorizations: list[Categorization] = field(default_factory=list)
    new_lists_created: list[str] = field(default_factory=list)
    stale_repos: list[StaleRepo] = field(default_factory=list)
    duplicate_groups: list[DuplicateGroup] = field(default_factory=list)
    language_stats: dict[str, int] = field(default_factory=dict)
    topic_stats: dict[str, int] = field(default_factory=dict)
    list_health: dict[str, str] = field(default_factory=dict)  # list_name -> warning
    errors: list[str] = field(default_factory=list)
    dry_run: bool = False

    @property
    def applied_count(self) -> int:
        return len([c for c in self.categorizations if not self.dry_run])

    @property
    def high_confidence_count(self) -> int:
        return len([c for c in self.categorizations if c.confidence == Confidence.HIGH])

    @property
    def low_confidence_count(self) -> int:
        return len([c for c in self.categorizations if c.confidence == Confidence.LOW])


@dataclass
class CacheEntry:
    """Cached categorization for a repo."""

    repo_full_name: str
    list_name: str
    categorized_at: str
    confidence: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Cache:
    """Persistent cache of previous categorizations."""

    version: int = 1
    last_run: str = ""
    entries: dict[str, CacheEntry] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "last_run": self.last_run,
            "entries": {k: v.to_dict() for k, v in self.entries.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Cache":
        entries = {}
        for k, v in data.get("entries", {}).items():
            entries[k] = CacheEntry(**v)
        return cls(
            version=data.get("version", 1),
            last_run=data.get("last_run", ""),
            entries=entries,
        )

    @classmethod
    def load(cls, path: str) -> "Cache":
        try:
            with open(path, "r") as f:
                return cls.from_dict(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            return cls()

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def is_cached(self, repo_full_name: str) -> bool:
        return repo_full_name in self.entries

    def add(self, categorization: Categorization) -> None:
        self.entries[categorization.repo_full_name] = CacheEntry(
            repo_full_name=categorization.repo_full_name,
            list_name=categorization.list_name,
            categorized_at=datetime.utcnow().isoformat(),
            confidence=categorization.confidence.value,
        )
