"""API-call budget tracker (INV-3, SPEC §4.3 / IMPLEMENTATION §3).

Levels mirror the evaluation cascade:
  mut | shard | full | anchor | val | audit | mutator | e0_infer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from prime.config import BudgetCfg


BUDGET_LEVELS = (
    "mut",
    "shard",
    "full",
    "anchor",
    "val",
    "audit",
    "mutator",
    "e0_infer",
    "other",
)


class BudgetExhausted(RuntimeError):
    """Raised when total_calls budget is exhausted and on_exhausted=stop."""


@dataclass
class TokenTracker:
    """Charges every API call to a named cascade level (INV-3)."""

    total_calls_limit: int = 60_000
    on_exhausted: str = "stop"
    calls_by_level: Dict[str, int] = field(default_factory=dict)
    tokens_by_level: Dict[str, int] = field(default_factory=dict)
    total_calls: int = 0
    total_tokens: int = 0
    scorer_calls: int = 0
    optimizer_calls: int = 0
    scorer_calls_limit: Optional[int] = None
    optimizer_calls_limit: Optional[int] = None
    exhausted: bool = False
    log: List[Dict[str, object]] = field(default_factory=list)

    @classmethod
    def from_cfg(cls, cfg: Optional[BudgetCfg] = None) -> "TokenTracker":
        cfg = cfg or BudgetCfg()
        scorer_lim = cfg.scorer_calls if cfg.scorer_calls is not None else int(cfg.total_calls)
        return cls(
            total_calls_limit=int(cfg.total_calls),
            on_exhausted=str(cfg.on_exhausted),
            scorer_calls_limit=int(scorer_lim) if scorer_lim is not None else None,
            optimizer_calls_limit=int(cfg.optimizer_calls) if cfg.optimizer_calls is not None else None,
        )

    def charge(
        self,
        level: str,
        n_calls: int = 1,
        tokens: int = 0,
        *,
        note: str = "",
        kind: str = "scorer",  # scorer | optimizer | other
    ) -> None:
        if level not in BUDGET_LEVELS:
            level = "other"
        n_calls = max(0, int(n_calls))
        tokens = max(0, int(tokens))
        if n_calls == 0 and tokens == 0:
            return

        remaining = self.total_calls_limit - self.total_calls
        if n_calls > remaining and self.on_exhausted == "stop":
            self.exhausted = True
            self.log.append(
                {
                    "event": "budget_exhausted",
                    "level": level,
                    "requested": n_calls,
                    "remaining": remaining,
                    "note": note,
                }
            )
            raise BudgetExhausted(
                f"Budget exhausted: need {n_calls} calls at '{level}', "
                f"remaining {remaining}/{self.total_calls_limit}"
            )

        if kind == "scorer" and self.scorer_calls_limit is not None:
            if self.scorer_calls + n_calls > self.scorer_calls_limit and self.on_exhausted == "stop":
                self.exhausted = True
                raise BudgetExhausted(
                    f"Scorer budget exhausted: need {n_calls}, "
                    f"used {self.scorer_calls}/{self.scorer_calls_limit}"
                )
        if kind == "optimizer" and self.optimizer_calls_limit is not None:
            if self.optimizer_calls + n_calls > self.optimizer_calls_limit and self.on_exhausted == "stop":
                self.exhausted = True
                raise BudgetExhausted(
                    f"Optimizer budget exhausted: need {n_calls}, "
                    f"used {self.optimizer_calls}/{self.optimizer_calls_limit}"
                )

        self.calls_by_level[level] = self.calls_by_level.get(level, 0) + n_calls
        self.tokens_by_level[level] = self.tokens_by_level.get(level, 0) + tokens
        self.total_calls += n_calls
        self.total_tokens += tokens
        if kind == "scorer":
            self.scorer_calls += n_calls
        elif kind == "optimizer":
            self.optimizer_calls += n_calls
        self.log.append(
            {
                "event": "charge",
                "level": level,
                "kind": kind,
                "n_calls": n_calls,
                "tokens": tokens,
                "total_calls": self.total_calls,
                "scorer_calls": self.scorer_calls,
                "optimizer_calls": self.optimizer_calls,
                "note": note,
            }
        )
        if self.total_calls >= self.total_calls_limit:
            self.exhausted = True

    def remaining(self) -> int:
        return max(0, self.total_calls_limit - self.total_calls)

    def snapshot(self) -> Dict[str, object]:
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_calls_limit": self.total_calls_limit,
            "remaining": self.remaining(),
            "exhausted": self.exhausted,
            "on_exhausted": self.on_exhausted,
            "calls_by_level": dict(self.calls_by_level),
            "tokens_by_level": dict(self.tokens_by_level),
            "scorer_calls": self.scorer_calls,
            "optimizer_calls": self.optimizer_calls,
            "scorer_calls_limit": self.scorer_calls_limit,
            "optimizer_calls_limit": self.optimizer_calls_limit,
        }
