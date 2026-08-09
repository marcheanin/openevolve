"""Single-worker Scorer + self-consistency uncertainty (ROADMAP_PHASE3 F5/F6)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from prime.config import EnsembleCfg
from prime.workers.ensemble import (
    INVALID,
    LLMWorker,
    build_workers,
    mock_predict,
    parallel_predict,
)


@dataclass
class ScoreResult:
    preds: np.ndarray  # int8, INVALID=-1 on failure
    n_calls: int
    n_invalid: int
    worker_preds: List[List[int]] = field(default_factory=list)

    @property
    def invalid_rate(self) -> float:
        if len(self.preds) == 0:
            return 0.0
        return float(np.mean(np.asarray(self.preds) < 0))


def uncertainty_from_votes(votes: Sequence[int]) -> float:
    """1 - max_vote_share over valid votes. Empty/INVALID-only → 1.0 (max uncertainty)."""
    valid = [int(v) for v in votes if int(v) != INVALID]
    if not valid:
        return 1.0
    counts: dict = {}
    for v in valid:
        counts[v] = counts.get(v, 0) + 1
    return float(1.0 - max(counts.values()) / len(valid))


def self_consistency_uncertainty(
    vote_matrix: Sequence[Sequence[int]],
) -> List[float]:
    """vote_matrix[example][sample] → uncertainty per example."""
    return [uncertainty_from_votes(row) for row in vote_matrix]


class Scorer:
    """One LLM worker + fail-closed parsing. Drop-in for ensemble when mode=single."""

    def __init__(
        self,
        cfg: EnsembleCfg,
        *,
        label_space: Optional[str] = None,
        use_mock: bool = False,
    ) -> None:
        self.cfg = cfg
        self.label_space = label_space or getattr(cfg, "label_space", "ordinal5") or "ordinal5"
        self.use_mock = use_mock
        self.fail_closed = bool(getattr(cfg, "fail_closed", True))
        self._workers: Optional[List[LLMWorker]] = None

    @property
    def n_workers(self) -> int:
        return max(1, len(self.cfg.workers))

    @property
    def is_single(self) -> bool:
        mode = getattr(self.cfg, "mode", "ensemble") or "ensemble"
        return mode == "single" or len(self.cfg.workers) == 1

    def _get_workers(self) -> List[LLMWorker]:
        if self._workers is None:
            # Ensure fail_closed propagates.
            self.cfg.fail_closed = self.fail_closed
            self._workers = build_workers(self.cfg, label_space=self.label_space)
        return self._workers

    def predict_batch(
        self,
        texts: List[str],
        prompt: str,
        *,
        labels_for_mock: Optional[List[int]] = None,
    ) -> ScoreResult:
        if not texts:
            return ScoreResult(preds=np.asarray([], dtype=np.int8), n_calls=0, n_invalid=0)

        if self.use_mock:
            labels = labels_for_mock or [0] * len(texts)
            ens, wp = mock_predict(
                texts,
                labels,
                self.n_workers,
                seed=0,
                aggregation=self.cfg.aggregation,
                label_space=self.label_space,
            )
        else:
            ens, wp = parallel_predict(
                self._get_workers(),
                texts,
                prompt,
                max_parallel=self.cfg.max_parallel,
                tie_break=self.cfg.tie_break,
                aggregation=self.cfg.aggregation,
                label_space=self.label_space,
                fail_closed=self.fail_closed,
            )
        preds = np.asarray(ens, dtype=np.int8)
        return ScoreResult(
            preds=preds,
            n_calls=len(texts) * self.n_workers,
            n_invalid=int(np.sum(preds < 0)),
            worker_preds=wp,
        )

    def self_consistency(
        self,
        texts: List[str],
        prompt: str,
        *,
        k: int = 3,
        temperature: float = 0.7,
        labels_for_mock: Optional[List[int]] = None,
    ) -> Tuple[List[float], int]:
        """
        k stochastic passes → uncertainty per example.
        Returns (uncertainties, n_calls).
        """
        if not texts:
            return [], 0
        if self.use_mock:
            # Synthetic: higher uncertainty when label is odd-index for variety.
            labels = labels_for_mock or list(range(len(texts)))
            unc = [0.0 if (labels[i] % 2 == 0) else (1.0 / 3.0) for i in range(len(texts))]
            return unc, len(texts) * k

        workers = self._get_workers()
        if not workers:
            return [1.0] * len(texts), 0
        worker = workers[0]
        old_temp = worker.temperature
        worker.temperature = float(temperature)
        vote_rows: List[List[int]] = [[] for _ in texts]
        n_calls = 0
        try:
            for _ in range(k):
                for i, text in enumerate(texts):
                    try:
                        worker.fail_closed = self.fail_closed
                        pred = worker.predict(text, prompt)
                    except Exception:
                        pred = INVALID if self.fail_closed else 0
                    vote_rows[i].append(int(pred))
                    n_calls += 1
        finally:
            worker.temperature = old_temp
        return self_consistency_uncertainty(vote_rows), n_calls

    def self_consistency_parallel(
        self,
        texts: List[str],
        prompt: str,
        *,
        k: int = 3,
        temperature: float = 0.7,
        max_parallel: Optional[int] = None,
    ) -> Tuple[List[float], int]:
        """Parallel variant of self_consistency (one worker, many texts × k)."""
        if not texts:
            return [], 0
        if self.use_mock:
            return self.self_consistency(texts, prompt, k=k, temperature=temperature)

        workers = self._get_workers()
        worker = workers[0]
        old_temp = worker.temperature
        worker.temperature = float(temperature)
        worker.fail_closed = self.fail_closed
        mp = max_parallel or self.cfg.max_parallel
        grid: List[List[Optional[int]]] = [[None] * k for _ in texts]
        n_calls = 0

        def _one(t_idx: int, s_idx: int) -> Tuple[int, int, int]:
            try:
                return t_idx, s_idx, int(worker.predict(texts[t_idx], prompt))
            except Exception:
                return t_idx, s_idx, INVALID if self.fail_closed else 0

        try:
            with ThreadPoolExecutor(max_workers=mp) as pool:
                futs = [pool.submit(_one, t, s) for t in range(len(texts)) for s in range(k)]
                for fut in as_completed(futs):
                    t_idx, s_idx, pred = fut.result()
                    grid[t_idx][s_idx] = pred
                    n_calls += 1
        finally:
            worker.temperature = old_temp

        vote_rows = [[int(grid[t][s] if grid[t][s] is not None else INVALID) for s in range(k)] for t in range(len(texts))]
        return self_consistency_uncertainty(vote_rows), n_calls
