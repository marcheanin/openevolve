"""Verbose stage tracing for smoke / pipeline validation runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class StageRecord:
    stage_id: str
    title: str
    status: str  # ok | warn | fail
    details: Dict[str, Any] = field(default_factory=dict)


class StageTracer:
    """Print and persist per-stage checkpoints for end-to-end smoke validation."""

    PIPELINE_STAGES = [
        "01_config",
        "02_run_context",
        "03_data_load",
        "04_clustering",
        "05_pool_init",
        "06_inference_seen",
        "07_acquisition",
        "08_fitness",
        "09_error_artifacts",
        "10_evolution",
        "11_consolidation",
        "12_selection_val",
        "13_pool_expand",
        "14_final_test",
        "15_summary",
    ]

    def __init__(self, run_dir: Path, enabled: bool = True, verbose: bool = True) -> None:
        self.run_dir = run_dir
        self.enabled = enabled
        self.verbose = verbose
        self.records: List[StageRecord] = []
        self._jsonl = run_dir / "smoke_trace.jsonl" if enabled else None

    def stage(
        self,
        stage_id: str,
        title: str,
        status: str = "ok",
        **details: Any,
    ) -> None:
        if not self.enabled:
            return
        rec = StageRecord(stage_id=stage_id, title=title, status=status, details=details)
        self.records.append(rec)
        idx = len(self.records)
        total = len(self.PIPELINE_STAGES)
        mark = {"ok": "OK", "warn": "WARN", "fail": "FAIL"}.get(status, status.upper())
        line = f"[STAGE {idx:02d}/{total}] {stage_id} {title}: {mark}"
        if self.verbose:
            self._safe_print(line)
            for k, v in details.items():
                self._safe_print(f"         {k}: {v}")
        if self._jsonl is not None:
            payload = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "stage_id": stage_id,
                "title": title,
                "status": status,
                "details": details,
            }
            with open(self._jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, default=str, ensure_ascii=False) + "\n")

    @staticmethod
    def _safe_print(msg: str) -> None:
        try:
            print(msg, flush=True)
        except UnicodeEncodeError:
            print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)

    def write_checklist(self) -> Path:
        path = self.run_dir / "smoke_checklist.json"
        if not self.enabled:
            return path
        expected = set(self.PIPELINE_STAGES)
        seen = {r.stage_id for r in self.records}
        missing = sorted(expected - seen)
        failed = [r.stage_id for r in self.records if r.status == "fail"]
        checklist = {
            "all_stages_passed": len(missing) == 0 and len(failed) == 0,
            "stages_completed": len(self.records),
            "stages_expected": len(self.PIPELINE_STAGES),
            "missing_stages": missing,
            "failed_stages": failed,
            "records": [
                {
                    "stage_id": r.stage_id,
                    "title": r.title,
                    "status": r.status,
                    "details": r.details,
                }
                for r in self.records
            ],
        }
        path.write_text(json.dumps(checklist, indent=2, default=str), encoding="utf-8")
        if self.verbose:
            verdict = "PASS" if checklist["all_stages_passed"] else "FAIL"
            print(f"\n[SMOKE CHECKLIST] {verdict} — {path}", flush=True)
            if missing:
                print(f"  missing: {missing}", flush=True)
            if failed:
                print(f"  failed: {failed}", flush=True)
        return path
