"""GPO (Li+ 2024) — placeholder until Yelp→Flipkart gate body lands (§6.4)."""

from __future__ import annotations

from baselines.api import OptimizerResult, Task


class GPOOptimizer:
    def run(self, task: Task) -> OptimizerResult:
        raise NotImplementedError(
            "GPO body not implemented. Implement before E5 matrix; "
            "gate via scripts/run_gpo_yelp_flipkart_gate.py"
        )
