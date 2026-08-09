#!/usr/bin/env python3
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen


def load_env(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    load_env(root / ".env")
    load_env(root.parent / "wilds_active_learn_approach" / ".env")
    key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        print("NO_KEY")
        raise SystemExit(1)

    def get(url: str) -> dict:
        req = Request(url, headers={"Authorization": f"Bearer {key}"})
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())

    k = get("https://openrouter.ai/api/v1/key")["data"]
    c = get("https://openrouter.ai/api/v1/credits")["data"]

    print("=== KEY (from .env) ===")
    print(f"label: {k.get('label')}")
    print(f"limit: ${k.get('limit', 0):.2f}")
    print(f"usage (lifetime): ${k.get('usage', 0):.2f}")
    print(f"remaining: ${k.get('limit_remaining', 0):.2f}")
    print(f"daily: ${k.get('usage_daily', 0):.2f}")
    print(f"weekly: ${k.get('usage_weekly', 0):.2f}")
    print(f"monthly: ${k.get('usage_monthly', 0):.2f}")
    print()
    print("=== ACCOUNT (all keys) ===")
    tc = float(c.get("total_credits") or 0)
    tu = float(c.get("total_usage") or 0)
    print(f"total_credits: ${tc:,.2f}")
    print(f"total_usage: ${tu:,.2f}")
    print(f"balance (credits - usage): ${tc - tu:,.2f}")


if __name__ == "__main__":
    main()
