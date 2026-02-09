#!/usr/bin/env python
"""
Utility to test loading the HoVer dataset, either from a local cache or directly via datasets.load_dataset.
Exits with code 0 on success, non-zero on failure.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Optional


def print_versions() -> None:
    try:
        import datasets  # type: ignore
        datasets_ver = datasets.__version__
    except Exception as e:  # pragma: no cover
        datasets = None  # type: ignore
        datasets_ver = f"NOT INSTALLED ({e})"
    try:
        import pyarrow  # type: ignore
        pyarrow_ver = pyarrow.__version__
    except Exception as e:  # pragma: no cover
        pyarrow = None  # type: ignore
        pyarrow_ver = f"NOT INSTALLED ({e})"
    py_ver = sys.version.split()[0]
    print(f"[env] python={py_ver}, datasets={datasets_ver}, pyarrow={pyarrow_ver}")


def try_load_from_cache(cache_path: str, split: str, sample: int) -> Optional[int]:
    """
    Attempt to load HoVer from a local datasets cache directory.
    Returns number of records on success, or None on failure.
    """
    from datasets import load_from_disk  # type: ignore

    if not os.path.exists(cache_path):
        print(f"[cache] not found: {cache_path}")
        return None
    try:
        print(f"[cache] loading from: {cache_path}")
        ds_or_dd = load_from_disk(cache_path)
        # ds_or_dd can be a DatasetDict or a single Dataset
        length: Optional[int] = None
        if hasattr(ds_or_dd, "keys"):
            # DatasetDict
            keys = list(ds_or_dd.keys())
            print(f"[cache] available splits: {keys}")
            if split in ds_or_dd:
                ds = ds_or_dd[split]
                length = len(ds)
                print(f"[cache] split='{split}' loaded, len={length}")
                if sample > 0:
                    print(f"[cache] sample[0:{sample}]:")
                    for i in range(min(sample, length)):
                        print(ds[i])
            else:
                print(f"[cache] split '{split}' not in cache; available: {keys}")
                return None
        else:
            # Single Dataset
            ds = ds_or_dd
            length = len(ds)
            print(f"[cache] single dataset loaded, len={length}")
            if sample > 0:
                print(f"[cache] sample[0:{sample}]:")
                for i in range(min(sample, length)):
                    print(ds[i])
        return length
    except Exception as e:
        print(f"[cache] failed: {e}")
        traceback.print_exc()
        return None


def try_load_from_hf(split: str, sample: int, streaming: bool = False) -> Optional[int]:
    """
    Attempt to load HoVer from HuggingFace Hub using datasets.load_dataset.
    Returns number of records on success, or None on failure.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        print(f"[hf] datasets not available: {e}")
        return None

    load_kwargs = {"split": split}
    # Older datasets support trust_remote_code; newer deprecate scripts entirely.
    # We pass it defensively; if unsupported it will be ignored/raise and we retry.
    try:
        print(f"[hf] trying load_dataset('hover', split='{split}', streaming={streaming})")
        ds = load_dataset("hover", split=split, streaming=streaming, trust_remote_code=False)  # type: ignore[call-arg]
    except TypeError:
        # trust_remote_code not supported
        print("[hf] retry without trust_remote_code (not supported in this datasets version)")
        ds = load_dataset("hover", split=split, streaming=streaming)  # type: ignore[call-arg]
    except Exception as e:
        print(f"[hf] failed: {e}")
        traceback.print_exc()
        return None

    try:
        if streaming:
            # For streaming, count first N to prove it's working
            print("[hf] streaming=True; counting first 100 items...")
            count = 0
            for _ in ds:
                count += 1
                if count >= 100:
                    break
            print(f"[hf] stream peek count={count}")
            return count
        # Non-streaming path
        length = len(ds)
        print(f"[hf] dataset loaded, len={length}")
        if sample > 0:
            print(f"[hf] sample[0:{sample}]:")
            for i in range(min(sample, length)):
                print(ds[i])
        return length
    except Exception as e:
        print(f"[hf] post-load handling failed: {e}")
        traceback.print_exc()
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Test loading the HoVer dataset.")
    parser.add_argument("--split", type=str, default="train", help="Split to load: train|validation|test")
    parser.add_argument("--cache-path", type=str, default=os.path.join(os.path.dirname(__file__), "hover_dataset_cache"),
                        help="Path to local datasets cache directory")
    parser.add_argument("--no-cache", action="store_true", help="Skip local cache and try HF directly")
    parser.add_argument("--force-hf", action="store_true", help="Force trying HF even if cache loads")
    parser.add_argument("--streaming", action="store_true", help="Try streaming mode (HF only)")
    parser.add_argument("--sample", type=int, default=3, help="How many examples to print if load succeeds")
    args = parser.parse_args()

    print_versions()
    print(f"[run] split='{args.split}', cache_path='{args.cache_path}', streaming={args.streaming}")

    loaded_count: Optional[int] = None

    if not args.no_cache:
        loaded_count = try_load_from_cache(args.cache_path, args.split, args.sample)
        if loaded_count is not None and not args.force_hf:
            print("[result] SUCCESS via cache")
            return 0

    hf_count = try_load_from_hf(args.split, args.sample, streaming=args.streaming)
    if hf_count is not None:
        print("[result] SUCCESS via HuggingFace")
        print("[hint] If you want to cache locally for future runs, you can run:")
        print("       python -c \"from datasets import load_dataset; ds=load_dataset('hover', split='train'); ds.save_to_disk('hover_dataset_cache')\"")
        return 0

    # If we got here, both attempts failed
    print("[result] FAILED to load HoVer dataset.")
    print("Possible fixes:")
    print('  - Install compatible versions: pip install "datasets==2.14.0" "pyarrow<15.0"')
    print("  - Or create a local cache using an older-compatible environment, then copy it to:")
    print(f"    {args.cache_path}")
    return 2


if __name__ == "__main__":
    sys.exit(main())


