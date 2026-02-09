#!/usr/bin/env python3
"""
Скрипт для загрузки и кеширования датасета HoVer.
Использует более старую версию datasets для загрузки, затем сохраняет локально.
"""

import os
import sys

def main():
    cache_dir = os.path.join(os.path.dirname(__file__), "hover_dataset_cache")
    
    if os.path.exists(cache_dir):
        print(f"Dataset cache already exists at {cache_dir}")
        print("You can use it by running the evolution.")
        return
    
    print("=" * 80)
    print("HoVer Dataset Cache Creator")
    print("=" * 80)
    print()
    print("This script will:")
    print("1. Load HoVer dataset using datasets library")
    print("2. Save it to local cache for faster access")
    print()
    print("NOTE: This requires compatible versions:")
    print("      datasets==2.14.0 and pyarrow<15.0")
    print("      (dataset scripts are not supported in datasets 4.0+)")
    print()
    
    # Check datasets and pyarrow versions
    try:
        import datasets
        import pyarrow
        datasets_version = datasets.__version__
        pyarrow_version = pyarrow.__version__
        
        datasets_major = int(datasets_version.split('.')[0])
        pyarrow_major = int(pyarrow_version.split('.')[0])
        
        issues = []
        if datasets_major >= 4:
            issues.append(f"datasets version {datasets_version} is too new (need 2.14.0)")
        if pyarrow_major >= 15:
            issues.append(f"pyarrow version {pyarrow_version} is too new (need < 15.0)")
        
        if issues:
            print(f"WARNING: Version compatibility issues detected:")
            for issue in issues:
                print(f"  - {issue}")
            print()
            print("Please install compatible versions:")
            print('  pip install "datasets==2.14.0" "pyarrow<15.0"')
            print()
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
        else:
            print(f"✓ Versions OK: datasets={datasets_version}, pyarrow={pyarrow_version}")
            print()
    except ImportError as e:
        print(f"ERROR: Missing library: {e}")
        print("Please install: pip install datasets pyarrow")
        return
    
    print()
    print("Loading HoVer dataset (this may take a while)...")
    print()
    
    try:
        from datasets import load_dataset
        
        # Try to load with trust_remote_code (for older versions)
        try:
            dataset = load_dataset("hover", split="train", trust_remote_code=True)
        except TypeError:
            # Newer versions don't have trust_remote_code
            dataset = load_dataset("hover", split="train")
        
        print(f"Dataset loaded successfully: {len(dataset)} examples")
        print()
        print(f"Saving to cache: {cache_dir}...")
        
        dataset.save_to_disk(cache_dir)
        
        print()
        print("=" * 80)
        print("SUCCESS!")
        print("=" * 80)
        print(f"Dataset cached at: {cache_dir}")
        print(f"Total examples: {len(dataset)}")
        print()
        print("You can now run the evolution:")
        print("  .\\run_evolution.ps1 --iterations 50")
        print()
        
    except Exception as e:
        print()
        print("=" * 80)
        print("ERROR")
        print("=" * 80)
        print(f"Failed to load dataset: {e}")
        print()
        print("SOLUTIONS:")
        print("1. Install compatible versions:")
        print('   pip install "datasets==2.14.0" "pyarrow<15.0"')
        print()
        print("2. Then run this script again")
        print()
        sys.exit(1)

if __name__ == "__main__":
    main()

