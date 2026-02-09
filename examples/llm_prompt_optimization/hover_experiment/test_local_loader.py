"""Test loading HoVer dataset from local JSON files."""
import sys
import os
import json

# Direct implementation to avoid importing evaluator (which requires OpenAI config)
def load_local_hover_dataset(split="train"):
    """Load HoVer dataset from local JSON files."""
    data_dir = os.path.join(os.path.dirname(__file__), "data", "hover", "tfidf_retrieved")
    
    # Map split names to filenames
    split_map = {
        "train": "train_tfidf_doc_retrieval_results.json",
        "dev": "dev_tfidf_doc_retrieval_results.json",
        "validation": "dev_tfidf_doc_retrieval_results.json",  # Alias for dev
        "test": "test_tfidf_doc_retrieval_results.json"
    }
    
    filename = split_map.get(split, split_map["train"])
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        # Fallback to train if requested split not found
        if split != "train":
            print(f"Warning: Split '{split}' file not found, using 'train'")
            filepath = os.path.join(data_dir, split_map["train"])
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"HoVer data file not found: {filepath}")
    
    print(f"Loading HoVer from local file: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} examples from {split} split")
    return data

print("Testing local HoVer dataset loader...")
print("=" * 80)

# Test loading train split
try:
    dataset = load_local_hover_dataset('train')
    print(f"\n[OK] Successfully loaded {len(dataset)} examples from train split")
    
    # Check first example
    if len(dataset) > 0:
        example = dataset[0]
        print(f"\nFirst example:")
        print(f"  ID: {example.get('id', 'N/A')}")
        print(f"  Claim: {example.get('claim', 'N/A')[:100]}...")
        print(f"  Label: {example.get('label', 'N/A')}")
        print(f"  Verifiable: {example.get('verifiable', 'N/A')}")
        
        # Count label distribution
        labels = {}
        for ex in dataset[:1000]:  # Sample first 1000
            label = ex.get('label', 'UNKNOWN')
            labels[label] = labels.get(label, 0) + 1
        print(f"\nLabel distribution (first 1000 examples):")
        for label, count in sorted(labels.items()):
            print(f"  {label}: {count}")
    
except Exception as e:
    print(f"\n[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("[OK] All tests passed!")

