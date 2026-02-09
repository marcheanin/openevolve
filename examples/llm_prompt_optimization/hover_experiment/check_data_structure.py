"""Check the structure of downloaded HoVer data files."""
import json
import os
import sqlite3

def check_json_files():
    """Check structure of JSON files."""
    data_dir = "data/hover/tfidf_retrieved"
    files = [
        "train_tfidf_doc_retrieval_results.json",
        "dev_tfidf_doc_retrieval_results.json",
        "test_tfidf_doc_retrieval_results.json"
    ]
    
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
            
        print(f"\n{'='*80}")
        print(f"Checking: {filename}")
        print(f"{'='*80}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Type: {type(data).__name__}")
        if isinstance(data, list):
            print(f"Length: {len(data)}")
            if len(data) > 0:
                print(f"First item type: {type(data[0]).__name__}")
                print(f"First item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'N/A'}")
                print(f"\nSample entry (first item):")
                print(json.dumps(data[0], indent=2)[:1000])
        elif isinstance(data, dict):
            print(f"Number of keys: {len(data)}")
            first_key = list(data.keys())[0]
            print(f"First key: {first_key}")
            print(f"First value type: {type(data[first_key]).__name__}")
            if isinstance(data[first_key], dict):
                print(f"First value keys: {list(data[first_key].keys())}")
                print(f"\nSample entry (first value):")
                print(json.dumps(data[first_key], indent=2)[:1000])

def check_database():
    """Check structure of SQLite database."""
    db_path = "data/wiki_wo_links.db"
    if not os.path.exists(db_path):
        print(f"\nDatabase not found: {db_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"Checking database: {db_path}")
    print(f"{'='*80}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables: {[t[0] for t in tables]}")
        
        # Check each table structure
        for table_name, in tables:
            print(f"\nTable: {table_name}")
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            print("Columns:")
            for col in columns:
                print(f"  - {col[1]} ({col[2]})")
            
            # Get sample row
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
            sample = cursor.fetchone()
            if sample:
                print(f"Sample row: {sample[:200]}")
        
        conn.close()
    except Exception as e:
        print(f"Error checking database: {e}")

if __name__ == "__main__":
    check_json_files()
    check_database()



