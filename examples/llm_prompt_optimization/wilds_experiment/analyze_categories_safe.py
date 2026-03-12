"""
Safe analysis of the WILDS Amazon reviews.csv to get
per-category statistics even if the CSV has malformed rows.
"""

import os
import csv

import numpy as np
import pandas as pd


def main() -> None:
    root_dir = os.path.join(".", "data")
    # Default directory used by WILDS Amazon v2.1
    data_dir = os.path.join(root_dir, "amazon_v2.1")
    reviews_path = os.path.join(data_dir, "reviews.csv")

    if not os.path.exists(reviews_path):
        print(f"reviews.csv not found at: {os.path.abspath(reviews_path)}")
        return

    print("=" * 80)
    print("SAFE WILDS Amazon Category Analysis")
    print("=" * 80)
    print(f"Reading: {os.path.abspath(reviews_path)}")

    # Use the Python engine and skip bad lines to avoid EOF / quoting issues.
    df = pd.read_csv(
        reviews_path,
        dtype={
            "reviewerID": str,
            "asin": str,
            "reviewTime": str,
            "unixReviewTime": "Int64",
            "reviewText": str,
            "summary": str,
            "verified": "boolean",
            "category": str,
            "reviewYear": "Int64",
            "overall": "Int64",
        },
        keep_default_na=False,
        na_values=[],
        engine="python",
        on_bad_lines="skip",
        quoting=csv.QUOTE_MINIMAL,
    )

    n_rows = len(df)
    print(f"\nTotal rows successfully parsed: {n_rows:,}")

    # Basic overall star distribution
    print("\n[1] Overall star rating distribution (all categories)")
    if "overall" in df.columns:
        counts = df["overall"].value_counts().sort_index()
        print("\nRating  Count       Percent")
        print("-" * 32)
        for rating, count in counts.items():
            pct = count / n_rows * 100 if n_rows > 0 else 0.0
            print(f"{int(rating):>2}     {int(count):>10,}   {pct:6.2f}%")
    else:
        print("Column 'overall' not found.")

    # Per-category counts
    print("\n[2] Category-level statistics")
    if "category" not in df.columns:
        print("Column 'category' not found.")
        return

    cat_counts = df["category"].value_counts()
    n_cats = len(cat_counts)
    print(f"\nNumber of categories: {n_cats}")
    print("\n{:<3} {:<40} {:>12} {:>8}".format("ID", "Category", "Reviews", "%"))
    print("-" * 70)
    for idx, (cat_name, count) in enumerate(cat_counts.items()):
        pct = count / n_rows * 100 if n_rows > 0 else 0.0
        print(f"{idx:<3} {cat_name[:38]:<40} {int(count):>12,} {pct:7.2f}%")

    # Optional: per-category star distribution (aggregated)
    print("\n[3] Per-category star distribution (aggregated)")
    if "overall" in df.columns:
        grouped = (
            df.groupby(["category", "overall"])["overall"]
            .count()
            .rename("count")
            .reset_index()
        )
        # For brevity, print a compact summary
        for cat_name, sub in grouped.groupby("category"):
            total_cat = int(sub["count"].sum())
            print(f"\nCategory: {cat_name}  (total {total_cat:,} reviews)")
            for _, row in sub.sort_values("overall").iterrows():
                rating = int(row["overall"])
                count = int(row["count"])
                pct = count / total_cat * 100 if total_cat > 0 else 0.0
                print(f"  {rating} stars: {count:>8,} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()

