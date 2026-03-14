# data/setup_olist.py
"""
Loads the Olist Brazilian E-Commerce dataset CSV files into a local SQLite database.

Usage:
    python data/setup_olist.py --csv-dir /path/to/olist_csvs
    python data/setup_olist.py --csv-dir /path/to/olist_csvs --db-path ./data/olist.db
"""
from __future__ import annotations

import glob
import os
import sys

import pandas as pd
from sqlalchemy import create_engine, text

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Olist CSV → SQLite table name mapping ─────────────────────────────────────
TABLE_MAP = {
    "olist_customers_dataset.csv": "olist_customers",
    "olist_geolocation_dataset.csv": "olist_geolocation",
    "olist_order_items_dataset.csv": "olist_order_items",
    "olist_order_payments_dataset.csv": "olist_order_payments",
    "olist_order_reviews_dataset.csv": "olist_order_reviews",
    "olist_orders_dataset.csv": "olist_orders",
    "olist_products_dataset.csv": "olist_products",
    "olist_sellers_dataset.csv": "olist_sellers",
    "product_category_name_translation.csv": "product_category_translation",
}

# ── Columns that should be parsed as datetimes ────────────────────────────────
DATE_COLS = {
    "olist_orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "olist_order_reviews": [
        "review_creation_date",
        "review_answer_timestamp",
    ],
}


def load_olist(csv_dir: str, db_path: str = "./data/olist.db"):
    """
    Load all Olist CSVs into a SQLite database.

    Args:
        csv_dir: Directory containing the Olist CSV files.
        db_path: Output path for the SQLite database file.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

    engine = create_engine(f"sqlite:///{db_path}")

    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not csv_files:
        print(f"❌ No CSV files found in: {csv_dir}")
        sys.exit(1)

    print(f"📂 Found {len(csv_files)} CSV files in: {csv_dir}")
    print(f"📦 Loading into: {db_path}")
    print()

    loaded = []
    failed = []

    for csv_path in sorted(csv_files):
        filename = os.path.basename(csv_path)
        # Use mapped name if known, else derive from filename
        table_name = TABLE_MAP.get(
            filename,
            filename.replace(".csv", "").replace("-", "_").lower(),
        )

        try:
            df = pd.read_csv(csv_path, low_memory=False)

            # Parse known datetime columns
            for date_col in DATE_COLS.get(table_name, []):
                if date_col in df.columns:
                    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

            # Write to SQLite
            df.to_sql(table_name, engine, if_exists="replace", index=False)
            print(f"  ✅ {table_name:<40} {len(df):>8,} rows  ·  {len(df.columns)} cols")
            loaded.append(table_name)

        except Exception as e:
            print(f"  ❌ Failed to load {filename}: {e}")
            failed.append(filename)

    # Verify
    with engine.connect() as conn:
        tables_in_db = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        ).fetchall()

    print()
    print(f"{'─'*55}")
    print(f"✅ Successfully loaded: {len(loaded)} tables")
    if failed:
        print(f"❌ Failed:             {len(failed)} files → {failed}")
    print(f"📁 Database saved at:  {os.path.abspath(db_path)}")
    print()
    print("🚀 Next steps:")
    print(f"   CLI:       python main.py --db-url sqlite:///{db_path}")
    print(f"   Dashboard: streamlit run ui/dashboard.py")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load Olist dataset into SQLite")
    parser.add_argument("--csv-dir", required=True, help="Directory with Olist CSV files")
    parser.add_argument("--db-path", default="./data/olist.db", help="Output SQLite path")
    args = parser.parse_args()

    load_olist(args.csv_dir, args.db_path)