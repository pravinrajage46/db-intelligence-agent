# create_demo_db.py
"""
Creates a realistic 8-table e-commerce SQLite database for instant demo.
No external downloads needed.

Run: python create_demo_db.py
"""
import os
import random
import sqlite3
import string
from datetime import datetime, timedelta

random.seed(42)


def rand_id(prefix: str, n: int = 8) -> str:
    chars = string.ascii_uppercase + string.digits
    return prefix + "".join(random.choices(chars, k=n))


def rand_dt(start: datetime, end: datetime) -> str:
    delta_days = (end - start).days
    dt = start + timedelta(
        days=random.randint(0, delta_days),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# Reference data
STATES      = ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "GO", "PE", "CE"]
CITIES      = ["Sao Paulo", "Rio de Janeiro", "Belo Horizonte", "Porto Alegre",
               "Curitiba", "Florianopolis", "Salvador", "Goiania", "Recife", "Fortaleza"]
CATEGORIES  = ["electronics", "clothing", "home_appliances", "sports", "books",
               "beauty", "toys", "furniture", "food", "automotive"]
STATUSES    = ["delivered", "shipped", "processing", "canceled", "invoiced"]
STATUS_W    = [65, 12, 8, 8, 7]
PAY_TYPES   = ["credit_card", "boleto", "debit_card", "voucher"]

START = datetime(2023, 1, 1)
END   = datetime(2024, 12, 31)


def create_demo_db(db_path: str = "./data/demo_ecommerce.db") -> str:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    print("🏗️  Creating 8 tables...")
    cur.executescript("""
    CREATE TABLE customers (
        customer_id         TEXT PRIMARY KEY,
        customer_unique_id  TEXT NOT NULL,
        zip_code            TEXT,
        city                TEXT,
        state               TEXT
    );
    CREATE TABLE sellers (
        seller_id   TEXT PRIMARY KEY,
        zip_code    TEXT,
        city        TEXT,
        state       TEXT
    );
    CREATE TABLE products (
        product_id                  TEXT PRIMARY KEY,
        product_category            TEXT,
        product_name_length         INTEGER,
        product_description_length  INTEGER,
        product_photos_qty          INTEGER,
        product_weight_g            REAL,
        product_length_cm           REAL,
        product_height_cm           REAL,
        product_width_cm            REAL
    );
    CREATE TABLE orders (
        order_id                        TEXT PRIMARY KEY,
        customer_id                     TEXT NOT NULL,
        order_status                    TEXT,
        order_purchase_timestamp        TEXT,
        order_approved_at               TEXT,
        order_delivered_carrier_date    TEXT,
        order_delivered_customer_date   TEXT,
        order_estimated_delivery_date   TEXT,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    );
    CREATE TABLE order_items (
        order_id            TEXT NOT NULL,
        order_item_id       INTEGER NOT NULL,
        product_id          TEXT,
        seller_id           TEXT,
        shipping_limit_date TEXT,
        price               REAL,
        freight_value       REAL,
        PRIMARY KEY (order_id, order_item_id),
        FOREIGN KEY (order_id)   REFERENCES orders(order_id),
        FOREIGN KEY (product_id) REFERENCES products(product_id),
        FOREIGN KEY (seller_id)  REFERENCES sellers(seller_id)
    );
    CREATE TABLE payments (
        order_id                TEXT NOT NULL,
        payment_sequential      INTEGER,
        payment_type            TEXT,
        payment_installments    INTEGER,
        payment_value           REAL,
        FOREIGN KEY (order_id) REFERENCES orders(order_id)
    );
    CREATE TABLE reviews (
        review_id               TEXT PRIMARY KEY,
        order_id                TEXT,
        review_score            INTEGER,
        review_comment_title    TEXT,
        review_comment_message  TEXT,
        review_creation_date    TEXT,
        review_answer_timestamp TEXT,
        FOREIGN KEY (order_id) REFERENCES orders(order_id)
    );
    CREATE TABLE geolocation (
        zip_code    TEXT,
        latitude    REAL,
        longitude   REAL,
        city        TEXT,
        state       TEXT
    );
    """)

    # ── Customers (500) ───────────────────────────────────────────────────────
    print("👥 500 customers...")
    customer_ids = [rand_id("CUST") for _ in range(500)]
    cur.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", [
        (cid, rand_id("UNIQ"),
         f"{random.randint(10000,99999):05d}",
         random.choice(CITIES),
         random.choice(STATES))
        for cid in customer_ids
    ])

    # ── Sellers (50) ─────────────────────────────────────────────────────────
    print("🏪 50 sellers...")
    seller_ids = [rand_id("SELL") for _ in range(50)]
    cur.executemany("INSERT INTO sellers VALUES (?,?,?,?)", [
        (sid, f"{random.randint(10000,99999):05d}",
         random.choice(CITIES), random.choice(STATES))
        for sid in seller_ids
    ])

    # ── Products (200) ────────────────────────────────────────────────────────
    print("📦 200 products...")
    product_ids = [rand_id("PROD") for _ in range(200)]
    cur.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?)", [
        (pid,
         random.choice(CATEGORIES),
         random.randint(20, 100),
         random.randint(100, 1000),
         random.randint(1, 5),
         round(random.uniform(100, 5000), 2),
         round(random.uniform(10, 100), 2),
         round(random.uniform(5, 50), 2),
         round(random.uniform(5, 50), 2))
        for pid in product_ids
    ])

    # ── Orders (2 000) ────────────────────────────────────────────────────────
    print("🛒 2,000 orders...")
    order_ids = [rand_id("ORD") for _ in range(2000)]
    orders_rows = []
    for oid in order_ids:
        purchase  = START + timedelta(days=random.randint(0, (END - START).days))
        approved  = purchase + timedelta(hours=random.randint(1, 48))
        carrier   = purchase + timedelta(days=random.randint(2, 7))
        estimated = purchase + timedelta(days=random.randint(8, 40))
        status    = random.choices(STATUSES, weights=STATUS_W)[0]
        delivered = None
        if status == "delivered":
            delivered = (purchase + timedelta(days=random.randint(5, 30))).strftime("%Y-%m-%d %H:%M:%S")
        orders_rows.append((
            oid, random.choice(customer_ids), status,
            purchase.strftime("%Y-%m-%d %H:%M:%S"),
            approved.strftime("%Y-%m-%d %H:%M:%S"),
            carrier.strftime("%Y-%m-%d %H:%M:%S"),
            delivered,                                  # realistic NULLs here
            estimated.strftime("%Y-%m-%d %H:%M:%S"),
        ))
    cur.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?)", orders_rows)

    # ── Order Items (≈4 000) ──────────────────────────────────────────────────
    print("🔢 ~4,000 order items...")
    items_rows = []
    for oid in order_ids:
        for item_num in range(1, random.randint(1, 4)):
            price = round(random.uniform(10, 800), 2)
            items_rows.append((
                oid, item_num,
                random.choice(product_ids),
                random.choice(seller_ids),
                rand_dt(START, END),
                price,
                round(price * random.uniform(0.05, 0.25), 2),
            ))
    cur.executemany("INSERT INTO order_items VALUES (?,?,?,?,?,?,?)", items_rows)

    # ── Payments (2 000) ─────────────────────────────────────────────────────
    print("💳 2,000 payments...")
    cur.executemany("INSERT INTO payments VALUES (?,?,?,?,?)", [
        (oid, 1,
         random.choice(PAY_TYPES),
         random.randint(1, 12),
         round(random.uniform(10, 2000), 2))
        for oid in order_ids
    ])

    # ── Reviews (1 600, ~20% with NULL comments = realistic quality variation) ─
    print("⭐ 1,600 reviews (with intentional null patterns)...")
    reviewed = random.sample(order_ids, 1600)
    reviews_rows = []
    for i, oid in enumerate(reviewed):
        score       = random.choices([1, 2, 3, 4, 5], weights=[5, 8, 12, 30, 45])[0]
        has_comment = random.random() > 0.40  # 40% null comments — realistic
        reviews_rows.append((
            rand_id("REV"), oid, score,
            f"Review {i}" if has_comment else None,
            f"Product quality score {score} out of 5." if has_comment else None,
            rand_dt(START, END),
            rand_dt(START, END),
        ))
    cur.executemany("INSERT INTO reviews VALUES (?,?,?,?,?,?,?)", reviews_rows)

    # ── Geolocation (2 000) ───────────────────────────────────────────────────
    print("🗺️  2,000 geolocation records...")
    cur.executemany("INSERT INTO geolocation VALUES (?,?,?,?,?)", [
        (f"{random.randint(10000,99999):05d}",
         round(random.uniform(-33.0, 5.0), 6),
         round(random.uniform(-73.0, -35.0), 6),
         random.choice(CITIES),
         random.choice(STATES))
        for _ in range(2000)
    ])

    conn.commit()
    conn.close()

    total = 500 + 50 + 200 + 2000 + len(items_rows) + 2000 + 1600 + 2000
    print(f"\n{'─'*55}")
    print(f"✅ Demo database created!")
    print(f"   Path  : {os.path.abspath(db_path)}")
    print(f"   Tables: 8")
    print(f"   Rows  : ~{total:,}")
    print(f"{'─'*55}\n")
    return db_path


if __name__ == "__main__":
    create_demo_db()
    print("Next step → run the agent:")
    print("  python main.py")
    print("  streamlit run ui/dashboard.py")