"""Seed a demo SQLite database with customers / products / orders tables.

Run once at install time (or automatically via main.py / Streamlit bootstrap).
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
    select,
)

from sql_agent.config import settings


def _engine():
    url = settings.database_url
    # Ensure parent directory exists for SQLite file paths.
    if url.startswith("sqlite:///"):
        rel = url.replace("sqlite:///", "", 1)
        db_path = settings.resolved_path(rel)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{db_path.as_posix()}"
    return create_engine(url, future=True)


def _build_metadata() -> MetaData:
    metadata = MetaData()

    Table(
        "customers",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(120), nullable=False),
        Column("email", String(200), nullable=False),
        Column("country", String(80), nullable=False),
        Column("created_at", DateTime, nullable=False),
    )

    Table(
        "products",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(120), nullable=False),
        Column("category", String(80), nullable=False),
        Column("unit_price", Float, nullable=False),
    )

    Table(
        "orders",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("customer_id", Integer, ForeignKey("customers.id"), nullable=False),
        Column("product_id", Integer, ForeignKey("products.id"), nullable=False),
        Column("quantity", Integer, nullable=False),
        Column("revenue", Float, nullable=False),
        Column("order_date", Date, nullable=False),
        Column("status", String(30), nullable=False),
    )

    return metadata


COUNTRIES = ["USA", "India", "Germany", "UK", "Brazil", "Japan", "Canada", "France"]
CATEGORIES = ["Electronics", "Books", "Apparel", "Home", "Toys"]
STATUSES = ["completed", "pending", "refunded", "cancelled"]


def seed(*, force: bool = False) -> None:
    """Create tables and seed with ~1 year of synthetic data."""
    engine = _engine()
    metadata = _build_metadata()

    if not force:
        with engine.connect() as conn:
            from sqlalchemy import inspect

            insp = inspect(conn)
            if {"customers", "products", "orders"}.issubset(set(insp.get_table_names())):
                # Already seeded.
                return

    metadata.drop_all(engine)
    metadata.create_all(engine)

    random.seed(42)
    now = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    customers_tbl = metadata.tables["customers"]
    products_tbl = metadata.tables["products"]
    orders_tbl = metadata.tables["orders"]

    customer_rows = [
        {
            "id": i,
            "name": f"Customer {i}",
            "email": f"customer{i}@example.com",
            "country": random.choice(COUNTRIES),
            "created_at": now - timedelta(days=random.randint(30, 720)),
        }
        for i in range(1, 101)
    ]

    product_rows = [
        {
            "id": i,
            "name": f"Product {i}",
            "category": random.choice(CATEGORIES),
            "unit_price": round(random.uniform(5, 500), 2),
        }
        for i in range(1, 31)
    ]

    order_rows = []
    oid = 1
    for days_ago in range(365, 0, -1):
        order_date = (now - timedelta(days=days_ago)).date()
        # Vary daily volume.
        for _ in range(random.randint(3, 15)):
            cust = random.choice(customer_rows)
            prod = random.choice(product_rows)
            qty = random.randint(1, 6)
            order_rows.append(
                {
                    "id": oid,
                    "customer_id": cust["id"],
                    "product_id": prod["id"],
                    "quantity": qty,
                    "revenue": round(qty * prod["unit_price"], 2),
                    "order_date": order_date,
                    "status": random.choices(
                        STATUSES, weights=[0.75, 0.1, 0.05, 0.1]
                    )[0],
                }
            )
            oid += 1

    with engine.begin() as conn:
        conn.execute(insert(customers_tbl), customer_rows)
        conn.execute(insert(products_tbl), product_rows)
        # Batched insert for orders.
        batch = 500
        for i in range(0, len(order_rows), batch):
            conn.execute(insert(orders_tbl), order_rows[i : i + batch])

        total = conn.execute(select(orders_tbl)).fetchall()
        print(
            f"Seeded demo DB at {settings.database_url}: "
            f"{len(customer_rows)} customers, {len(product_rows)} products, {len(total)} orders"
        )


if __name__ == "__main__":
    seed(force=False)
