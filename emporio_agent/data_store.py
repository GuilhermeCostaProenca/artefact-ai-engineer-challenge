"""Loads the store's CSV tables into typed, read-only records.

The CSV files are never modified; all cleaning happens in memory at load time.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

FILE_PREFIX = "desafio_tecnico_ai_eng - "


@dataclass(frozen=True)
class Category:
    category_id: int
    name: str
    description: str


@dataclass(frozen=True)
class Product:
    product_id: int
    name: str
    category_id: int
    price_brl: float
    description: str
    stock_quantity: int
    status: str  # active | discontinued | coming_soon
    specs: dict[str, str]
    title_text: str = field(repr=False)  # normalized name + category
    search_text: str = field(repr=False)  # normalized name + category + description + specs


@dataclass(frozen=True)
class Promotion:
    promotion_id: int
    product_id: int
    discount_percent: float
    description: str
    is_active: bool


@dataclass(frozen=True)
class Customer:
    customer_id: int
    name: str
    phone: str
    email: str
    city: str


@dataclass(frozen=True)
class OrderItem:
    product_id: int
    quantity: int


@dataclass(frozen=True)
class Order:
    order_id: int
    customer_id: int
    order_date: date
    status: str  # pending | confirmed | shipped | delivered | cancelled
    total_brl: float
    payment_method: str
    tracking_code: str | None
    estimated_delivery: date | None
    notes: str | None
    items: tuple[OrderItem, ...]


def normalize(text: str) -> str:
    """Lowercase, strip accents and collapse punctuation so 'Violão GD-20' ~ 'violao gd20'."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = re.sub(r"(?<=\w)[-/.](?=\w)", "", text)  # GD-20 -> gd20, KA-15S -> ka15s
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def only_digits(text: str) -> str:
    return re.sub(r"\D", "", text or "")


class DataStore:
    def __init__(self, data_dir: Path):
        self.categories = {c.category_id: c for c in _load_categories(data_dir)}
        self.products = {p.product_id: p for p in _load_products(data_dir, self.categories)}
        self.promotions = list(_load_promotions(data_dir))
        self.customers = {c.customer_id: c for c in _load_customers(data_dir)}
        self.orders = {o.order_id: o for o in _load_orders(data_dir)}

    def category_name(self, category_id: int) -> str:
        category = self.categories.get(category_id)
        return category.name if category else "Sem categoria"

    def active_promotion(self, product_id: int) -> Promotion | None:
        """Best active promotion for a product (promotions are never cumulative)."""
        active = [p for p in self.promotions if p.product_id == product_id and p.is_active]
        return max(active, key=lambda p: p.discount_percent, default=None)


def _read(data_dir: Path, table: str) -> list[dict[str, str]]:
    path = data_dir / f"{FILE_PREFIX}{table}.csv"
    with path.open(encoding="utf-8", newline="") as f:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def _load_categories(data_dir: Path):
    for row in _read(data_dir, "categories"):
        yield Category(int(row["category_id"]), row["name"], row["description"])


def _load_products(data_dir: Path, categories: dict[int, Category]):
    for row in _read(data_dir, "products"):
        try:
            specs = {k: str(v) for k, v in json.loads(row["specs"] or "{}").items()}
        except json.JSONDecodeError:
            specs = {}
        category_id = int(row["category_id"])
        category = categories.get(category_id)
        title_text = normalize(f"{row['name']} {category.name if category else ''}")
        search_text = f"{title_text} {normalize(' '.join([row['description'], *specs.values()]))}"
        yield Product(
            product_id=int(row["product_id"]),
            name=row["name"],
            category_id=category_id,
            price_brl=float(row["price_brl"]),
            description=row["description"],
            stock_quantity=max(int(row["stock_quantity"] or 0), 0),
            status=row["status"] or "active",
            specs=specs,
            title_text=title_text,
            search_text=search_text,
        )


def _load_promotions(data_dir: Path):
    for row in _read(data_dir, "promotions"):
        yield Promotion(
            promotion_id=int(row["promotion_id"]),
            product_id=int(row["product_id"]),
            discount_percent=float(row["discount_percent"]),
            description=row["description"],
            is_active=row["is_active"] == "1",
        )


def _load_customers(data_dir: Path):
    for row in _read(data_dir, "customers"):
        yield Customer(int(row["customer_id"]), row["name"], row["phone"], row["email"], row["city"])


def _load_orders(data_dir: Path):
    items: dict[int, list[OrderItem]] = {}
    for row in _read(data_dir, "order_items"):
        items.setdefault(int(row["order_id"]), []).append(
            OrderItem(product_id=int(row["product_id"]), quantity=int(row["quantity"]))
        )
    for row in _read(data_dir, "orders"):
        order_id = int(row["order_id"])
        yield Order(
            order_id=order_id,
            customer_id=int(row["customer_id"]),
            order_date=date.fromisoformat(row["order_date"]),
            status=row["status"],
            total_brl=float(row["total_brl"]),
            payment_method=row["payment_method"],
            tracking_code=row["tracking_code"] or None,
            estimated_delivery=date.fromisoformat(row["estimated_delivery"]) if row["estimated_delivery"] else None,
            notes=row["notes"] or None,
            items=tuple(items.get(order_id, [])),
        )
