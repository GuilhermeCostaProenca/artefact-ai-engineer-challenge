from emporio_agent.data_store import normalize, only_digits
from emporio_agent.policies import POLICY_SECTIONS


def test_tables_load_with_expected_sizes(store):
    assert len(store.products) == 65
    assert len(store.customers) == 50
    assert len(store.orders) == 20
    assert len(store.promotions) == 25
    assert len(store.categories) == 9


def test_order_items_are_attached_to_orders(store):
    assert [(i.product_id, i.quantity) for i in store.orders[15].items] == [(82, 1), (106, 1)]
    assert all(order.items for order in store.orders.values())


def test_empty_csv_fields_become_none(store):
    pending = store.orders[13]
    assert pending.tracking_code is None and pending.estimated_delivery is None


def test_active_promotion_ignores_inactive_entries(store):
    assert store.active_promotion(95) is None  # Takamine GD20: only an expired promo
    assert store.active_promotion(121).discount_percent == 20


def test_normalize_folds_accents_and_model_punctuation():
    assert normalize("Violão Takamine GD-20") == "violao takamine gd20"
    assert normalize("KA-15S") == "ka15s"
    assert only_digits("(67) 99812-3456") == "67998123456"


def test_policy_manual_is_split_into_expected_sections():
    assert set(POLICY_SECTIONS) == {
        "about_store", "business_hours", "payment_methods", "returns_and_exchanges",
        "shipping_and_delivery", "promotions", "warranty", "privacy", "service_procedures",
    }


def test_policy_sections_keep_key_facts_from_the_pdf():
    assert "Rua 14 de Maio, 3200" in POLICY_SECTIONS["about_store"]
    assert "09:00 às 18:00" in POLICY_SECTIONS["business_hours"]
    assert "09:00 às 13:00" in POLICY_SECTIONS["business_hours"]
    assert "5% de desconto" in POLICY_SECTIONS["payment_methods"]
    assert "7 (sete) dias corridos após o recebimento" in POLICY_SECTIONS["returns_and_exchanges"]
    assert "30 (trinta) dias corridos após a compra" in POLICY_SECTIONS["returns_and_exchanges"]
    assert "R$ 500,00" in POLICY_SECTIONS["shipping_and_delivery"]
    assert "90 (noventa) dias" in POLICY_SECTIONS["warranty"]
