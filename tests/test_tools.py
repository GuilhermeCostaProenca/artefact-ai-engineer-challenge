import json

import pytest

from emporio_agent.tools import MAX_FAILED_ORDER_LOOKUPS, execute_tool


def run(ctx, name, args):
    result, is_error = execute_tool(ctx, name, args)
    return result, is_error


# ----------------------------------------------------------------- product discovery

def test_guitars_up_to_1000_are_in_stock_and_within_budget(ctx):
    result, _ = run(ctx, "search_products", {"category": "Violões", "max_price_brl": 1000, "limit": 20})
    products = result["products"]
    assert result["total_matches"] == 12
    assert all(p["current_price_brl"] <= 1000 and p["availability"] == "in_stock" for p in products)
    assert 96 not in {p["product_id"] for p in products}  # Giannini GF-3D has zero stock


def test_price_lookup_by_model_name_tolerates_punctuation(ctx):
    for query in ("Takamine GD20", "takamine gd-20", "violão Takamine"):
        result, _ = run(ctx, "search_products", {"query": query})
        assert result["products"][0]["product_id"] == 95
        assert result["products"][0]["current_price_brl"] == 2199.0


def test_search_matches_portuguese_plurals(ctx):
    result, _ = run(ctx, "search_products", {"query": "ukuleles tenor"})
    assert result["total_matches"] == 4
    assert all("Tenor" in p["name"] for p in result["products"])


def test_unavailable_items_only_show_when_requested(ctx):
    hidden, _ = run(ctx, "search_products", {"query": "Shelby SN-7C"})
    shown, _ = run(ctx, "search_products", {"query": "Shelby SN-7C", "available_only": False})
    assert 113 not in {p["product_id"] for p in hidden["products"]}
    assert "parciais" in hidden["note"]  # other Shelby models, clearly flagged as partial matches
    assert shown["products"][0]["product_id"] == 113
    assert shown["products"][0]["availability"] == "discontinued"


def test_unknown_product_search_returns_note_not_fabrication(ctx):
    result, _ = run(ctx, "search_products", {"query": "flauta transversal"})
    assert result["products"] == []
    assert "note" in result


def test_invalid_category_lists_valid_ones(ctx):
    result, is_error = run(ctx, "search_products", {"category": "Amplificadores"})
    assert is_error and "Guitarras" in result["valid_categories"]


def test_empty_category_is_flagged(ctx):
    result, _ = run(ctx, "search_products", {"category": "Instrumentos de Sopro (Metais)"})
    assert result["products"] == [] and "Não há produtos" in result["note"]


def test_out_of_stock_product_offers_in_stock_alternatives(ctx):
    result, _ = run(ctx, "get_product_details", {"product_id": 96})
    assert result["availability"] == "out_of_stock"
    assert result["alternatives"] and all(a["availability"] == "in_stock" for a in result["alternatives"])
    assert all(a["category"] == "Violões" for a in result["alternatives"])


def test_coming_soon_product_is_not_sold_as_available(ctx):
    result, _ = run(ctx, "get_product_details", {"product_id": 130})
    assert result["availability"] == "coming_soon" and result["stock_quantity"] == 0


def test_unknown_product_id_is_an_error(ctx):
    result, is_error = run(ctx, "get_product_details", {"product_id": 9999})
    assert is_error and "não encontrado" in result["error"]


# ----------------------------------------------------------------- promotions & payments

def test_only_active_promotions_are_listed_with_original_price(ctx):
    result, _ = run(ctx, "list_active_promotions", {})
    assert {p["product_id"] for p in result["active_promotions"]} == {90, 94, 121, 127}
    ohana = next(p for p in result["active_promotions"] if p["product_id"] == 121)
    assert ohana["active_promotion"] == {
        "promotion": "Desconto Exclusivo",
        "discount_percent": 20.0,
        "original_price_brl": 549.0,
        "promotional_price_brl": 439.2,
    }


def test_expired_promotion_is_reported_but_not_applied(ctx):
    result, _ = run(ctx, "get_product_details", {"product_id": 95})
    assert result["active_promotion"] is None
    assert result["current_price_brl"] == 2199.0
    assert result["expired_promotions"] == [{"promotion": "Oferta Especial", "discount_percent": 12.0}]


def test_pix_discount_applies_to_list_price(ctx):
    result, _ = run(ctx, "get_payment_options", {"product_id": 95})
    assert result["pix"]["total_brl"] == 2089.05
    assert result["combined_payment_allowed"] is True


def test_pix_discount_is_not_cumulative_with_promotion(ctx):
    result, _ = run(ctx, "get_payment_options", {"product_id": 121})
    assert result["price_is_promotional"] is True
    assert result["pix"]["total_brl"] == result["total_brl"] == 439.2


@pytest.mark.parametrize(
    ("amount", "max_installments"),
    [(1049.0, 10), (1200.0, 12), (450.0, 5), (159.9, 3), (120.0, 2), (49.0, 1)],
)
def test_installment_limits_follow_policy_bands(ctx, amount, max_installments):
    result, _ = run(ctx, "get_payment_options", {"amount_brl": amount})
    assert result["credit_card_interest_free"]["max_installments"] == max_installments


def test_payment_options_requires_exactly_one_subject(ctx):
    _, is_error = run(ctx, "get_payment_options", {})
    assert is_error
    _, is_error = run(ctx, "get_payment_options", {"product_id": 95, "amount_brl": 10})
    assert is_error


# ----------------------------------------------------------------- orders & privacy

def test_order_lookup_with_matching_phone(ctx):
    result, is_error = run(ctx, "get_order_status", {"order_id": 8, "phone": "(67) 99654-7821"})
    assert not is_error and result["verified"]
    assert result["status"] == "shipped" and result["tracking_code"] == "BRJL5544332BR"
    assert result["delivery_overdue_days"] == 248


@pytest.mark.parametrize("phone", ["67996547821", "+55 67 99654-7821", "99654-7821"])
def test_phone_formats_are_accepted(ctx, phone):
    result, _ = run(ctx, "get_order_status", {"order_id": 8, "phone": phone})
    assert result["verified"]


def test_order_result_never_contains_contact_data(ctx, store):
    result, _ = run(ctx, "get_order_status", {"order_id": 8, "phone": "67996547821"})
    serialized = json.dumps(result, ensure_ascii=False)
    customer = store.customers[2]
    assert customer.email not in serialized and customer.phone not in serialized
    assert customer.name not in serialized  # first name only
    assert result["customer_first_name"] == "Ana"


def test_wrong_phone_and_missing_order_give_the_same_answer(ctx):
    wrong_phone, _ = run(ctx, "get_order_status", {"order_id": 8, "phone": "67999990000"})
    missing_order, _ = run(ctx, "get_order_status", {"order_id": 999, "phone": "67996547821"})
    assert wrong_phone["error"] == missing_order["error"]
    assert "Ana" not in json.dumps(wrong_phone, ensure_ascii=False)


def test_cannot_read_another_customers_order_with_own_phone(ctx):
    # Customer 2 (order 8) tries order 1, which belongs to customer 3.
    result, is_error = run(ctx, "get_order_status", {"order_id": 1, "phone": "67996547821"})
    assert is_error and not result["verified"]


def test_lookups_lock_after_repeated_failures(ctx):
    for _ in range(MAX_FAILED_ORDER_LOOKUPS):
        run(ctx, "get_order_status", {"order_id": 1, "phone": "67900000000"})
    result, is_error = run(ctx, "get_order_status", {"order_id": 1, "phone": "(67) 98432-1098"})
    assert is_error and "Limite" in result["error"]


def test_short_phone_is_rejected_without_consuming_an_attempt(ctx):
    _, is_error = run(ctx, "get_order_status", {"order_id": 1, "phone": "1234"})
    assert is_error and ctx.failed_order_lookups == 0


def test_delivered_order_reports_return_windows(ctx):
    result, _ = run(ctx, "get_order_status", {"order_id": 1, "phone": "67984321098"})
    windows = result["policy_windows"]
    assert windows["receipt_date_is_estimate"] is True
    assert windows["regret_return_7d_after_receipt"] is False
    assert windows["legal_warranty_90d_after_receipt"] is False


def test_cancelled_order_exposes_cancellation_reason(ctx):
    result, _ = run(ctx, "get_order_status", {"order_id": 18, "phone": "67980987654"})
    assert result["status"] == "cancelled" and "fora de estoque" in result["notes"]


# ----------------------------------------------------------------- policies & escalation

def test_policy_tool_returns_requested_sections(ctx):
    result, is_error = run(ctx, "get_store_policy", {"topics": ["business_hours", "about_store"]})
    assert not is_error and set(result) == {"business_hours", "about_store"}


def test_policy_tool_rejects_unknown_topic(ctx):
    _, is_error = run(ctx, "get_store_policy", {"topics": ["refund_everything"]})
    assert is_error


def test_escalation_is_logged_with_protocol(ctx):
    result, is_error = run(ctx, "escalate_to_human", {"reason": "complaint", "summary": "Pedido 8 atrasado.", "order_id": 8})
    assert not is_error and result["protocol"].startswith("EM-20261103-")
    logged = json.loads((ctx.runtime_dir / "escalations.jsonl").read_text(encoding="utf-8"))
    assert logged["order_id"] == 8 and logged["reason"] == "complaint"


# ----------------------------------------------------------------- input validation

@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("get_product_details", {"product_id": "abc"}),
        ("get_product_details", {"product_id": True}),
        ("get_product_details", {}),
        ("search_products", {"max_price_brl": -5}),
        ("search_products", {"query": "x" * 500}),
        ("search_products", {"available_only": "sim"}),
        ("get_order_status", {"order_id": 1}),
        ("escalate_to_human", {"reason": "hack", "summary": "x"}),
        ("drop_tables", {}),
    ],
)
def test_malformed_tool_calls_return_errors_instead_of_raising(ctx, name, args):
    result, is_error = run(ctx, name, args)
    assert is_error and "error" in result
