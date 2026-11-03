"""Deterministic tools the agent can call.

Every number the customer sees (price, stock, discount, installment, order status)
comes from these functions, never from the model. Each tool validates its input and
returns a JSON-serializable dict; errors are returned as {"error": ...} so the model
can recover instead of the turn crashing.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .data_store import DataStore, Product, normalize, only_digits
from .policies import POLICY_SECTIONS, POLICY_TOPICS

MAX_FAILED_ORDER_LOOKUPS = 3
PIX_DISCOUNT = 0.05
COMBINED_PAYMENT_MIN_TOTAL = 2000.0
# (max installments in the band, minimum installment value) — policy section 3.1
INSTALLMENT_BANDS = ((3, 50.0), (6, 80.0), (12, 100.0))

STOPWORDS = {
    "a", "o", "as", "os", "de", "da", "do", "das", "dos", "e", "em", "um", "uma",
    "para", "pra", "com", "por", "ate", "que", "qual", "quais", "me", "tem", "voces",
}

ORDER_STATUS_LABELS = {
    "pending": "Aguardando pagamento/processamento",
    "confirmed": "Confirmado (em separação)",
    "shipped": "Enviado",
    "delivered": "Entregue",
    "cancelled": "Cancelado",
}

AVAILABILITY_LABELS = {
    "in_stock": "Em estoque",
    "out_of_stock": "Esgotado no momento",
    "discontinued": "Fora de linha (descontinuado)",
    "coming_soon": "Em breve (ainda não disponível para venda)",
}


class ToolInputError(ValueError):
    """Raised when the model sends arguments that fail validation."""


@dataclass
class ToolContext:
    store: DataStore
    now: Callable[[], datetime]
    runtime_dir: Path
    failed_order_lookups: int = 0


# --------------------------------------------------------------------------- helpers

def availability(product: Product) -> str:
    if product.status == "discontinued":
        return "discontinued"
    if product.status == "coming_soon":
        return "coming_soon"
    return "in_stock" if product.stock_quantity > 0 else "out_of_stock"


def effective_price(store: DataStore, product: Product) -> tuple[float, dict | None]:
    promo = store.active_promotion(product.product_id)
    if promo is None:
        return product.price_brl, None
    promo_price = round(product.price_brl * (1 - promo.discount_percent / 100), 2)
    return promo_price, {
        "promotion": promo.description,
        "discount_percent": promo.discount_percent,
        "original_price_brl": product.price_brl,
        "promotional_price_brl": promo_price,
    }


def product_summary(store: DataStore, product: Product) -> dict[str, Any]:
    price, promo = effective_price(store, product)
    status = availability(product)
    return {
        "product_id": product.product_id,
        "name": product.name,
        "category": store.category_name(product.category_id),
        "list_price_brl": product.price_brl,
        "current_price_brl": price,
        "active_promotion": promo,
        "availability": status,
        "availability_label": AVAILABILITY_LABELS[status],
        "stock_quantity": product.stock_quantity if status == "in_stock" else 0,
    }


def _query_variants(token: str) -> set[str]:
    """Cheap Portuguese plural folding: violoes -> violao, teclados -> teclado."""
    variants = {token}
    if token.endswith("oes") or token.endswith("aes"):
        variants.add(token[:-3] + "ao")
    if token.endswith("ao"):
        variants.add(token[:-2] + "oes")  # violao also matches the category "Violões"
    if token.endswith("ns"):
        variants.add(token[:-2] + "m")
    if token.endswith("s") and len(token) > 3:
        variants.add(token[:-1])
    return variants


def _token_in(variants: set[str], words: set[str]) -> bool:
    return bool(variants & words) or any(len(v) >= 4 and any(w.startswith(v) for w in words) for v in variants)


def _match_score(tokens: list[str], product: Product) -> tuple[int, int]:
    """(tokens matched anywhere, tokens matched in name/category) — the second breaks ties."""
    title_words = set(product.title_text.split())
    all_words = set(product.search_text.split())
    matched = in_title = 0
    for token in tokens:
        variants = _query_variants(token)
        if _token_in(variants, all_words):
            matched += 1
            in_title += _token_in(variants, title_words)
    return matched, in_title


def _require_int(args: dict, key: str, *, minimum: int = 1, maximum: int = 10**6, required: bool = True) -> int | None:
    value = args.get(key)
    if value is None:
        if required:
            raise ToolInputError(f"'{key}' é obrigatório.")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise ToolInputError(f"'{key}' deve ser um número inteiro.")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ToolInputError(f"'{key}' deve estar entre {minimum} e {maximum}.")
    return value


def _optional_number(args: dict, key: str) -> float | None:
    value = args.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ToolInputError(f"'{key}' deve ser um número positivo.")
    return float(value)


def _optional_text(args: dict, key: str, max_len: int = 200) -> str | None:
    value = args.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ToolInputError(f"'{key}' deve ser texto.")
    value = value.strip()
    if len(value) > max_len:
        raise ToolInputError(f"'{key}' excede {max_len} caracteres.")
    return value or None


# --------------------------------------------------------------------------- tools

def search_products(ctx: ToolContext, args: dict) -> dict:
    store = ctx.store
    query = _optional_text(args, "query")
    category = _optional_text(args, "category", max_len=60)
    min_price = _optional_number(args, "min_price_brl")
    max_price = _optional_number(args, "max_price_brl")
    available_only = args.get("available_only", True)
    if not isinstance(available_only, bool):
        raise ToolInputError("'available_only' deve ser true ou false.")
    limit = _require_int(args, "limit", maximum=20, required=False) or 10

    category_id = None
    if category:
        matches = [c for c in store.categories.values() if normalize(c.name) == normalize(category)]
        if not matches:
            return {
                "error": f"Categoria '{category}' não existe.",
                "valid_categories": [c.name for c in store.categories.values()],
            }
        category_id = matches[0].category_id

    tokens = [t for t in normalize(query).split() if t not in STOPWORDS] if query else []
    candidates = []
    for product in store.products.values():
        if category_id is not None and product.category_id != category_id:
            continue
        if available_only and availability(product) != "in_stock":
            continue
        price, _ = effective_price(store, product)
        if min_price is not None and price < min_price:
            continue
        if max_price is not None and price > max_price:
            continue
        score = _match_score(tokens, product) if tokens else (0, 0)
        candidates.append((score, price, product))

    exact_match = True
    if tokens:
        full = [c for c in candidates if c[0][0] == len(tokens)]
        if full:
            candidates = full
        else:
            # Keep only the strongest partial matches instead of everything that shares one word.
            exact_match = False
            best = max((c[0] for c in candidates if c[0][0] > 0), default=None)
            candidates = [c for c in candidates if c[0] == best] if best else []

    candidates.sort(key=lambda c: (-c[0][0], -c[0][1], c[1]))
    result: dict[str, Any] = {
        "total_matches": len(candidates),
        "returned": min(len(candidates), limit),
        "products": [product_summary(store, p) for _, _, p in candidates[:limit]],
    }
    if not candidates:
        result["note"] = "Nenhum produto encontrado com esses critérios."
    elif tokens and not exact_match:
        result["note"] = "Nenhum produto contém todos os termos; resultados são correspondências parciais."
    if category_id is not None and not any(p.category_id == category_id for p in store.products.values()):
        result["note"] = "Não há produtos cadastrados nessa categoria no momento."
    return result


def get_product_details(ctx: ToolContext, args: dict) -> dict:
    store = ctx.store
    product_id = _require_int(args, "product_id")
    product = store.products.get(product_id)
    if product is None:
        return {"error": f"Produto {product_id} não encontrado no catálogo."}

    details = product_summary(store, product)
    details["description"] = product.description
    details["specs"] = product.specs
    expired = [
        {"promotion": p.description, "discount_percent": p.discount_percent}
        for p in store.promotions
        if p.product_id == product_id and not p.is_active
    ]
    if expired:
        details["expired_promotions"] = expired

    if details["availability"] != "in_stock":
        details["alternatives"] = _alternatives(store, product)
    return details


def _alternatives(store: DataStore, product: Product, limit: int = 3) -> list[dict]:
    """Closest-priced in-stock products in the same category (policy 7.3)."""
    pool = [
        p for p in store.products.values()
        if p.category_id == product.category_id and p.product_id != product.product_id and availability(p) == "in_stock"
    ]
    pool.sort(key=lambda p: abs(p.price_brl - product.price_brl))
    return [product_summary(store, p) for p in pool[:limit]]


def list_active_promotions(ctx: ToolContext, args: dict) -> dict:
    store = ctx.store
    promos = []
    seen = set()
    for promo in store.promotions:
        if not promo.is_active or promo.product_id in seen:
            continue
        product = store.products.get(promo.product_id)
        if product is None:
            continue
        seen.add(promo.product_id)
        promos.append(product_summary(store, product))
    promos.sort(key=lambda p: -p["active_promotion"]["discount_percent"])
    return {"active_promotions": promos, "count": len(promos)}


def get_payment_options(ctx: ToolContext, args: dict) -> dict:
    store = ctx.store
    product_id = _require_int(args, "product_id", required=False)
    amount = _optional_number(args, "amount_brl")
    quantity = _require_int(args, "quantity", maximum=50, required=False) or 1

    if (product_id is None) == (amount is None):
        raise ToolInputError("Informe exatamente um entre 'product_id' e 'amount_brl'.")

    is_promotional = False
    subject: dict[str, Any] = {}
    if product_id is not None:
        product = store.products.get(product_id)
        if product is None:
            return {"error": f"Produto {product_id} não encontrado no catálogo."}
        unit_price, promo = effective_price(store, product)
        is_promotional = promo is not None
        total = round(unit_price * quantity, 2)
        subject = {"product_id": product_id, "name": product.name, "quantity": quantity, "unit_price_brl": unit_price}
    else:
        if amount <= 0:
            raise ToolInputError("'amount_brl' deve ser maior que zero.")
        total = round(amount, 2)

    credit_options = []
    for installments in range(1, 13):
        minimum = next(m for limit, m in INSTALLMENT_BANDS if installments <= limit)
        value = round(total / installments, 2)
        if installments == 1 or value >= minimum:
            credit_options.append({"installments": installments, "installment_value_brl": value})

    pix_total = total if is_promotional else round(total * (1 - PIX_DISCOUNT), 2)
    return {
        **subject,
        "total_brl": total,
        "price_is_promotional": is_promotional,
        "pix": {
            "total_brl": pix_total,
            "discount_applied": not is_promotional,
            "note": "Desconto de 5% no PIX não é cumulativo com promoções." if is_promotional else "5% de desconto à vista no PIX.",
        },
        "debit_card": {"total_brl": total},
        "boleto": {"total_brl": total, "note": "Compensação em até 3 dias úteis."},
        "credit_card_interest_free": {
            "max_installments": credit_options[-1]["installments"],
            "options": credit_options,
            "rule": "Até 3x: parcela ≥ R$ 50; 4x–6x: parcela ≥ R$ 80; 7x–12x: parcela ≥ R$ 100.",
        },
        "combined_payment_allowed": total > COMBINED_PAYMENT_MIN_TOTAL,
    }


def get_order_status(ctx: ToolContext, args: dict) -> dict:
    """Order lookup gated by a second factor (phone on file) to protect customer data."""
    if ctx.failed_order_lookups >= MAX_FAILED_ORDER_LOOKUPS:
        return {
            "error": "Limite de tentativas de verificação atingido nesta conversa.",
            "action": "Encaminhe o cliente ao atendimento humano pelo WhatsApp (67) 3321-4500.",
        }

    order_id = _require_int(args, "order_id")
    phone_digits = only_digits(_optional_text(args, "phone", max_len=30) or "")
    if phone_digits.startswith("55") and len(phone_digits) >= 12:
        phone_digits = phone_digits[2:]
    if len(phone_digits) < 9:
        raise ToolInputError("'phone' deve ter ao menos 9 dígitos (com ou sem DDD).")

    store = ctx.store
    order = store.orders.get(order_id)
    customer = store.customers.get(order.customer_id) if order else None
    if customer is None or not only_digits(customer.phone).endswith(phone_digits):
        ctx.failed_order_lookups += 1
        # Same message whether the order exists or not: avoids leaking which order numbers are valid.
        return {
            "verified": False,
            "error": "Não encontramos um pedido com esse número e telefone.",
            "attempts_left": MAX_FAILED_ORDER_LOOKUPS - ctx.failed_order_lookups,
        }

    today = ctx.now().date()
    items = []
    for item in order.items:
        product = store.products.get(item.product_id)
        items.append({"product_id": item.product_id, "name": product.name if product else "Produto não cadastrado", "quantity": item.quantity})

    result: dict[str, Any] = {
        "verified": True,
        "order_id": order.order_id,
        "customer_first_name": customer.name.split()[0],
        "status": order.status,
        "status_label": ORDER_STATUS_LABELS.get(order.status, order.status),
        "order_date": order.order_date.isoformat(),
        "days_since_order": (today - order.order_date).days,
        "total_brl": order.total_brl,
        "payment_method": _payment_label(order.payment_method),
        "items": items,
        "tracking_code": order.tracking_code,
        "estimated_delivery": order.estimated_delivery.isoformat() if order.estimated_delivery else None,
        "notes": order.notes,
    }
    if order.status == "shipped" and order.estimated_delivery and today > order.estimated_delivery:
        result["delivery_overdue_days"] = (today - order.estimated_delivery).days
    if order.status == "delivered":
        result["policy_windows"] = _policy_windows(order, today)
    return result


def _policy_windows(order, today) -> dict:
    """Return/warranty windows. The data has no real receipt date, so the estimated
    delivery date is used as a proxy for 'received' (documented assumption)."""
    received = order.estimated_delivery or order.order_date
    since_received = (today - received).days
    since_purchase = (today - order.order_date).days
    return {
        "assumed_receipt_date": received.isoformat(),
        "receipt_date_is_estimate": True,
        "days_since_receipt": since_received,
        "days_since_purchase": since_purchase,
        "regret_return_7d_after_receipt": since_received <= 7,
        "preference_exchange_7d": since_received <= 7,
        "defect_exchange_30d_after_purchase": since_purchase <= 30,
        "legal_warranty_90d_after_receipt": since_received <= 90,
    }


def _payment_label(method: str) -> str:
    if method.startswith("credit_"):
        return f"Cartão de crédito em {method.removeprefix('credit_')}"
    return {"pix": "PIX", "debit": "Cartão de débito", "boleto": "Boleto bancário"}.get(method, method)


def get_store_policy(ctx: ToolContext, args: dict) -> dict:
    topics = args.get("topics")
    if not isinstance(topics, list) or not topics:
        raise ToolInputError(f"'topics' deve ser uma lista não vazia com valores de {list(POLICY_TOPICS)}.")
    unknown = [t for t in topics if t not in POLICY_SECTIONS]
    if unknown:
        raise ToolInputError(f"Tópicos inválidos: {unknown}. Válidos: {list(POLICY_TOPICS)}.")
    return {topic: POLICY_SECTIONS[topic] for topic in dict.fromkeys(topics)}


def escalate_to_human(ctx: ToolContext, args: dict) -> dict:
    reason = args.get("reason")
    if reason not in ESCALATION_REASONS:
        raise ToolInputError(f"'reason' deve ser um de {list(ESCALATION_REASONS)}.")
    summary = _optional_text(args, "summary", max_len=1000)
    if not summary:
        raise ToolInputError("'summary' é obrigatório.")
    order_id = _require_int(args, "order_id", required=False)

    now = ctx.now()
    ticket = {
        "protocol": f"EM-{now:%Y%m%d}-{secrets.token_hex(3).upper()}",
        "created_at": now.isoformat(timespec="seconds"),
        "reason": reason,
        "summary": summary,
        "order_id": order_id,
    }
    ctx.runtime_dir.mkdir(parents=True, exist_ok=True)
    with (ctx.runtime_dir / "escalations.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(ticket, ensure_ascii=False) + "\n")
    return {"registered": True, "protocol": ticket["protocol"], "response_time": "Retorno da equipe em até 24 horas úteis."}


ESCALATION_REASONS = ("complaint", "order_issue", "return_or_exchange_request", "policy_question", "other")

# --------------------------------------------------------------------------- schemas

TOOL_HANDLERS: dict[str, Callable[[ToolContext, dict], dict]] = {
    "search_products": search_products,
    "get_product_details": get_product_details,
    "list_active_promotions": list_active_promotions,
    "get_payment_options": get_payment_options,
    "get_order_status": get_order_status,
    "get_store_policy": get_store_policy,
    "escalate_to_human": escalate_to_human,
}


def tool_definitions(store: DataStore) -> list[dict]:
    categories = [c.name for c in store.categories.values()]
    return [
        {
            "name": "search_products",
            "description": (
                "Busca instrumentos no catálogo da loja com preço atual, promoção ativa e disponibilidade em tempo real. "
                "Use para qualquer pergunta sobre produtos, preços, marcas, modelos ou faixas de preço. "
                "Por padrão retorna só itens disponíveis para compra; use available_only=false para checar um modelo específico "
                "que pode estar esgotado, fora de linha ou em breve. Filtros de preço usam o preço atual (com promoção)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Termos livres: marca, modelo, tipo (ex.: 'Takamine GD20', 'violão nylon elétrico')."},
                    "category": {"type": "string", "enum": categories, "description": "Filtra por categoria."},
                    "min_price_brl": {"type": "number", "description": "Preço mínimo em reais."},
                    "max_price_brl": {"type": "number", "description": "Preço máximo em reais."},
                    "available_only": {"type": "boolean", "description": "Se true (padrão), só itens em estoque e à venda."},
                    "limit": {"type": "integer", "description": "Máximo de itens (1-20, padrão 10)."},
                },
                "required": [],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_product_details",
            "description": (
                "Detalhes de um produto pelo product_id: descrição, especificações, preço, promoção ativa, "
                "promoções expiradas e, se indisponível, alternativas semelhantes em estoque."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"product_id": {"type": "integer"}},
                "required": ["product_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "list_active_promotions",
            "description": "Lista todas as promoções vigentes no sistema, com preço original, percentual e preço promocional.",
            "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        },
        {
            "name": "get_payment_options",
            "description": (
                "Calcula as condições de pagamento (PIX com desconto, débito, boleto, parcelas sem juros no crédito "
                "e se pode combinar formas). Informe product_id (usa o preço atual) ou amount_brl, nunca os dois."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "integer"},
                    "quantity": {"type": "integer", "description": "Quantidade (padrão 1), usada com product_id."},
                    "amount_brl": {"type": "number", "description": "Valor total em reais, quando não houver um produto específico."},
                },
                "required": [],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_order_status",
            "description": (
                "Consulta um pedido. Exige o número do pedido E o telefone cadastrado do cliente (verificação de identidade). "
                "Só chame depois que o cliente informar os dois na conversa; nunca invente ou presuma o telefone."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "integer"},
                    "phone": {"type": "string", "description": "Telefone informado pelo cliente, com ou sem DDD."},
                },
                "required": ["order_id", "phone"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_store_policy",
            "description": (
                "Retorna o texto oficial do manual de políticas da loja para um ou mais tópicos. Use para endereço, contato, "
                "horários, pagamento, trocas/devoluções, frete/entrega/rastreamento, regras de promoção, garantia, privacidade "
                "e procedimentos de atendimento. Responda com base no texto retornado."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "topics": {"type": "array", "items": {"type": "string", "enum": list(POLICY_TOPICS)}, "minItems": 1},
                },
                "required": ["topics"],
                "additionalProperties": False,
            },
        },
        {
            "name": "escalate_to_human",
            "description": (
                "Registra um atendimento para a equipe humana e retorna um protocolo. Use para reclamações, problemas com "
                "pedido (atraso, avaria, cobrança), pedidos de troca/devolução que precisam ser processados, ou quando a "
                "política exigir consulta à gerência. Não inclua telefone ou e-mail no resumo."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "enum": list(ESCALATION_REASONS)},
                    "summary": {"type": "string", "description": "Resumo objetivo do caso para a equipe."},
                    "order_id": {"type": "integer"},
                },
                "required": ["reason", "summary"],
                "additionalProperties": False,
            },
        },
    ]


def execute_tool(ctx: ToolContext, name: str, args: dict) -> tuple[dict, bool]:
    """Run a tool by name. Returns (result, is_error)."""
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return {"error": f"Ferramenta desconhecida: {name}"}, True
    if not isinstance(args, dict):
        return {"error": "Argumentos devem ser um objeto JSON."}, True
    try:
        result = handler(ctx, args)
    except ToolInputError as exc:
        return {"error": f"Entrada inválida: {exc}"}, True
    return result, "error" in result
