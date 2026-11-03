"""Agent-loop tests with a scripted fake Claude client (no network, no API key)."""

from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from emporio_agent.agent import FALLBACK_BETA, REFUSAL_MESSAGE, SERVICE_UNAVAILABLE_MESSAGE, StoreAgent


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(name, tool_input, block_id="toolu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=block_id)


def response(stop_reason, *blocks):
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks))


class FakeClient:
    """Mimics client.beta.messages.create, returning scripted responses in order."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def make_agent(settings, store):
    def factory(*responses):
        client = FakeClient(*responses)
        return StoreAgent(client, settings, store), client
    return factory


def test_plain_answer_without_tools(make_agent):
    agent, client = make_agent(response("end_turn", text_block("Oi! Como posso ajudar?")))
    reply = agent.reply("oi")
    assert reply.text == "Oi! Como posso ajudar?" and reply.tool_calls == []
    assert len(client.calls) == 1


def test_tool_call_result_is_fed_back_to_the_model(make_agent):
    agent, client = make_agent(
        response("tool_use", tool_block("search_products", {"query": "Takamine GD20"})),
        response("end_turn", text_block("O Takamine GD20 sai por R$ 2.199,00.")),
    )
    reply = agent.reply("Quanto custa o Takamine GD20?")

    assert reply.tool_calls[0].name == "search_products"
    assert reply.tool_calls[0].result["products"][0]["current_price_brl"] == 2199.0
    tool_result = client.calls[1]["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "toolu_1"
    assert '"current_price_brl": 2199.0' in tool_result["content"]
    assert tool_result["is_error"] is False


def test_parallel_tool_calls_return_in_a_single_user_message(make_agent):
    agent, client = make_agent(
        response(
            "tool_use",
            tool_block("get_store_policy", {"topics": ["returns_and_exchanges"]}, "toolu_a"),
            tool_block("get_order_status", {"order_id": 1, "phone": "67984321098"}, "toolu_b"),
        ),
        response("end_turn", text_block("...")),
    )
    agent.reply("Posso devolver o pedido 1? Meu telefone é (67) 98432-1098")
    results = client.calls[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["toolu_a", "toolu_b"]


def test_invalid_tool_input_is_reported_as_tool_error(make_agent):
    agent, client = make_agent(
        response("tool_use", tool_block("get_product_details", {"product_id": "abc"})),
        response("end_turn", text_block("Pode me passar o modelo?")),
    )
    reply = agent.reply("detalhes do produto abc")
    assert reply.tool_calls[0].is_error
    assert client.calls[1]["messages"][-1]["content"][0]["is_error"] is True


def test_request_uses_cache_effort_and_refusal_fallback(make_agent):
    agent, client = make_agent(response("end_turn", text_block("ok")))
    agent.reply("oi")
    call = client.calls[0]
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "03/11/2026 14:00" in call["system"][0]["text"]
    assert call["output_config"] == {"effort": "medium"}
    assert call["betas"] == [FALLBACK_BETA] and call["fallbacks"] == "default"
    assert {t["name"] for t in call["tools"]} >= {"search_products", "get_order_status", "get_store_policy"}


def test_history_is_kept_across_turns(make_agent):
    agent, client = make_agent(
        response("end_turn", text_block("Prazer, Ana!")),
        response("end_turn", text_block("Seu nome é Ana.")),
    )
    agent.reply("Meu nome é Ana")
    agent.reply("Qual é o meu nome?")
    roles = [m["role"] for m in client.calls[1]["messages"]]
    assert roles == ["user", "assistant", "user"]


def test_api_failure_returns_friendly_message_and_rolls_back_turn(make_agent):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    agent, _ = make_agent(anthropic.APIConnectionError(request=request))
    reply = agent.reply("oi")
    assert reply.failed and reply.text == SERVICE_UNAVAILABLE_MESSAGE
    assert agent.messages == []  # no dangling user message


def test_refusal_is_handled_without_breaking_history(make_agent):
    agent, _ = make_agent(response("refusal"))
    reply = agent.reply("algo proibido")
    assert reply.text == REFUSAL_MESSAGE and agent.messages == []


def test_runaway_tool_loop_is_capped(make_agent):
    loops = [response("tool_use", tool_block("list_active_promotions", {}, f"t{i}")) for i in range(10)]
    agent, client = make_agent(*loops)
    reply = agent.reply("promoções?")
    assert reply.failed and len(client.calls) == 8
    assert agent.messages == []
