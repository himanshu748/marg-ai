import io
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from marg.agent.llm import ProviderUnavailable
from marg.agent.modal_llm import MODEL, Allowance, ModalLLM, messages_for_modal
from marg.agent.tools import TOOL_SPECS


def test_allowance_is_atomic_and_persistent(tmp_path):
    path = tmp_path / "allowance.sqlite3"
    budget = Allowance(path, limit=6, expires=9999999999)

    def reserve(_):
        try:
            budget.reserve(10, 0)
            return True
        except ProviderUnavailable:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(reserve, range(20))) == 2
    assert Allowance(path, limit=6).status()["reserved_micro_usd"] == 6


def test_expired_allowance_refuses_before_network(tmp_path):
    with pytest.raises(ProviderUnavailable, match="expired"):
        Allowance(tmp_path / "budget.sqlite3", expires=1).reserve(10, 1)


def test_modal_tool_history_preserves_ids_and_results():
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "c1",
                        "name": "finalize",
                        "input": {"summary": "Checked"},
                    }
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "c1",
                        "content": [{"json": {"status": "finalized"}}],
                    }
                }
            ],
        },
    ]
    out = messages_for_modal(messages)
    assert out[-2]["tool_calls"][0]["id"] == out[-1]["tool_call_id"] == "c1"
    assert json.loads(out[-1]["content"])[0]["json"]["status"] == "finalized"


@pytest.fixture
def provider(monkeypatch, tmp_path):
    monkeypatch.setenv("MARG_MODAL_ENDPOINT", "https://test.us-west.modal.direct")
    monkeypatch.setenv("MARG_MODAL_PROXY_TOKEN", "local-test-only")
    monkeypatch.setenv("MARG_MODAL_ENABLED", "1")
    return ModalLLM(Allowance(tmp_path / "budget.sqlite3", expires=9999999999))


def response(arguments='{"summary":"Checked"}', name="finalize", finish="tool_calls"):
    stream = io.BytesIO(
        json.dumps(
            {
                "model": MODEL,
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                "choices": [
                    {
                        "finish_reason": finish,
                        "message": {
                            "content": "Evidence checked.",
                            "reasoning_content": "must not be exposed",
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "function": {"name": name, "arguments": arguments},
                                }
                            ],
                        },
                    }
                ],
            }
        ).encode()
    )
    stream.headers = {"x-request-id": "request-test"}
    return stream


def test_real_provider_contract_and_usage(monkeypatch, provider):
    sent = []

    def open_request(request, timeout):
        sent.append(json.loads(request.data))
        return response()

    monkeypatch.setattr("marg.agent.modal_llm.urlopen", open_request)
    result = provider.converse([], TOOL_SPECS)
    assert result.tool_calls[0].name == "finalize"
    assert provider.details()["live_inference"] is True
    assert provider.details()["requests"][0]["prompt_tokens"] == 100
    assert sent[0]["max_tokens"] == 1024 and sent[0]["reasoning_effort"] == "none"
    assert "must not be exposed" not in str(result)


@pytest.mark.parametrize(
    "arguments,name,finish",
    [
        ("[1]", "finalize", "tool_calls"),
        ("{}", "shell", "tool_calls"),
        ("{}", "finalize", "length"),
    ],
)
def test_invalid_provider_output_fails_closed(
    monkeypatch, provider, arguments, name, finish
):
    monkeypatch.setattr(
        "marg.agent.modal_llm.urlopen",
        lambda *a, **kw: response(arguments, name, finish),
    )
    with pytest.raises(ProviderUnavailable):
        provider.converse([], TOOL_SPECS)
    assert provider.live_inference is False
    assert provider.allowance.status()["requests"] == 1


def test_context_limit_does_not_consume_allowance(provider):
    with pytest.raises(ProviderUnavailable, match="context"):
        provider.converse(
            [{"role": "user", "content": [{"text": "x" * 70000}]}], TOOL_SPECS
        )
    assert provider.allowance.status()["requests"] == 0
