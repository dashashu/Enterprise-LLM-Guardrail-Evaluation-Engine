"""Tests for the optional, human-readable evaluation report."""

import json
import sys
from types import SimpleNamespace

import pytest

from app import config as config_module
from app.config import Settings
from app.provider import ProviderError
from app.schemas import JudgeScore
import evals.run as eval_runner


def test_markdown_report_has_metrics_and_omits_sensitive_case_content():
    summary = {
        "cases": 2,
        "degraded_count": 1,
        "exact_match_rate": 0.5,
        "faithfulness_mean": 0.75,
        "completeness_mean": 0.8,
        "human_review_count": 1,
        "by_category": {
            "safe | <script>": {"cases": 1, "exact_match_rate": 1.0, "human_review_count": 0},
            "general": {"cases": 1, "exact_match_rate": 0.0, "human_review_count": 1},
        },
        "by_risk": {
            "normal": {"cases": 1, "exact_match_rate": 1.0, "human_review_count": 0},
            "high": {"cases": 1, "exact_match_rate": 0.0, "human_review_count": 1},
        },
        "output": "evals/results.jsonl",
    }
    records = [
        {"id": "case|one\nnext", "exact_match": True, "source": "model",
         "human_review": False, "judge": {"faithfulness": 0.75, "completeness": 0.8,
                                         "rationale": "PRIVATE_JUDGE_RATIONALE"},
         "answer": "PRIVATE_ANSWER", "expected": "PRIVATE_EXPECTED",
         "context": "PRIVATE_CONTEXT"},
        {"id": "case-two", "exact_match": False, "source": "fallback",
         "human_review": True, "judge": None, "answer": "PRIVATE_FALLBACK"},
    ]

    report = eval_runner.render_markdown_report(summary, records)

    assert "| Exact match rate | 50.0% |" in report
    assert "| Degraded responses | 1 |" in report
    assert "| Judged cases | 1/2 |" in report
    assert "| Mean faithfulness (0–1) | 0.75 |" in report
    assert "| Mean completeness (0–1) | 0.80 |" in report
    assert "## By category" in report and "## By risk" in report
    assert "safe \\| &lt;script&gt;" in report
    assert "| case\\|one next | Yes | 0.75 | 0.80 | model | No |" in report
    assert "| case-two | No | — | — | fallback | Yes |" in report
    for private in ("PRIVATE_ANSWER", "PRIVATE_EXPECTED", "PRIVATE_CONTEXT",
                    "PRIVATE_JUDGE_RATIONALE", "PRIVATE_FALLBACK"):
        assert private not in report


@pytest.mark.asyncio
async def test_run_writes_optional_report_and_preserves_jsonl_and_summary(tmp_path, monkeypatch):
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text("\n".join(json.dumps(case) for case in [
        {"id": "one", "prompt": "Prompt one", "context": "Sensitive context",
         "expected": "Sensitive answer", "category": "factual", "risk": "normal"},
        {"id": "two", "prompt": "Prompt two", "expected": "Other answer",
         "category": "outage", "risk": "high"},
    ]))

    settings = Settings(provider="openai", model="test-model", api_key="test",
                        app_api_keys=("test-client",), redis_url="redis://localhost",
                        rate_limit_per_minute=2, llm_timeout_seconds=1,
                        request_deadline_seconds=2, llm_max_retries=0,
                        cache_ttl_seconds=60)
    monkeypatch.setattr(eval_runner.Settings, "from_env", lambda *, require_app_api_keys: settings)

    class FakeService:
        async def generate(self, request, client_id, check_rate):
            assert client_id == "evaluation" and check_rate is False
            if request.prompt == "Prompt one":
                return SimpleNamespace(answer="Sensitive answer", source="model", request_id="r1")
            return SimpleNamespace(answer="Fallback answer", source="fallback", request_id="r2")

    async def fake_judge(case, answer, provider, configured_settings):
        assert case.id == "one" and answer == "Sensitive answer"
        return JudgeScore(faithfulness=0.9, completeness=0.8, rationale="Private rationale")

    monkeypatch.setattr(eval_runner, "GenerationService", lambda *args: FakeService())
    monkeypatch.setattr(eval_runner, "judge", fake_judge)

    output_path = tmp_path / "artifacts" / "results.jsonl"
    report_path = tmp_path / "artifacts" / "report.md"
    summary_path = tmp_path / "artifacts" / "summary.json"
    summary = await eval_runner.run(cases_path, output_path, 0.0, 100, report_path, summary_path)

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [record["answer"] for record in records] == ["Sensitive answer", "Fallback answer"]
    assert records[0]["judge"]["rationale"] == "Private rationale"
    assert set(summary) == {"cases", "mode", "degraded_count", "exact_match_rate", "faithfulness_mean",
                            "completeness_mean", "human_review_count", "by_category",
                            "by_risk", "output"}
    assert summary["mode"] == "live"
    assert summary["exact_match_rate"] == 0.5
    assert summary["degraded_count"] == 1
    assert summary["human_review_count"] == 1
    assert json.loads(summary_path.read_text()) == summary
    report = report_path.read_text()
    assert "| one | Yes | 0.90 | 0.80 | model | No |" in report
    assert "| two | No | — | — | fallback | Yes |" in report
    assert "Sensitive answer" not in report
    assert "Sensitive context" not in report
    assert "Private rationale" not in report


@pytest.mark.asyncio
async def test_report_cannot_replace_jsonl_output(tmp_path):
    same_path = tmp_path / "results.jsonl"
    with pytest.raises(ValueError, match="different paths"):
        await eval_runner.run(tmp_path / "cases.jsonl", same_path, 0.0, 1, same_path)


@pytest.mark.asyncio
async def test_summary_cannot_replace_cases_or_results(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "results.jsonl"
    for unsafe_summary in (cases_path, output_path):
        with pytest.raises(ValueError, match="different paths"):
            await eval_runner.run(cases_path, output_path, 0.0, 1,
                                  summary_path=unsafe_summary)


@pytest.mark.asyncio
async def test_eval_uses_fresh_model_calls_without_redis_or_cached_fallback(tmp_path, monkeypatch):
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text("\n".join(json.dumps({"id": case_id, "prompt": "Same question",
                                                   "expected": "Fresh answer"})
                                     for case_id in ("first", "second")))
    settings = Settings(provider="openai", model="test-model", api_key="test", app_api_keys=(),
                        redis_url="redis://unavailable", rate_limit_per_minute=2,
                        llm_timeout_seconds=1, request_deadline_seconds=2,
                        llm_max_retries=0, cache_ttl_seconds=60)
    monkeypatch.setattr(eval_runner.Settings, "from_env", lambda *, require_app_api_keys: settings)

    class FakeProvider:
        calls = 0

        def __init__(self, settings, client):
            pass

        async def complete(self, messages, timeout):
            type(self).calls += 1
            if type(self).calls == 1:
                return '{"answer":"Fresh answer"}'
            raise ProviderError("model unavailable")

    async def fake_judge(case, answer, provider, settings):
        return JudgeScore(faithfulness=1, completeness=1, rationale="ok")

    monkeypatch.setattr(eval_runner, "HTTPProvider", FakeProvider)
    monkeypatch.setattr(eval_runner, "judge", fake_judge)
    output_path = tmp_path / "results.jsonl"
    await eval_runner.run(cases_path, output_path, 0, 2)
    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [record["source"] for record in records] == ["model", "fallback"]
    assert FakeProvider.calls == 2


def test_cli_passes_report_and_summary_paths_and_prints_json(tmp_path, monkeypatch, capsys):
    received = []

    async def fake_run(*args):
        received.append(args)
        return {"cases": 1, "output": str(args[1])}

    monkeypatch.setattr(eval_runner, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["evals.run", "--cases", str(tmp_path / "cases.jsonl"),
                                     "--output", str(tmp_path / "results.jsonl"),
                                     "--report", str(tmp_path / "report.md"),
                                     "--summary", str(tmp_path / "summary.json")])
    eval_runner.main()

    assert received[0][-2:] == (tmp_path / "report.md", tmp_path / "summary.json")
    assert json.loads(capsys.readouterr().out) == {"cases": 1, "output": str(tmp_path / "results.jsonl")}


def test_cli_missing_provider_config_is_actionable_and_creates_no_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config_module, "_DOTENV_PATH", tmp_path / ".env")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("MODERATION_MODE", "local")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("APP_API_KEYS", raising=False)
    summary_path = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", ["evals.run", "--cases", str(tmp_path / "cases.jsonl"),
                                     "--output", str(tmp_path / "results.jsonl"),
                                     "--summary", str(summary_path)])

    with pytest.raises(SystemExit) as exc:
        eval_runner.main()

    assert exc.value.code == 2
    output = capsys.readouterr()
    assert not output.out
    assert "OPENAI_API_KEY" in output.err and "LLM_MODEL" in output.err
    assert "APP_API_KEYS" not in output.err
    assert "Traceback" not in output.err
    assert not summary_path.exists()
