"""Standalone HTML evaluation report behavior and safety checks."""

import json
import sys
from types import SimpleNamespace

import pytest

from app.config import Settings
import evals.run as eval_runner


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_html_report_escapes_all_case_and_response_content():
    attack = '<script>alert("x")</script>'
    attribute_attack = '"><img src=x onerror=alert(1)>'
    case = eval_runner.Case(
        id=attack,
        prompt=attribute_attack,
        context=attack,
        expected=attribute_attack,
        required_points=[attack],
        category=attribute_attack,
        risk=attack,
    )
    record = {
        "id": attack, "category": attribute_attack, "risk": attack,
        "answer": attack, "expected": attribute_attack,
        "source": attribute_attack, "exact_match": False, "human_review": True,
        "judge": {"faithfulness": 0.4, "completeness": 0.5,
                  "rationale": attribute_attack},
    }
    summary = {
        "mode": "replay", "cases": 1, "exact_match_rate": 0.0,
        "degraded_count": 0, "faithfulness_mean": 0.4,
        "completeness_mean": 0.5, "human_review_count": 1,
        "by_category": {attribute_attack: {"cases": 1,
                                              "exact_match_rate": 0.0,
                                              "human_review_count": 1}},
        "by_risk": {attack: {"cases": 1, "exact_match_rate": 0.0,
                             "human_review_count": 1}},
    }

    report = eval_runner.render_html_report(summary, [record], [case])

    assert report.startswith("<!doctype html>")
    assert "Content-Security-Policy" in report
    assert "default-src 'none'" in report
    assert "Offline replay" in report
    assert "No model or judge calls were made" in report
    assert "Supplied answer" in report and "Supplied judge score" in report
    assert "40.0%" not in report  # Judge scores use the 0–1 scale.
    assert "0.40" in report and "0.50" in report
    assert attack not in report
    assert attribute_attack not in report
    assert "&lt;script&gt;" in report
    assert "&lt;img src=x onerror=alert(1)&gt;" in report
    assert "<script" not in report and "<img src=x" not in report


def test_replay_cli_writes_html_with_examples_and_preserves_other_outputs(
        tmp_path, monkeypatch, capsys):
    cases_path = tmp_path / "cases.jsonl"
    replay_path = tmp_path / "replay.jsonl"
    output_path = tmp_path / "reports" / "results.jsonl"
    markdown_path = tmp_path / "reports" / "evaluation.md"
    summary_path = tmp_path / "reports" / "summary.json"
    html_path = tmp_path / "reports" / "evaluation.html"
    write_jsonl(cases_path, [
        {"id": "one", "prompt": "What is the window?", "context": "Return in 30 days.",
         "expected": "30 days", "required_points": ["30 days"],
         "category": "policy", "risk": "normal"},
        {"id": "two", "prompt": "What is missing?", "context": "No warranty stated.",
         "expected": "I do not know", "category": "policy", "risk": "high"},
    ])
    write_jsonl(replay_path, [
        {"id": "one", "answer": "14 days"},
        {"id": "two", "answer": "I do not know"},
    ])
    monkeypatch.setattr(sys, "argv", [
        "evals.run", "--cases", str(cases_path), "--replay", str(replay_path),
        "--output", str(output_path), "--report", str(markdown_path),
        "--summary", str(summary_path), "--html-report", str(html_path),
        "--sample-rate", "0",
    ])

    eval_runner.main()

    summary = json.loads(capsys.readouterr().out)
    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    report = html_path.read_text(encoding="utf-8")
    assert summary["mode"] == "replay" and summary["exact_match_rate"] == 0.5
    assert json.loads(summary_path.read_text()) == summary
    assert len(records) == 2 and all(record["source"] == "replay" for record in records)
    assert "# Evaluation Report (Offline Replay)" in markdown_path.read_text()
    for text in ("What is the window?", "Return in 30 days.", "30 days", "14 days",
                 "I do not know", "Required points", "Not scored", "50.0%", "replay"):
        assert text in report
    assert "Generated answer" not in report


@pytest.mark.asyncio
@pytest.mark.parametrize("collision", ["cases", "replay", "output", "markdown", "summary"])
async def test_html_report_cannot_overwrite_any_input_or_output(tmp_path, collision):
    cases_path = tmp_path / "cases.jsonl"
    replay_path = tmp_path / "replay.jsonl"
    output_path = tmp_path / "results.jsonl"
    markdown_path = tmp_path / "evaluation.md"
    summary_path = tmp_path / "summary.json"
    paths = {"cases": cases_path, "replay": replay_path, "output": output_path,
             "markdown": markdown_path, "summary": summary_path}
    write_jsonl(cases_path, [{"id": "one", "prompt": "Question", "expected": "Answer"}])
    write_jsonl(replay_path, [{"id": "one", "answer": "Answer"}])

    with pytest.raises(ValueError, match="different paths"):
        await eval_runner.run(cases_path, output_path, 0, 100,
                              report_path=markdown_path, summary_path=summary_path,
                              replay_path=replay_path, html_report_path=paths[collision])

    assert json.loads(cases_path.read_text())["id"] == "one"
    assert json.loads(replay_path.read_text())["id"] == "one"
    assert not output_path.exists()


def test_html_live_report_labels_unscored_and_fallback():
    case = eval_runner.Case(id="outage", prompt="Q", expected="A")
    record = {"id": "outage", "category": "general", "risk": "normal",
              "answer": "Temporary error", "expected": "A", "source": "fallback",
              "exact_match": False, "human_review": True, "judge": None}
    group = {"general": {"cases": 1, "exact_match_rate": 0, "human_review_count": 1}}
    summary = {"mode": "live", "cases": 1, "exact_match_rate": 0,
               "degraded_count": 1, "faithfulness_mean": None,
               "completeness_mean": None, "human_review_count": 1,
               "by_category": group, "by_risk": group}

    report = eval_runner.render_html_report(summary, [record], [case])

    assert "Live evaluation" in report
    assert "Generated answer" in report and "Temporary error" in report
    assert "fallback" in report and "Not scored" in report
    assert "Degraded responses" in report


@pytest.mark.asyncio
async def test_all_fallback_live_report_does_not_claim_model_quality(tmp_path, monkeypatch):
    cases_path = tmp_path / "cases.jsonl"
    results_path = tmp_path / "results.jsonl"
    markdown_path = tmp_path / "evaluation.md"
    html_path = tmp_path / "evaluation.html"
    write_jsonl(cases_path, [{
        "id": "outage", "prompt": "Is the service available?",
        "expected": "Service temporarily unavailable.",
    }])
    settings = Settings(
        provider="openai", model="test-model", api_key="unused", app_api_keys=(),
        redis_url="redis://unused", rate_limit_per_minute=1,
        llm_timeout_seconds=1, request_deadline_seconds=2, llm_max_retries=0,
        cache_ttl_seconds=0,
    )
    monkeypatch.setattr(eval_runner.Settings, "from_env", lambda *, require_app_api_keys: settings)
    monkeypatch.setattr(eval_runner, "HTTPProvider", lambda *args: object())

    class FallbackService:
        async def generate(self, request, client_id, check_rate):
            return SimpleNamespace(
                answer="Service temporarily unavailable.", source="fallback", request_id="r1",
            )

    async def no_judge(*args):
        raise AssertionError("Fallback answers must not be sent to an LLM judge")

    monkeypatch.setattr(eval_runner, "GenerationService", lambda *args: FallbackService())
    monkeypatch.setattr(eval_runner, "judge", no_judge)

    summary = await eval_runner.run(
        cases_path, results_path, 0, 1,
        report_path=markdown_path, html_report_path=html_path,
    )

    record = json.loads(results_path.read_text().strip())
    markdown = markdown_path.read_text()
    html = html_path.read_text()
    assert summary["mode"] == "live"
    assert summary["exact_match_rate"] == 1.0  # The fallback happens to match.
    assert summary["degraded_count"] == 1
    assert record["source"] == "fallback" and record["judge"] is None
    assert "Model quality not measured" in markdown and "Model quality not measured" in html
    assert "The case returned a fallback response" in markdown
    assert "The case returned a fallback response" in html
    assert "Exact match rate measures returned responses, not the configured model" in markdown
    assert "No LLM-as-a-Judge scores were produced" in html
    assert "| Model-generated responses | 0/1 |" in markdown
    assert "| Exact match rate | 100.0% |" in markdown
    assert "Model-generated responses</dt><dd>0 / 1" in html
