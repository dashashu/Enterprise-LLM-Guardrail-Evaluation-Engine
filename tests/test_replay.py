"""Offline evaluation of supplied answers, with no provider dependency."""

import json
import sys

import pytest

import evals.run as eval_runner


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


@pytest.mark.asyncio
async def test_replay_scores_supplied_answers_without_provider_calls(tmp_path, monkeypatch):
    cases_path = tmp_path / "cases.jsonl"
    replay_path = tmp_path / "answers.jsonl"
    output_path = tmp_path / "reports" / "results.jsonl"
    report_path = tmp_path / "reports" / "evaluation.md"
    summary_path = tmp_path / "reports" / "summary.json"
    write_jsonl(cases_path, [
        {"id": "matching", "prompt": "Question 1", "expected": "  An   answer  ", "category": "facts"},
        {"id": "wrong", "prompt": "Question 2", "expected": "Correct answer", "category": "facts"},
        {"id": "high", "prompt": "Question 3", "expected": "Caution", "risk": "high"},
        {"id": "unjudged", "prompt": "Question 4", "expected": "Private expected"},
    ])
    write_jsonl(replay_path, [
        {"id": "unjudged", "answer": "Private expected"},
        {"id": "high", "answer": "Caution", "judge": {
            "faithfulness": 1, "completeness": 1, "rationale": "Supplied score"}},
        {"id": "wrong", "answer": "Private wrong answer", "judge": {
            "faithfulness": 1, "completeness": 1, "rationale": "Supplied score"}},
        {"id": "matching", "answer": "an answer", "judge": {
            "faithfulness": 1, "completeness": 1, "rationale": "Supplied score"}},
    ])

    def unexpected(*args, **kwargs):
        raise AssertionError("Offline replay must not configure or call a provider")

    monkeypatch.setattr(eval_runner.Settings, "from_env", unexpected)
    monkeypatch.setattr(eval_runner.httpx, "AsyncClient", unexpected)
    summary = await eval_runner.run(cases_path, output_path, 0.0, 100,
                                    report_path, summary_path, replay_path)

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [record["id"] for record in records] == ["matching", "wrong", "high", "unjudged"]
    assert [record["exact_match"] for record in records] == [True, False, True, True]
    assert [record["human_review"] for record in records] == [False, True, True, False]
    assert all(record["source"] == "replay" for record in records)
    assert all(record["model"] is None and record["provider"] is None for record in records)
    assert records[-1]["judge"] is None
    assert summary["mode"] == "replay"
    assert summary["degraded_count"] == 0
    assert summary["exact_match_rate"] == 0.75
    assert summary["human_review_count"] == 2
    assert summary["faithfulness_mean"] == 1
    assert json.loads(summary_path.read_text()) == summary
    report = report_path.read_text()
    assert report.startswith("# Evaluation Report (Offline Replay)\n")
    assert "No model or judge calls were made" in report
    assert "| Degraded responses | N/A (offline replay) |" in report
    assert "Private wrong answer" not in report
    assert "Private expected" not in report
    assert "Supplied score" not in report


@pytest.mark.asyncio
@pytest.mark.parametrize("replay_records,error", [
    ([], "missing"),
    ([{"id": "one", "answer": "yes"}, {"id": "extra", "answer": "no"}], "extra"),
    ([{"id": "one", "answer": "yes"}, {"id": "one", "answer": "again"}], "Duplicate replay id"),
    ([{"id": "one", "answer": ""}], "Invalid replay answer"),
    ([{"id": "one", "answer": "yes", "judge": {"faithfulness": 2,
       "completeness": 1, "rationale": "invalid"}}], "Invalid replay answer"),
    ([{"id": "one", "answer": "yes", "judge": None}], "Invalid replay answer"),
    ([{"id": "one", "answer": "yes", "source": "model"}], "Invalid replay answer"),
])
async def test_replay_rejects_incomplete_or_malformed_inputs_before_writing(
        tmp_path, replay_records, error):
    cases_path = tmp_path / "cases.jsonl"
    replay_path = tmp_path / "answers.jsonl"
    output_path = tmp_path / "results.jsonl"
    summary_path = tmp_path / "summary.json"
    write_jsonl(cases_path, [{"id": "one", "prompt": "Question", "expected": "yes"}])
    write_jsonl(replay_path, replay_records)

    with pytest.raises(ValueError, match=error):
        await eval_runner.run(cases_path, output_path, 0, 100,
                              summary_path=summary_path, replay_path=replay_path)
    assert not output_path.exists()
    assert not summary_path.exists()


@pytest.mark.asyncio
async def test_replay_input_cannot_be_overwritten_by_report_destinations(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    replay_path = tmp_path / "answers.jsonl"
    write_jsonl(cases_path, [{"id": "one", "prompt": "Question", "expected": "yes"}])
    write_jsonl(replay_path, [{"id": "one", "answer": "yes"}])
    for destination in ("output", "report", "summary"):
        paths = {"output_path": tmp_path / "results.jsonl",
                 "report_path": tmp_path / "report.md",
                 "summary_path": tmp_path / "summary.json"}
        paths[f"{destination}_path"] = replay_path
        with pytest.raises(ValueError, match="different paths"):
            await eval_runner.run(cases_path, paths["output_path"], 0, 100,
                                  paths["report_path"], paths["summary_path"], replay_path)


def test_replay_cli_generates_report_without_credentials(tmp_path, monkeypatch, capsys):
    cases_path = tmp_path / "cases.jsonl"
    replay_path = tmp_path / "answers.jsonl"
    output_path = tmp_path / "results.jsonl"
    report_path = tmp_path / "evaluation.md"
    summary_path = tmp_path / "summary.json"
    write_jsonl(cases_path, [{"id": "one", "prompt": "Question", "expected": "Answer"}])
    write_jsonl(replay_path, [{"id": "one", "answer": "Answer"}])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setattr(sys, "argv", ["evals.run", "--cases", str(cases_path),
                                     "--replay", str(replay_path), "--output", str(output_path),
                                     "--report", str(report_path), "--summary", str(summary_path)])

    eval_runner.main()

    assert json.loads(capsys.readouterr().out)["mode"] == "replay"
    assert report_path.exists() and summary_path.exists() and output_path.exists()


@pytest.mark.asyncio
async def test_replay_flags_low_supplied_score_and_spot_sample(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    replay_path = tmp_path / "answers.jsonl"
    output_path = tmp_path / "results.jsonl"
    write_jsonl(cases_path, [
        {"id": "low-score", "prompt": "First", "expected": "yes"},
        {"id": "sampled", "prompt": "Second", "expected": "yes"},
    ])
    write_jsonl(replay_path, [
        {"id": "low-score", "answer": "yes", "judge": {
            "faithfulness": 0.6, "completeness": 1, "rationale": "Supplied score"}},
        {"id": "sampled", "answer": "yes", "judge": {
            "faithfulness": 1, "completeness": 1, "rationale": "Supplied score"}},
    ])

    await eval_runner.run(cases_path, output_path, 0, 100, replay_path=replay_path)
    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [record["human_review"] for record in records] == [True, False]

    await eval_runner.run(cases_path, output_path, 1, 100, replay_path=replay_path)
    sampled = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert all(record["human_review"] for record in sampled)
