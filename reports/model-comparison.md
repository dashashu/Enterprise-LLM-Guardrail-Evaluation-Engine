# Model Test Summary

Run date: 2026-10-07. All runs used the same six cases from `evals/sample_cases.jsonl`. Open the [HTML report index](index.html) to browse the reports.

| Run | Model responses | Judged cases | Exact match | Mean faithfulness | Mean completeness | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Luna comparison run | 6/6 | 6/6 | 66.7% | 1.00 | 0.98 | Completed |
| Sol comparison run | 0/6 | 0/6 | Not measured | Not scored | Not scored | Blocked by `credit_balance_exhausted` |
| Astra comparison run | — | — | — | — | — | Not started after the billing error |
| Earlier Luna run | 6/6 | 6/6 | 83.3% | 1.00 | 1.00 | Completed |

The Luna comparison run had two exact-match differences: `30` versus `30 days`, and `I do not know.` versus `I do not know`. The judge scored both as faithful, with completeness of 1.00 and 0.90 respectively. This illustrates why the exact-match and judge scores should be read together.

The Sol run received HTTP `429` with `credit_balance_exhausted` for all six cases. Its saved responses are deterministic fallbacks, so its recorded zero exact-match rate does not measure Sol's answer quality. Astra was not called after that billing error. On successful runs, the configured model also acted as the judge; judge scores are not an independent cross-model assessment. Six cases and two Luna runs are too few to establish a production ranking.

## Report files

| Run | HTML report | Markdown report | JSON summary | Detailed JSONL |
| --- | --- | --- | --- | --- |
| Luna comparison | [Open HTML](gpt-6-luna-evaluation.html) | [Open Markdown](gpt-6-luna-evaluation.md) | [Open JSON](gpt-6-luna-summary.json) | [Open JSONL](gpt-6-luna-results.jsonl) |
| Sol comparison | [Open HTML](gpt-6.1-sol-evaluation.html) | [Open Markdown](gpt-6.1-sol-evaluation.md) | [Open JSON](gpt-6.1-sol-summary.json) | [Open JSONL](gpt-6.1-sol-results.jsonl) |
| Earlier Luna | [Open HTML](live-evaluation.html) | [Open Markdown](live-evaluation.md) | [Open JSON](live-summary.json) | [Open JSONL](live-results.jsonl) |

Restore API credits or configure a funded project key before completing the Sol and Astra evaluations. The JSONL files contain generated answers and should be handled as evaluation data.
