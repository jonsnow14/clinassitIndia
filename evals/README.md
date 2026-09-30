# Eval quality track

Offline evaluation harness for ClinAssistIndia. Design: [`docs/eval-pipleine.md`](../docs/eval-pipleine.md).

This is **not** a PHC product feature. Production agents stay human-triggered. The harness may call the same functions because it is a test runner.

## E0 / PR-1 + PR-2 (this tree)

- Tagged-union gold schema (`app.eval.schema`)
- Ops-agent judge over `evals/gold/agents.jsonl` (9 Purnia ranking rows)
- Suggested-actions judge over `evals/gold/suggested.jsonl` (exact `_suggested()` order)
- Clinical / adversarial / scenario JSONL: schema + census only (no clinical/safety judge yet)
- CLI `--suite unit_ops`
- Merge gate: `pytest -m unit` (no Sarvam, no Chroma)

Census in `evals/gold/manifest.json` is the frozen v1 target (49 rows). On disk in E0: 9 agents + 12 clinical + 4 suggested + 4 scenarios + 4 adversarial. Retrieval JSONL waits for PR-3 / E1.

## Run

From `apps/api`:

```bash
python -m app.eval
python -m app.eval.run --suite unit_ops
pytest -m unit -q
```

Exit codes for the CLI: `0` all ops assertions pass, `1` a gold row failed, `2` missing/invalid gold or an unimplemented suite.

Do not import `app.eval` from `app.main`.
