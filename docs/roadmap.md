# Roadmap

## Current

- Route library parser.
- Metadata guardrail router.
- Linear hidden-vector router.
- Thinker / Worker / Verifier loop.
- Workflow DAG executor.
- Provider boundary with mock and OpenAI-compatible adapters.
- Redacted JSONL traces.
- SVF experiment utility.
- Route eval harness.

## Next

- Add optional local hidden-state extractors behind extras.
- Add SGLang/vLLM adapter examples for LoRA route targets.
- Add CMA-ES training scaffolding for small route heads.
- Add OpenTelemetry trace export.
- Add richer eval reports with per-route confusion matrices.
- Add typed workflow plan import/export.

## Non-Goals

- Shipping proprietary model weights.
- Claiming reproduction of Sakana Fugu or TRINITY benchmark scores without the
  original private setup.
- Hiding provider selection behind opaque prompts without diagnostics.

