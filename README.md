# Limen

Open-source orchestration primitives for routing tasks across models, roles,
workflows, and LoRA adapters.

Limen is a Limes Labs project for building Fugu-style orchestration systems in
the open. It treats model choice as a policy layer: a small router decides which
worker, role, adapter, or workflow should handle a request, while the expensive
generation remains inside frozen external models or local adapters.

The name means "threshold": Limen sits at the boundary between a user request
and the model or workflow best suited to cross it.

## What This Is

- A transparent route-library format for model and LoRA routing.
- A deterministic metadata router with guardrails and diagnostics.
- A bias-free linear-head router over hidden vectors for TRINITY-style
  experiments.
- A Thinker / Worker / Verifier orchestration loop.
- A Conductor-style workflow DAG executor with explicit access lists.
- Provider adapters, including deterministic mock providers and an
  OpenAI-compatible chat-completions adapter.
- Redacted JSONL tracing with content hashes instead of raw private prompts.
- A small eval harness for route fixture accuracy.
- SVF utilities for singular-value adaptation experiments.

## What This Is Not

- It is not Sakana Fugu, and it is not affiliated with Sakana AI.
- It does not include proprietary model weights, Sakana checkpoints, private
  provider pools, API keys, private traces, or benchmark claims.
- It does not copy closed-source Fugu internals. It implements public ideas from
  cited papers, technical writing, and open-source repositories in original
  Limes Labs code.

## Quickstart

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
limen --library-dir examples/routes "Fix a failing pytest and run verification."
```

Expected shape:

```json
{
  "route_id": "code",
  "target": "coding-worker",
  "reason": "specialist_signal"
}
```

Run checks:

```bash
make test
make lint
make typecheck
```

## Architecture

```text
messages
  -> router
     -> metadata route, linear head route, or scripted route
  -> role injector
     -> Thinker, Worker, Verifier
  -> provider pool
     -> mock, OpenAI-compatible, or downstream adapter
  -> verifier parser
     -> ACCEPT / REVISE bounded loop
  -> trace sink
     -> JSONL events, hashes, redaction
```

For larger jobs, Limen also supports workflow plans:

```text
user request
  -> WorkflowPlan[step_id, agent_id, prompt, access_list]
  -> sequential DAG execution
  -> each step sees only outputs named in its access list
```

## Design Principles

- Route decisions must be inspectable.
- Ambiguous specialist matches fall back to the entry route.
- Verifier output must explicitly start with `ACCEPT` or `REVISE`.
- Provider credentials are passed at the application boundary, not read from
  process environment inside reusable library code.
- Traces should be useful without leaking prompt content or secrets.
- Model weights and LoRA checkpoints stay outside the repository.

## Sources And Attribution

Limen is inspired by, and cites, these sources:

- Sakana AI, "Sakana Fugu": https://sakana.ai/fugu/
- SakanaAI/fugu: https://github.com/SakanaAI/fugu
- Jinglue Xu et al., "TRINITY: An Evolved LLM Coordinator":
  https://arxiv.org/abs/2512.04695
- Stefan Nielsen et al., "Learning to Orchestrate Agents in Natural Language
  with the Conductor": https://arxiv.org/abs/2512.04388
- nshkrdotcom/trinity_coordinator:
  https://github.com/nshkrdotcom/trinity_coordinator
- MindLab-Research/Mixture-of-LoRA-Harness:
  https://github.com/MindLab-Research/Mixture-of-LoRA-Harness
- Di Zhang, "How Fugu Is Implemented - A Technical Inspection":
  https://di-zhang-llm.github.io/blog/how-fugu-is-implemented-a-technical-inspect/

See [docs/sources.md](docs/sources.md) for the detailed attribution map.

## License

Apache-2.0. See [LICENSE](LICENSE).
