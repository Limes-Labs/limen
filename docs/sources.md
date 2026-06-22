# Sources And Attribution

Limen is original Limes Labs code, but it is intentionally built on ideas from
public work. This file tracks the sources used and how they influenced the
project.

## Sakana Fugu

- Product page: https://sakana.ai/fugu/
- GitHub repository: https://github.com/SakanaAI/fugu

Influence: the product framing of a multi-agent system exposed as one model API,
the distinction between a fast router-style mode and a workflow-style mode, and
the operational need for provider selection without single-vendor dependency.

Implementation boundary: Limen does not use Sakana APIs, copy private Fugu
logic, include Sakana checkpoints, or claim Fugu benchmark reproduction.

## TRINITY

- Paper: https://arxiv.org/abs/2512.04695
- HTML: https://arxiv.org/html/2512.04695

Influence: compact coordinator, hidden-state routing, bias-free linear head,
Thinker / Worker / Verifier roles, multi-turn verification loop, and
evolutionary optimization as a promising training direction.

Implementation boundary: Limen includes a NumPy linear-head router and SVF
utility for open experiments. It does not include TRINITY weights or reproduce
reported benchmark scores.

## Conductor

- Paper: https://arxiv.org/abs/2512.04388
- HTML: https://arxiv.org/html/2512.04388

Influence: natural-language orchestration, workflow graph planning, per-step
model assignment, and access lists that define which prior outputs each worker
can see.

Implementation boundary: Limen includes an explicit `WorkflowPlan` DAG executor.
It does not train a 7B Conductor model.

## nshkrdotcom/trinity_coordinator

- Repository: https://github.com/nshkrdotcom/trinity_coordinator

Influence: clean separation between hidden-state extraction, routing head,
role injection, provider boundary, verifier parsing, trace events, artifact
metadata, and runtime gates. The project also demonstrates how to keep live
provider calls behind explicit boundaries and how to validate route decisions
with fixtures.

Implementation boundary: Limen is Python-first and does not copy Elixir source.
The architecture is adapted into original modules with different APIs and a
lighter dependency profile.

## MindLab-Research/Mixture-of-LoRA-Harness

- Repository: https://github.com/MindLab-Research/Mixture-of-LoRA-Harness

Influence: markdown route libraries, entry route fallback, specialist routing
only on clear signals, deterministic metadata guardrails, adapter names as
server-visible targets, and explicit KV/history policy documentation.

Implementation boundary: Limen implements its own route parser and metadata
router. It does not include LoRA weights, private datasets, or SGLang patches.

## Di Zhang Technical Inspection

- Article: https://di-zhang-llm.github.io/blog/how-fugu-is-implemented-a-technical-inspect/

Influence: the concise explanation that Fugu is a policy over models, not a new
monolithic model; the breakdown of a small trainable surface, SVF offsets,
linear selection head, raw `role: content` transcript routing, verifier loop,
and workflow DAG line.

Implementation boundary: Limen treats this article as technical commentary and
attributes it. It does not rely on unpublished artifacts.
