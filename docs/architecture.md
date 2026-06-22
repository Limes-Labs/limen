# Architecture

Limen separates orchestration into small, testable components.

## Route Library

Route libraries are markdown files with front matter and sections. They can
describe remote models, local workers, LoRA adapters, or workflow lanes.

The entry route, usually `L0`, handles ordinary chat and ambiguous requests.
Specialist routes are selected only when signals are clear enough.

## Routers

`MetadataRouter` is the transparent baseline. It scores strong, positive, and
negative signals, then returns a route decision with diagnostics.

`LinearHeadRouter` is the learned-router shape. It accepts an externally
computed hidden vector and applies a bias-free matrix multiplication. The first
rows select the worker. Optional trailing rows select a role.

`TranscriptVectorRouter` is the SLM-style bridge. It formats messages as raw
`role: content` lines, calls an injected hidden-vector extractor, then delegates
to `LinearHeadRouter`. The extractor is deliberately external so applications
can use a local small language model, a cached embedding service, or a test
double without binding Limen to a specific inference stack. Diagnostics include
the transcript format and a transcript hash, not the raw transcript text.

`LinearHeadArtifact` stores the learned routing head as a compressed `.npz`
file with a versioned JSON manifest. The artifact records `num_agents`,
`role_names`, and user metadata next to the matrix, so trained heads can be
reviewed, moved between services, and loaded without hidden process state.

`ScriptedRouter` exists for tests and demos.

## Evaluation

The route eval harness reads JSONL fixtures with `id`, `prompt`, and
`expected_route_id`. Reports include:

- total examples;
- correct examples;
- scalar accuracy;
- misses with expected and actual routes;
- confusion matrix grouped by expected route;
- per-route total, correct count, and accuracy.

## Roles

The runtime supports three canonical roles:

- `Thinker`: plan, critique, decompose, or suggest the next role.
- `Worker`: perform the concrete step.
- `Verifier`: check the current answer and start with `ACCEPT` or `REVISE`.

Unknown verifier text is treated as revision for loop control.

## Provider Boundary

The provider pool maps `agent_id` to `ProviderSpec`, then dispatches through an
adapter. The default repository includes:

- `MockProvider` for deterministic local tests.
- `OpenAICompatibleProvider` for chat-completions-compatible endpoints.

Applications can add their own adapters for Anthropic, Gemini, local inference
servers, SGLang, vLLM, or internal services.

## Tracing

Trace events are schema-versioned JSON objects. Sensitive keys are redacted and
large/private content should be represented by hashes. The JSONL sink appends
events for local audit and eval debugging.

## Workflows

`WorkflowPlan` is a sequential DAG. Each step declares:

- `id`
- `agent_id`
- `prompt`
- `access`: earlier step ids visible to this step

Forward references fail validation. This preserves explicit information flow and
prevents accidental all-to-all context sharing.

## SVF

`SVFDecomposition` supports singular-value adaptation experiments. It freezes
singular vectors, applies per-singular-value offsets, and normalizes the sum of
singular values during reconstruction.
