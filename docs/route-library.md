# Route Library Format

Limen route libraries use one markdown file per route.

```markdown
---
id: code
target: coding-worker
priority: 100
---

## Description
Coding, tests, patches, and repository verification.

## Strong Signals
pytest, failing test, apply_patch

## Positive Signals
repository, code, verify

## Negative Signals
explain, overview

## Examples
Fix a failing pytest => code
```

## Front Matter

- `id`: stable route id returned by routers.
- `target`: downstream model, provider slot, adapter name, or workflow lane.
- `adapter_name`: accepted as an alias for `target` for LoRA libraries.
- `priority`: tie-breaker for deterministic routing.
- `source_path`: optional local checkpoint path. Keep real weights out of git.

Route ids must be unique across the library. Loading fails if two files declare
the same `id`, because silent overwrites would make router evals misleading.

## Sections

- `Description`: shown in router prompts and docs.
- `Routing Rules`: bullet list for language-model routers.
- `Strong Signals`: high-confidence comma-separated phrases.
- `Positive Signals`: weaker comma-separated phrases.
- `Negative Signals`: phrases that suppress the route.
- `Examples`: prompt-to-route examples.
- `Datasets`: optional local eval references.

## Policy

The metadata router is conservative:

- clear specialist signal: route to specialist;
- conflicting specialist signals: route to entry;
- ordinary chat or ambiguous work: route to entry;
- invalid model-produced route: route to entry unless metadata has a stronger
  specialist match.
