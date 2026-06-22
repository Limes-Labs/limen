# Contributing To Limen

Limen is early. Contributions should improve routing quality, provider
boundaries, eval coverage, documentation, or operational safety without making
the system opaque.

## Principles

- Keep router behavior inspectable.
- Add tests for behavior changes.
- Do not commit model weights, LoRA checkpoints, private prompts, traces,
  provider credentials, or benchmark outputs containing private data.
- Keep provider credentials outside reusable library code.
- Prefer explicit typed data structures over unstructured dictionaries at module
  boundaries.
- Cite external research, code, or technical writing that materially influenced
  a change.

## Development

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
make check
```

## Pull Requests

- Describe what changed and why.
- Include tests for new behavior.
- Update docs when public APIs, route formats, traces, or provider behavior
  change.
- Keep changes scoped. Avoid unrelated rewrites.

