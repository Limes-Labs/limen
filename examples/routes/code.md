---
id: code
target: coding-worker
priority: 100
---

## Description
Coding, repository edits, tests, patches, debugging, and verification.

## Routing Rules
- Use code for implementation, bug fixes, tests, diffs, and repository work.
- Do not use code for general explanations unless the user asks to modify code.

## Strong Signals
pytest, failing test, stack trace, apply_patch, refactor, bugfix

## Positive Signals
repository, code, test, lint, typecheck, diff, commit

## Negative Signals
explain, overview, summarize

## Examples
Fix a failing pytest and run verification => code
Refactor this module and add tests => code

