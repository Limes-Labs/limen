# Security Policy

Limen is pre-production research infrastructure. Treat it as unsafe for
untrusted autonomous execution until you have reviewed and constrained your
provider adapters, tool access, and deployment environment.

## Sensitive Data Rules

- Do not commit API keys, provider tokens, private prompt traces, model weights,
  LoRA checkpoints, or private benchmark outputs.
- Use JSONL tracing in redacted mode and prefer hashes for prompt and response
  content.
- Pass credentials explicitly from your application boundary. Library code
  should not read environment variables directly.
- Review provider adapters before enabling live calls.

## Reporting Vulnerabilities

Please do not open a public issue for sensitive vulnerabilities. Until Limes
Labs publishes a dedicated security contact, send a private note to the project
maintainers through the organization owner.

