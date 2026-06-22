# Provider Accounting

Limen treats model choice and test-time compute as auditable runtime state. A
router may decide which model to call, but the application should still be able
to answer:

- which provider and model were used;
- how many provider calls were made;
- how many input and output tokens were reported;
- what the estimated cost was under declared pricing;
- whether a run stopped because it hit a cost budget.

## Provider Specs

`ProviderSpec` records the concrete provider and model behind an `agent_id`.
When prices are known, declare them per million tokens:

```python
from limen.providers import ProviderSpec

spec = ProviderSpec(
    agent_id=0,
    provider="openai_compatible",
    model="example-model",
    input_cost_per_million_tokens=1.00,
    output_cost_per_million_tokens=2.00,
)
```

If pricing is omitted, Limen still records provider usage when the adapter
returns it, but `estimated_cost_usd` stays `None`.

`ProviderPool.model_manifest()` returns a public manifest of the configured
pool: `agent_id`, optional display name, provider, model, and declared prices.
`Orchestrator` writes this manifest to the `run_started` trace event.

## Usage Extraction

`OpenAICompatibleProvider` parses the standard chat-completions `usage` object:

```json
{
  "prompt_tokens": 1000,
  "completion_tokens": 500,
  "total_tokens": 1500
}
```

The parsed usage is attached to `ProviderResponse`. `ProviderPool` estimates
cost from the declared `ProviderSpec` pricing.

## Run Totals

`Orchestrator.run()` aggregates usage and estimated cost across turns:

```python
result = orchestrator.run([{"role": "user", "content": "Solve it."}])

print(result.usage)
print(result.estimated_cost_usd)
```

Trace events include per-call usage/cost on `provider_called` and run totals on
`run_completed`. Raw prompt and response text are not written to traces by
default; hashes are used instead.

## Cost Budgets

Use `max_estimated_cost_usd` to stop a run once declared provider pricing and
reported usage cross a budget:

```python
orchestrator = Orchestrator(
    router=router,
    provider_pool=provider_pool,
    max_estimated_cost_usd=0.25,
)
```

If the budget is reached after a provider call and the run has not already been
accepted, the run returns `status="cost_budget_exceeded"`.

This is intentionally explicit. A benchmark or production report that omits
model pool, provider calls, token counts, and cost is not a complete
orchestration report.
