---
id: lora
target: lora-router
adapter_name: lora-router
priority: 80
---

## Description
LoRA adapter routing, adapter metadata, SGLang/vLLM serving, and specialist
adapter policy.

## Routing Rules
- Use lora when the task is about choosing, loading, or evaluating adapters.
- Keep raw checkpoint paths out of public traces.

## Strong Signals
LoRA, adapter, SGLang, vLLM, checkpoint

## Positive Signals
route, specialist, local model, serving

## Negative Signals
frontend, css, pytest

## Examples
Route this request between LoRA adapters => lora

