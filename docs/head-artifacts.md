# Head Artifacts

`LinearHeadArtifact` is the persistence format for learned Limen routing heads.
It is meant for the small trainable surface described by Fugu/TRINITY-style
systems: an externally computed hidden vector enters a bias-free linear matrix,
then the resulting logits select an agent and, optionally, a role.

## Format

Artifacts are compressed NumPy `.npz` files with two entries:

- `weights`: a float32 matrix shaped `(num_agents + num_roles, hidden_dim)`.
- `manifest`: a JSON string with `schema_version`, `num_agents`, `role_names`,
  and arbitrary JSON-serializable `metadata`.

The current schema version is `1`. Loaders reject unknown schema versions so a
future format change cannot silently reinterpret routing weights.

## Example

```python
import numpy as np

from limen.artifacts import LinearHeadArtifact
from limen.routers import LinearHeadRouter

router = LinearHeadRouter(
    np.zeros((5, 1024), dtype=np.float32),
    num_agents=2,
    role_names=("Worker", "Thinker", "Verifier"),
)

artifact = LinearHeadArtifact.from_router(
    router,
    metadata={
        "transcript_format": "raw_role_content_v1",
        "training_run": "local-smoke-test",
    },
)
artifact.save("artifacts/router-head.npz")

loaded = LinearHeadArtifact.load("artifacts/router-head.npz")
router = loaded.to_router()
```

Inspect the artifact from the CLI:

```bash
limen artifact inspect artifacts/router-head.npz
```

The command prints the artifact type, schema version, matrix shape, agent/role
layout, and metadata. It does not print the full weight matrix.

## Metadata

Recommended metadata fields:

- `transcript_format`: input formatting used while extracting hidden states.
- `backbone`: local model or service that produced the hidden vector.
- `training_run`: run identifier or experiment slug.
- `source_fixture`: route fixture or eval set used to validate the head.
- `created_by`: owner, team, or automation that produced the artifact.

Do not store provider credentials, private prompts, or raw training traces in
artifact metadata. Use hashes or external run ids for sensitive material.
