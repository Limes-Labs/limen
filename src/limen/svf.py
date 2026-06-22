from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class SVFDecomposition:
    """Singular-value fine-tuning helper for small open experiments.

    The decomposition freezes left/right singular vectors and adapts only
    per-singular-value offsets. Reconstruction applies the same sum-preserving
    normalization described in the Fugu/TRINITY analyses.
    """

    u: npt.NDArray[np.float32]
    singular_values: npt.NDArray[np.float32]
    vh: npt.NDArray[np.float32]
    source_shape: tuple[int, ...]

    @classmethod
    def decompose(cls, matrix: npt.ArrayLike) -> SVFDecomposition:
        value = np.asarray(matrix, dtype=np.float32)
        if value.ndim != 2:
            raise ValueError(f"SVF expects a rank-2 matrix, got shape={value.shape}")
        u, singular_values, vh = np.linalg.svd(value, full_matrices=False)
        return cls(
            u=np.asarray(u, dtype=np.float32),
            singular_values=np.asarray(singular_values, dtype=np.float32),
            vh=np.asarray(vh, dtype=np.float32),
            source_shape=value.shape,
        )

    def reconstruct(self, scale_offsets: npt.ArrayLike) -> npt.NDArray[np.float32]:
        offsets = np.asarray(scale_offsets, dtype=np.float32)
        if offsets.shape != self.singular_values.shape:
            raise ValueError(
                f"scale offset shape must be {self.singular_values.shape}, got {offsets.shape}"
            )

        scaled = self.singular_values * (1.0 + offsets)
        denominator = float(np.sum(scaled))
        if denominator == 0.0:
            raise ValueError("scaled singular values sum to zero")
        normalization = float(np.sum(self.singular_values)) / denominator
        reconstructed = (self.u * scaled.reshape(1, -1)) @ self.vh
        return np.asarray(reconstructed * normalization, dtype=np.float32)


def svf_parameter_count(matrices: list[npt.ArrayLike]) -> int:
    """Return the number of trainable SVF offsets for a list of matrices."""

    total = 0
    for matrix in matrices:
        value = np.asarray(matrix)
        if value.ndim != 2:
            raise ValueError(f"expected rank-2 matrix, got shape={value.shape}")
        total += min(value.shape)
    return total
