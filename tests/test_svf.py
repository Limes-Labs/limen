import numpy as np

from limen.svf import SVFDecomposition


def test_svf_zero_offsets_reconstruct_original_matrix() -> None:
    matrix = np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 0.5]], dtype=np.float32)

    decomp = SVFDecomposition.decompose(matrix)
    reconstructed = decomp.reconstruct(np.zeros(decomp.singular_values.shape, dtype=np.float32))

    assert np.allclose(reconstructed, matrix, atol=1e-5)


def test_svf_nonzero_offsets_preserve_singular_value_sum() -> None:
    matrix = np.array([[2.0, 0.0], [0.0, 1.0], [0.5, 0.25]], dtype=np.float32)

    decomp = SVFDecomposition.decompose(matrix)
    adapted = decomp.reconstruct(np.array([0.5, -0.2], dtype=np.float32))

    original_sum = float(np.sum(decomp.singular_values))
    adapted_sum = float(np.sum(np.linalg.svd(adapted, full_matrices=False, compute_uv=False)))
    assert adapted.shape == matrix.shape
    np.testing.assert_allclose(adapted_sum, original_sum, rtol=1e-5)
