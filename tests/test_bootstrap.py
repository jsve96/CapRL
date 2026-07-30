import numpy as np

from caprl.evaluation.bootstrap import paired_block_bootstrap


def test_paired_bootstrap_is_reproducible_and_preserves_pairing():
    first = np.arange(100, dtype=float) + 2.0
    second = np.arange(100, dtype=float)
    first_result = paired_block_bootstrap(first, second, block_length=10, n_samples=100, seed=8)
    second_result = paired_block_bootstrap(first, second, block_length=10, n_samples=100, seed=8)
    np.testing.assert_allclose(first_result.samples, 2.0)
    np.testing.assert_array_equal(first_result.samples, second_result.samples)
    assert first_result.estimate == 2.0

