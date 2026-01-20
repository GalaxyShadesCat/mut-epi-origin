import numpy as np

from scripts.scores import compute_local_scores


def test_perfect_inverse_scores_negative():
    rng = np.random.default_rng(42)
    M = rng.normal(size=500)
    D = -M
    res = compute_local_scores(M, D, w=5)
    assert np.nanmean(res.total) < -0.9
    assert res.global_score < -0.9


def test_perfect_same_direction_scores_positive():
    rng = np.random.default_rng(1)
    M = rng.normal(size=500)
    D = M.copy()
    res = compute_local_scores(M, D, w=5)
    assert np.nanmean(res.total) > 0.9
    assert res.global_score > 0.9


def test_uncorrelated_noise_scores_mid():
    rng = np.random.default_rng(7)
    M = rng.normal(size=800)
    D = rng.normal(size=800)
    res = compute_local_scores(M, D, w=5)
    mean_score = np.nanmean(res.total)
    assert -0.2 < mean_score < 0.2
