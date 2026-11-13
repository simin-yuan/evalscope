"""Bootstrap interval estimates for continuous metrics (#2011, second slice)."""

import time

import pytest

from evalscope.api.metric import SampleScore, Score
from evalscope.metrics.aggregators import Mean
from evalscope.metrics.aggregators.ci import (
    BOOTSTRAP_SEED,
    DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_ci_metadata,
    bootstrap_interval,
)


def test_bootstrap_is_reproducible_with_the_fixed_seed() -> None:
    values = [0.1, 0.4, 0.35, 0.8, 0.55]

    assert bootstrap_interval(values) == bootstrap_interval(values)


def test_bootstrap_brackets_the_mean() -> None:
    values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    low, high = bootstrap_interval(values)

    assert low < 0.5 < high
    assert high - low > 0.0


def test_bootstrap_of_a_constant_is_degenerate() -> None:
    low, high = bootstrap_interval([0.7] * 10)

    assert low == pytest.approx(0.7)
    assert high == pytest.approx(0.7)


def test_bootstrap_edges() -> None:
    assert bootstrap_interval([]) == (0.0, 0.0)
    assert bootstrap_interval([0.42]) == (0.42, 0.42)


def test_metadata_records_the_method_and_sample_count() -> None:
    metadata = bootstrap_ci_metadata([0.2, 0.4, 0.6, 0.8])

    assert metadata['ci_method'] == 'bootstrap'
    assert metadata['ci_samples'] == DEFAULT_BOOTSTRAP_SAMPLES
    assert metadata['ci_level'] == pytest.approx(0.95)
    assert metadata['ci_low'] < metadata['ci_high']
    assert metadata['low_confidence'] is True  # 4 samples


def test_bootstrap_cost_stays_within_the_documented_budget() -> None:
    """O(resamples * n) with numpy: 5000 values must stay well under a second."""
    values = [i / 5000 for i in range(5000)]

    start = time.perf_counter()
    bootstrap_interval(values)
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0


def test_mean_attaches_bootstrap_to_continuous_metrics() -> None:
    scores = [SampleScore(sample_id=i, score=Score(value={'bleu': value}))
              for i, value in enumerate([0.2, 0.4, 0.6, 0.8])]
    result = Mean()(scores)

    assert result[0].num_correct is None
    assert result[0].metadata is not None
    assert result[0].metadata['ci_method'] == 'bootstrap'
    assert result[0].metadata['ci_low'] < result[0].score < result[0].metadata['ci_high']


def test_mean_skips_the_interval_for_non_finite_values() -> None:
    scores = [SampleScore(sample_id=i, score=Score(value={'bleu': value}))
              for i, value in enumerate([0.2, float('nan')])]
    result = Mean()(scores)

    assert result[0].metadata is None


def test_seed_is_part_of_the_contract() -> None:
    values = [((i * 37) % 101) / 100 for i in range(50)]
    first = bootstrap_interval(values, seed=BOOTSTRAP_SEED)
    second = bootstrap_interval(values, seed=BOOTSTRAP_SEED + 1)

    assert first != second
