import pandas as pd

from caprl.data.splits import make_walk_forward_folds, select_half_open


def test_fold_segments_are_ordered_and_disjoint():
    dates = pd.bdate_range("2000-01-03", "2020-12-31")
    folds = make_walk_forward_folds(dates)
    assert folds
    for fold in folds:
        segments = [
            select_half_open(dates, fold.train),
            select_half_open(dates, fold.calibration),
            select_half_open(dates, fold.validation),
            select_half_open(dates, fold.test),
        ]
        assert all(len(segment) > 0 for segment in segments)
        assert all(left[-1] < right[0] for left, right in zip(segments, segments[1:]))

