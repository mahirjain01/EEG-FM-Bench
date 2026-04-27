"""
Chance-corrected balanced accuracy for UniformEvalBench ranking.

Formula
-------
    cc(ba, K) = (ba - 1/K) / (1 - 1/K)

where K is the number of classes for dataset d.  The result is NOT clipped —
values below 0 are valid and indicate below-chance performance; the aggregate
A_m and B_m scores clip to [0, 1] only at the final mean step.

Why chance-correct?
-------------------
A raw balanced accuracy of 0.55 on a binary task (chance = 0.50) represents
half the useful signal of 0.55 on a 5-class task (chance = 0.20).  Without
correction, datasets with more classes dominate the cross-dataset average.
Prior work including EEG-FM-Bench reports raw balanced accuracy, which makes
cross-dataset aggregation numerically incomparable.
"""

# Number of output classes per dataset.
# Used to compute per-dataset chance level: c_d = 1 / K_d.
DATASET_N_CLASSES: dict[str, int] = {
    "bcic_2a":            4,   # left / right / foot / tongue
    "seed_iv":            4,   # neutral / sad / fear / happy
    "hmc":                5,   # W / N1 / N2 / N3 / REM
    "tuab":               2,   # normal / abnormal
    "tuev":               6,   # SPSW / GPED / PLED / EYEM / ARTF / BCKG
    "adftd":              3,   # AD / FTD / CN
    "epilepsy_mimickers": 2,   # Epileptic / Mimicker
    "motor_mv_img":       4,   # 4-class movement imagery
    "siena_scalp":        2,   # seizure / non-seizure
    "workload":           2,   # high / low cognitive workload
}


def chance_correct(ba: float, n_classes: int) -> float:
    """Return chance-corrected balanced accuracy (unclipped).

    Parameters
    ----------
    ba:        raw balanced accuracy in [0, 1]
    n_classes: number of classes for this dataset

    Returns
    -------
    Chance-corrected score. Positive = above chance; negative = below chance.
    """
    chance = 1.0 / n_classes
    denom = 1.0 - chance
    return (ba - chance) / denom


def chance_level(dataset: str) -> float:
    """Return chance-level balanced accuracy for a dataset by name."""
    n = DATASET_N_CLASSES.get(dataset)
    if n is None:
        raise KeyError(f"Unknown dataset '{dataset}'. Add it to DATASET_N_CLASSES.")
    return 1.0 / n
