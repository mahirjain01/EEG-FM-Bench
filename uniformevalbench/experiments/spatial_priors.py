#!/usr/bin/env python3
"""
Spatial priors for NAS (Neurophysiological Alignment Score) — Axis C.

get_prior(dataset_name, ch_names, sigma=20.0) → np.ndarray [n_channels]

The prior p_t over channels for task t is a Gaussian mixture centered on
task-relevant electrodes (literature-derived, see §9.1 of the draft).
Electrode positions are approximate 2D projections of the standard 10-10
system onto a unit disk (azimuthal equidistant from Cz).

Datasets with no strong spatial prior (adftd, siena_scalp) use a broad
Gaussian centered on Cz (sigma=60 mm), which approximates a uniform
distribution across the scalp when the channel density is high.
"""

import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Standard 10-10 electrode positions in millimetres on a flattened scalp disk.
# Coordinates: x = left(-) / right(+), y = anterior(+) / posterior(-).
# Radius of the disk ≈ 90 mm (nasion-to-inion / 2).
# Values from standard azimuthal equidistant projection of the 10-10 system.
# ---------------------------------------------------------------------------
_POS: dict[str, tuple[float, float]] = {
    # Frontal pole
    "FP1": (-27.0,  85.0), "FPZ": (0.0,   87.0), "FP2": (27.0,  85.0),
    # Anterior frontal
    "AF9": (-63.0,  67.0), "AF7": (-51.0,  71.0), "AF5": (-39.0,  74.0),
    "AF3": (-26.0,  74.0), "AF1": (-13.0,  76.0), "AFZ": (0.0,   77.0),
    "AF2": (13.0,   76.0), "AF4": (26.0,   74.0), "AF6": (39.0,  74.0),
    "AF8": (51.0,   71.0), "AF10":(63.0,   67.0),
    # Frontal
    "T1":  (-90.0,  15.0), "F9":  (-71.0,  52.0), "F7":  (-55.0,  57.0),
    "F5":  (-43.0,  62.0), "F3":  (-29.0,  65.0), "F1":  (-14.0,  68.0),
    "FZ":  (0.0,   70.0),  "F2":  (14.0,   68.0),  "F4":  (29.0,  65.0),
    "F6":  (43.0,   62.0), "F8":  (55.0,   57.0),  "F10": (71.0,  52.0),
    "T2":  (90.0,   15.0),
    # Fronto-central
    "FT9": (-81.0,  25.0), "FT7": (-63.0,  37.0), "FC5": (-50.0,  41.0),
    "FC3": (-33.0,  44.0), "FC1": (-16.0,  46.0), "FCZ": (0.0,   47.0),
    "FC2": (16.0,   46.0), "FC4": (33.0,   44.0), "FC6": (50.0,  41.0),
    "FT8": (63.0,   37.0), "FT10":(81.0,   25.0),
    # Central / temporal
    "A1":  (-95.0,   0.0), "T9":  (-90.0,   0.0), "T7":  (-72.0,  0.0),
    "C5":  (-54.0,   0.0), "C3":  (-36.0,   0.0), "C1":  (-18.0,  0.0),
    "CZ":  (0.0,    0.0),  "C2":  (18.0,    0.0), "C4":  (36.0,   0.0),
    "C6":  (54.0,   0.0),  "C8":  (63.0,    0.0), "T8":  (72.0,   0.0),
    "T10": (90.0,   0.0),  "A2":  (95.0,    0.0),
    # Old names aliased to same position
    "T3":  (-72.0,  0.0),  "T4":  (72.0,   0.0),
    # Central-parietal
    "TP9": (-81.0, -25.0), "TP7": (-63.0, -37.0), "CP5": (-50.0, -41.0),
    "CP3": (-33.0, -44.0), "CP1": (-16.0, -46.0), "CPZ": (0.0,  -47.0),
    "CP2": (16.0,  -46.0), "CP4": (33.0,  -44.0), "CP6": (50.0, -41.0),
    "TP8": (63.0,  -37.0), "TP10":(81.0,  -25.0),
    # Parietal / temporal
    "P9":  (-71.0, -52.0), "P7":  (-55.0, -57.0), "P5":  (-43.0, -62.0),
    "P3":  (-29.0, -65.0), "P1":  (-14.0, -68.0), "PZ":  (0.0,  -70.0),
    "P2":  (14.0,  -68.0), "P4":  (29.0,  -65.0), "P6":  (43.0, -62.0),
    "P8":  (55.0,  -57.0), "P10": (71.0,  -52.0),
    # Old names
    "T5":  (-55.0, -57.0), "T6":  (55.0,  -57.0),
    # Parieto-occipital
    "PO9": (-63.0, -67.0), "PO7": (-51.0, -71.0), "PO5": (-39.0, -74.0),
    "PO3": (-26.0, -74.0), "PO1": (-13.0, -76.0), "POZ": (0.0,  -77.0),
    "PO2": (13.0,  -76.0), "PO4": (26.0,  -74.0), "PO6": (39.0, -74.0),
    "PO8": (51.0,  -71.0), "PO10":(63.0,  -67.0),
    # Occipital
    "O1":  (-27.0, -85.0), "OZ":  (0.0,  -87.0),  "O2":  (27.0, -85.0),
    "I1":  (-27.0, -95.0), "IZ":  (0.0,  -97.0),  "I2":  (27.0, -95.0),
}

# ---------------------------------------------------------------------------
# Task-relevant electrodes by dataset (literature-derived, see §9.1).
# Motor imagery: sensorimotor cortex (Wolpaw et al., 2002; Pfurtscheller, 1999)
# Sleep: AASM standard montage (Berry et al., 2012)
# Cognitive workload: frontal midline theta + parietal alpha (Onton et al., 2005)
# Seizure (siena_scalp): temporal/occipital — broad seizure-onset zone literature
# Dementia (adftd): global EEG changes; no strong focal prior → broad Cz prior
# ---------------------------------------------------------------------------
_TASK_ELECTRODES: dict[str, list[str]] = {
    "bcic_2a": ["C3", "CZ", "C4", "FC3", "FCZ", "FC4", "CP3", "CPZ", "CP4"],
    "motor_mv_img": ["C3", "CZ", "C4", "FC3", "FCZ", "FC4", "CP3", "CPZ", "CP4"],
    "hmc": ["FZ", "CZ", "PZ", "O1", "O2", "F4", "C4", "C3"],
    "workload": ["FZ", "PZ", "F3", "F4", "CZ", "FCZ", "AF3", "AF4", "AFZ"],
    "siena_scalp": ["T7", "T8", "F7", "F8", "T5", "T6", "O1", "O2",
                    "T3", "T4", "FP1", "FP2", "FT7", "FT8"],
    "adftd": ["CZ"],   # diffuse prior, broad sigma used below
}

# Datasets that use a broad (near-uniform) prior
_BROAD_SIGMA: dict[str, float] = {
    "adftd": 70.0,
    "siena_scalp": 45.0,
}


def _gaussian_mixture(
    ch_positions: np.ndarray,        # [C, 2]
    centers: np.ndarray,             # [K, 2]
    sigma: float,
) -> np.ndarray:
    """Evaluate unnormalized Gaussian mixture at each channel position."""
    # [C, K] pairwise squared distances
    diff = ch_positions[:, None, :] - centers[None, :, :]   # [C, K, 2]
    sq_dist = (diff ** 2).sum(-1)                            # [C, K]
    weights = np.exp(-sq_dist / (2 * sigma ** 2))            # [C, K]
    return weights.sum(-1)                                    # [C]


def get_prior(
    dataset_name: str,
    ch_names: list[str],
    sigma: Optional[float] = None,
) -> Optional[np.ndarray]:
    """
    Return normalized spatial prior p_t over the given channel list.

    Parameters
    ----------
    dataset_name : str
        One of the 6 benchmark datasets.
    ch_names : list[str]
        Channel names (uppercase, standard 10-10 format, e.g. ['CZ', 'C3', ...]).
    sigma : float, optional
        Gaussian width in mm. Defaults to 20 mm (focal prior) or dataset-specific
        broad sigma from _BROAD_SIGMA.

    Returns
    -------
    np.ndarray [len(ch_names)], normalized to sum to 1.
    None if dataset unknown or no channel positions found.
    """
    if dataset_name not in _TASK_ELECTRODES:
        return None
    if not ch_names:
        return None

    task_sigma = sigma if sigma is not None else _BROAD_SIGMA.get(dataset_name, 20.0)
    relevant = _TASK_ELECTRODES[dataset_name]

    # Resolve positions for task-relevant electrodes (skip unknowns)
    center_positions = []
    for name in relevant:
        pos = _POS.get(name.upper())
        if pos is not None:
            center_positions.append(pos)
    if not center_positions:
        return None
    centers = np.array(center_positions, dtype=np.float32)

    # Resolve positions for dataset channels
    ch_xy = []
    fallback_pos = np.array([0.0, 0.0], dtype=np.float32)  # Cz as fallback
    for ch in ch_names:
        pos = _POS.get(ch.upper())
        ch_xy.append(pos if pos is not None else fallback_pos)
    ch_positions = np.array(ch_xy, dtype=np.float32)  # [C, 2]

    weights = _gaussian_mixture(ch_positions, centers, task_sigma)
    total = weights.sum()
    if total <= 0:
        return np.ones(len(ch_names), dtype=np.float32) / len(ch_names)
    return (weights / total).astype(np.float32)


def get_all_dataset_priors(sigma: Optional[float] = None) -> dict:
    """Return prior definitions (metadata) for all 6 datasets."""
    return {
        ds: {
            "relevant_electrodes": _TASK_ELECTRODES.get(ds, []),
            "sigma_mm": sigma if sigma is not None else _BROAD_SIGMA.get(ds, 20.0),
        }
        for ds in ["bcic_2a", "hmc", "adftd", "motor_mv_img", "siena_scalp", "workload"]
    }


if __name__ == "__main__":
    # Quick smoke test: print priors for all datasets with HMC's 4-channel montage
    hmc_chs = ["F4", "C4", "O2", "C3"]
    p = get_prior("hmc", hmc_chs)
    print("HMC prior (F4, C4, O2, C3):", p)

    motor_chs = ["FZ", "FC3", "FC1", "FCZ", "FC2", "FC4", "C5", "C3", "C1",
                 "CZ", "C2", "C4", "C6", "CP3", "CP1", "CPZ", "CP2", "CP4",
                 "P1", "PZ", "P2", "POZ"]
    p2 = get_prior("bcic_2a", motor_chs)
    print("BCIC-2A prior (22 channels):", np.round(p2, 4))
