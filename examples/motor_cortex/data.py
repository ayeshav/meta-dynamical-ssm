from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subject_id(session_key: str) -> str:
    """'sub-C_ses-CO-20131003_...' → 'C'"""
    return session_key.split("_")[0].split("-")[1]


def _instability_score(rates: torch.Tensor) -> dict[str, float]:
    """Quantify firing-rate non-stationarity across trials.

    drift: |linear trend slope| × n_trials / mean_rate — fraction of the mean
           the population rate drifts over the full session. High (>~0.5) means
           slow electrode drift.
    cv:    std / mean of per-trial mean rates — high (>~0.4) means erratic
           bursts or sudden dropouts.
    """
    trial_mean = rates.mean(dim=(1, 2)).numpy().astype(np.float64)  # [n_trials]
    n = len(trial_mean)
    mu = trial_mean.mean()
    if mu < 1e-8:
        return {"drift": 0.0, "cv": 0.0}
    t = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    slope = float(np.dot(t, trial_mean) / (np.dot(t, t) + 1e-12))
    drift = abs(slope * n) / mu
    cv = float(trial_mean.std() / mu)
    return {"drift": drift, "cv": cv}


def _decode_r2(
    rates: torch.Tensor,
    velocity: torch.Tensor,
    n_folds: int = 5,
    *,
    start_bin: int = 12,
    alpha: float = 0.01,
) -> float:
    """Notebook-style Ridge R² from rates → velocity.

    Uses the first 80% of trials for training and the final 20% for validation,
    discards bins before `start_bin`, then flattens trials and time bins before
    fitting a Ridge decoder. `n_folds` is kept for config/API compatibility.
    """
    del n_folds

    if rates.ndim != 3 or velocity.ndim != 3 or rates.shape[:2] != velocity.shape[:2]:
        return float("-inf")

    n_trials, n_bins, n_neurons = rates.shape
    if n_trials < 2 or n_bins <= start_bin or n_neurons == 0:
        return float("-inf")

    split = int(n_trials * 0.8)
    if split <= 0 or split >= n_trials:
        return float("-inf")

    X_train = rates[:split, start_bin:].reshape(-1, n_neurons).numpy().astype(np.float64)
    Y_train = velocity[:split, start_bin:].reshape(-1, 2).numpy().astype(np.float64)
    X_val = rates[split:, start_bin:].reshape(-1, n_neurons).numpy().astype(np.float64)
    Y_val = velocity[split:, start_bin:].reshape(-1, 2).numpy().astype(np.float64)

    if len(X_train) == 0 or len(X_val) == 0:
        return float("-inf")

    # Match sklearn Ridge's default intercept by centering on the training set.
    x_mean = X_train.mean(axis=0, keepdims=True)
    y_mean = Y_train.mean(axis=0, keepdims=True)
    Xc = X_train - x_mean
    Yc = Y_train - y_mean

    W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(n_neurons), Xc.T @ Yc)
    Y_pred = (X_val - x_mean) @ W + y_mean

    ss_res = np.sum((Y_val - Y_pred) ** 2)
    ss_tot = np.sum((Y_val - Y_val.mean(axis=0, keepdims=True)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class MotorCortexData:
    train: dict[str, torch.Tensor]       # session_key -> [n_trials, T, n_neurons]
    val: dict[str, torch.Tensor]         # held-out within-subject sessions (same format)
    metadata: dict[str, dict]            # all sessions (train + val)
    decode_r2: dict[str, float]          # all sessions, pre-threshold filter
    stability: dict[str, dict]           # all sessions: {"drift": float, "cv": float}

    @property
    def train_dims(self) -> dict[str, int]:
        return {k: v.shape[-1] for k, v in self.train.items()}


# ---------------------------------------------------------------------------
# Session loading and splitting
# ---------------------------------------------------------------------------

def _load_co_sessions(
    path: str,
    train_subjects: list[str],
    decode_r2_threshold: float,
    val_fraction: float,
    n_folds: int,
    max_rate_drift: float,
    max_rate_cv: float,
) -> tuple[dict, dict, dict, dict, dict]:
    """Load CO sessions, filter by subject, held-out R², and rate stability; split train/val chronologically."""
    raw = torch.load(path, map_location="cpu", weights_only=False)

    # Score all sessions first
    all_r2: dict[str, float] = {}
    all_stability: dict[str, dict] = {}
    passed: list[dict] = []
    for s in raw:
        key = s["sub"]
        r2 = _decode_r2(s["rates"], s["velocity"], n_folds=n_folds)
        stab = _instability_score(s["rates"])
        all_r2[key] = r2
        all_stability[key] = stab
        if (
            _subject_id(key) in train_subjects
            and r2 >= decode_r2_threshold
            and stab["drift"] <= max_rate_drift
            and stab["cv"] <= max_rate_cv
        ):
            passed.append(s)

    # Group by subject and sort chronologically (key contains YYYYMMDD)
    by_subject: dict[str, list[dict]] = {}
    for s in passed:
        subj = _subject_id(s["sub"])
        by_subject.setdefault(subj, []).append(s)
    for subj in by_subject:
        by_subject[subj].sort(key=lambda s: s["sub"])

    train_obs: dict[str, torch.Tensor] = {}
    val_obs: dict[str, torch.Tensor] = {}
    meta: dict[str, dict] = {}

    for subj, sessions in by_subject.items():
        n_val = max(1, round(len(sessions) * val_fraction))
        train_sessions = sessions[:-n_val]
        val_sessions = sessions[-n_val:]

        for s in train_sessions:
            key = s["sub"]
            train_obs[key] = s["rates"]
            meta[key] = {k: v for k, v in s.items() if k not in ("y", "rates", "velocity")}
            meta[key]["task"] = "centre_out"
            meta[key]["split"] = "train"

        for s in val_sessions:
            key = s["sub"]
            val_obs[key] = s["rates"]
            meta[key] = {k: v for k, v in s.items() if k not in ("y", "rates", "velocity")}
            meta[key]["task"] = "centre_out"
            meta[key]["split"] = "val"

    return train_obs, val_obs, meta, all_r2, all_stability


def _load_maze_sessions(
    path: str,
    decode_r2_threshold: float,
    n_folds: int,
    max_rate_drift: float,
    max_rate_cv: float,
) -> tuple[dict, dict, dict, dict]:
    raw = torch.load(path, map_location="cpu", weights_only=False)

    train_obs: dict[str, torch.Tensor] = {}
    all_r2: dict[str, float] = {}
    all_stability: dict[str, dict] = {}
    meta: dict[str, dict] = {}

    for s in raw:
        key = s["subject"]
        r2 = _decode_r2(s["rates"], s["velocity"], n_folds=n_folds)
        stab = _instability_score(s["rates"])
        all_r2[key] = r2
        all_stability[key] = stab
        if (
            r2 >= decode_r2_threshold
            and stab["drift"] <= max_rate_drift
            and stab["cv"] <= max_rate_cv
        ):
            train_obs[key] = s["rates"]
            meta[key] = {k: v for k, v in s.items() if k not in ("y", "rates", "velocity")}
            meta[key]["split"] = "train"

    return train_obs, meta, all_r2, all_stability


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_motor_cortex_data(
    co_path: str,
    maze_path: str,
    decode_r2_threshold: float = 0.1,
    n_folds: int = 5,
    tasks: tuple[str, ...] = ("centre_out", "maze"),
    co_train_subjects: list[str] | None = None,
    co_val_fraction: float = 0.2,
    max_rate_drift: float = 0.5,
    max_rate_cv: float = 0.4,
    device: str | torch.device = "cpu",
) -> MotorCortexData:
    """Load and prepare motor cortex sessions for meta-SSM training.

    CO sessions:
      - Filtered to `co_train_subjects` (default: all subjects).
      - Sorted chronologically per subject; last `co_val_fraction` held out as val.
      - Uses smoothed `rates` as observations (Gaussian likelihood).

    Maze sessions:
      - All passing R² and stability thresholds go to train; no held-out split.

    Both tasks filtered by notebook-style held-out ridge R² (rates → velocity)
    and firing-rate stability across trials (drift and CV thresholds).
    """
    if co_train_subjects is None:
        co_train_subjects = ["C", "M", "J", "T"]

    train_obs: dict[str, torch.Tensor] = {}
    val_obs: dict[str, torch.Tensor] = {}
    all_meta: dict[str, dict] = {}
    all_r2: dict[str, float] = {}
    all_stability: dict[str, dict] = {}

    if "centre_out" in tasks:
        tr, val, meta, r2, stab = _load_co_sessions(
            co_path, co_train_subjects, decode_r2_threshold, co_val_fraction, n_folds,
            max_rate_drift, max_rate_cv,
        )
        train_obs.update(tr)
        val_obs.update(val)
        all_meta.update(meta)
        all_r2.update(r2)
        all_stability.update(stab)

    if "maze" in tasks:
        tr, meta, r2, stab = _load_maze_sessions(
            maze_path, decode_r2_threshold, n_folds, max_rate_drift, max_rate_cv,
        )
        train_obs.update(tr)
        all_meta.update(meta)
        all_r2.update(r2)
        all_stability.update(stab)

    train_obs = {k: v.to(device) for k, v in train_obs.items()}
    val_obs = {k: v.to(device) for k, v in val_obs.items()}

    return MotorCortexData(
        train=train_obs, val=val_obs, metadata=all_meta,
        decode_r2=all_r2, stability=all_stability,
    )


def sample_batch(
    observations: dict[str, torch.Tensor],
    batch_size: int,
    *,
    n_sessions: int | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Sample `batch_size` trials from each session.

    If `n_sessions` is given, randomly subsample that many sessions per step
    instead of using all of them — reduces gradient noise from rare sessions
    and prevents overfitting to small-dataset sessions.
    """
    keys = list(observations.keys())
    if n_sessions is not None and n_sessions < len(keys):
        perm = torch.randperm(len(keys), generator=generator)
        keys = [keys[i] for i in perm[:n_sessions].tolist()]
    batch = {}
    for key in keys:
        y = observations[key]
        n = y.shape[0]
        size = min(batch_size, n)
        indices = torch.randint(n, (size,), generator=generator, device=y.device)
        batch[key] = {"y": y[indices]}
    return batch
