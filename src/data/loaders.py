"""Data loaders for cancer treatment sequences."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CancerDataLoader:
    """Base class for loading cancer datasets."""

    def __init__(self, source: str = "synthetic"):
        """
        Initialize data loader.

        Args:
            source: Data source ('synthetic', 'seer', 'custom')
        """
        self.source = source

    def load(self, **kwargs) -> Dict:
        """
        Load dataset from specified source.

        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        if self.source == "synthetic":
            return self._load_synthetic(**kwargs)
        elif self.source == "seer":
            return self._load_seer(**kwargs)
        else:
            raise ValueError(f"Unknown source: {self.source}")

    def _load_synthetic(
        self,
        n_patients: int = 1000,
        n_timesteps: int = 4,
        n_covariates: int = 10,
        confounding_strength: float = 1.0,
        seed: int = 42,
    ) -> Dict:
        """
        Generate synthetic cancer treatment dataset.

        Args:
            n_patients: Number of patients
            n_timesteps: Number of treatment decision points
            n_covariates: Number of covariates per timestep
            confounding_strength: Strength of time-dependent confounding
            seed: Random seed

        Returns:
            Dictionary with patient data
        """
        np.random.seed(seed)
        logger.info(f"Generating synthetic dataset: {n_patients} patients, {n_timesteps} timesteps")

        # Generate baseline covariates
        X = np.random.randn(n_patients, n_covariates)  # Shape: (N, d_x)
        X[:, 0] = np.abs(X[:, 0])  # Age-like covariate

        # Generate treatment sequences
        # A_t ∈ {0: no_treatment, 1: chemo, 2: radio, 3: immuno}
        A = np.zeros((n_patients, n_timesteps), dtype=int)
        L = np.zeros((n_patients, n_timesteps))  # Disease response/confounder

        for t in range(n_timesteps):
            # L_t: disease state (partially observed confounder)
            if t == 0:
                L[:, t] = X[:, 0] + np.random.randn(n_patients) * 0.5
            else:
                # L_t depends on previous treatment and L_{t-1}
                L[:, t] = (
                    L[:, t - 1]
                    - 0.5 * A[:, t - 1]
                    + confounding_strength * np.random.randn(n_patients) * 0.3
                )

            # Treatment assignment: confounded by L_t (time-dependent confounding)
            prob_treatment = 1 / (1 + np.exp(-(-L[:, t] + np.random.randn(n_patients) * 0.2)))
            A[:, t] = (prob_treatment > 0.5).astype(int)

        # Generate survival times
        T = np.zeros(n_patients)
        for i in range(n_patients):
            # True effect: receiving more treatments → better survival
            treatment_effect = -np.sum(A[i, :]) * 0.3
            # Confounding effect: low disease markers → better survival
            confounding_effect = -L[i, -1] * 0.5
            # Baseline hazard
            baseline_hazard = -np.log(np.random.uniform(0, 1)) / (X[i, 0] * 0.1)
            T[i] = baseline_hazard + treatment_effect + confounding_effect

        # Event indicators (some patients censored)
        E = np.random.binomial(1, 0.7, n_patients)

        # Assemble dataset
        data = {
            "X": X,  # (N, d_x) baseline and time-varying covariates
            "A": A,  # (N, T) treatment sequences
            "L": L,  # (N, T) disease response (partially observed)
            "T": T,  # (N,) survival times
            "E": E,  # (N,) event indicators
        }

        logger.info(f"Generated data - Mean survival: {T.mean():.2f}, Event rate: {E.mean():.2%}")

        # Train/Val/Test split
        splits = self._split_data(data, train_frac=0.6, val_frac=0.2)
        return splits

    def _load_seer(self, cohort: str = "lung_cancer", **kwargs) -> Dict:
        """
        Load SEER cancer registry data.

        Note: Requires downloading SEER data from https://seer.cancer.gov/

        Args:
            cohort: Cancer type cohort

        Returns:
            Dictionary with patient data
        """
        raise NotImplementedError(
            "SEER data loading not implemented. "
            "Please download data from https://seer.cancer.gov/ "
            "and implement custom loader."
        )

    @staticmethod
    def _split_data(data: Dict, train_frac: float = 0.6, val_frac: float = 0.2) -> Dict:
        """
        Split dataset into train/val/test.

        Args:
            data: Raw data dictionary
            train_frac: Fraction for training
            val_frac: Fraction for validation

        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        n = data["X"].shape[0]
        indices = np.random.permutation(n)

        train_idx = indices[: int(n * train_frac)]
        val_idx = indices[int(n * train_frac) : int(n * (train_frac + val_frac))]
        test_idx = indices[int(n * (train_frac + val_frac)) :]

        splits = {}
        for split_name, split_idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            splits[split_name] = {key: val[split_idx] for key, val in data.items()}

        return splits


def load_cancer_dataset(
    source: str = "synthetic",
    **kwargs,
) -> Dict:
    """
    Convenience function to load cancer dataset.

    Args:
        source: Data source ('synthetic', 'seer')
        **kwargs: Additional arguments passed to loader

    Returns:
        Dictionary with 'train', 'val', 'test' splits
    """
    loader = CancerDataLoader(source=source)
    return loader.load(**kwargs)


if __name__ == "__main__":
    # Test synthetic data generation
    data = load_cancer_dataset(
        source="synthetic",
        n_patients=100,
        n_timesteps=4,
        confounding_strength=1.0,
    )
    print("Train set shapes:")
    for key, val in data["train"].items():
        print(f"  {key}: {val.shape}")
