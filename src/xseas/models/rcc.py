"""
Radially Constrained Clustering (RCC) implementation.

This module provides a clustering algorithm specifically designed for seasonal
time series analysis with circular constraints.
"""
from typing import Optional, Tuple, List
from random import randint
import numpy as np


class RCC:
    """
    Radially Constrained Clustering for seasonal time series analysis.
    
    This algorithm identifies seasonal breakpoints in time series data while
    respecting circular time constraints and minimum season length requirements.
    
    Parameters
    ----------
    data_to_cluster : np.ndarray
        Time series data with timesteps on first dimension and features on second.
    n_seas : int
        Number of seasons/clusters to identify.
    n_iter : int, default=1000
        Maximum number of iterations for optimization.
    learning_rate : int, default=1
        Maximum number of days for stochastic breakpoint updates.
    scheduling_factor : int, default=1
        Factor for reducing learning rate during optimization.
    min_len : int, default=1
        Minimum length for each season in days.
    starting_bp : Optional[List[int]], default=None
        Initial breakpoints. If None, generates equally distributed breakpoints.
        
    Raises
    ------
    ValueError
        If impossible to create n_seas seasons of min_len days given data length.
    """
    
    def __init__(
        self,
        data_to_cluster: np.ndarray,
        n_seas: int,
        n_iter: int = 1000,
        learning_rate: int = 1,
        scheduling_factor: int = 1,
        min_len: int = 1,
        starting_bp: Optional[List[int]] = None
    ) -> None:
        self.len_serie = data_to_cluster.shape[0]
        self.data_to_cluster = data_to_cluster
        self.starting_bp = starting_bp
        
        # Validate parameters
        if self.len_serie / n_seas < min_len:
            raise ValueError(
                f'Cannot create {n_seas} seasons of {min_len} days. '
                f'Data length: {self.len_serie}, required: {n_seas * min_len}'
            )
        
        self.n_seas = n_seas
        self.min_len = min_len
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.scheduling_factor = scheduling_factor
        
        # Results will be stored here after fitting
        self.breakpoints: Optional[np.ndarray] = None
        self.centroid_history: Optional[List] = None
        self.error_history: Optional[np.ndarray] = None
        self.breakpoint_history: Optional[List] = None
        self.learningrate_history: Optional[List] = None
        self.prediction_history: Optional[List] = None
    
    def fit(self) -> None:
        """Fit the RCC model to the data."""
        results = self._single_fit()
        (
            self.breakpoints,
            self.centroid_history,
            self.error_history,
            self.breakpoint_history,
            self.learningrate_history,
            self.prediction_history
        ) = results
    
    def _single_fit(self) -> Tuple:
        """
        Core fitting algorithm.
        
        Returns
        -------
        Tuple containing breakpoints, centroids, error history, 
        breakpoint history, learning rate history, and prediction history.
        """
        # Initialize tracking lists
        prediction_history = []
        breakpoint_list = []
        centroid_list = []
        error_list = []
        learningrate_list = []
        
        for iteration in range(self.n_iter):
            # Generate or update breakpoints
            if iteration == 0:
                upgrade, breakpoints = self._generate_starting_breakpoints()
            else:
                upgrade, breakpoints = self._update_breakpoints(breakpoints)
            
            # Generate season indices
            season_indices = self._generate_season_indices(breakpoints)
            
            # Check minimum season length constraint
            if not self._check_season_lengths(season_indices):
                breakpoints = self._revert_breakpoints(breakpoints, upgrade)
                continue
            
            # Update stored breakpoints and compute metrics
            self.breakpoints = breakpoints
            breakpoint_list.append(self.breakpoints.copy())
            
            prediction = self._get_prediction()
            prediction_history.append(prediction)
            
            centroids, error = self._compute_metrics(season_indices)
            centroid_list.append(centroids)
            error_list.append(np.nanmean(error))
            learningrate_list.append(self.learning_rate)
            
            # Optimize based on error improvement
            if iteration > 1:
                if error_list[-1] > error_list[-2]:
                    breakpoints = self._revert_breakpoints(breakpoints, upgrade)
                elif (
                    error_list[-1] < error_list[-2] and 
                    self.scheduling_factor > 1 and 
                    self.learning_rate > 1 and 
                    iteration > 3
                ):
                    self.learning_rate = self._schedule_learning_rate()
        
        return (
            np.sort(np.int32(breakpoints)),
            np.float64(centroid_list),
            np.float64(error_list),
            np.int32(breakpoint_list),
            np.int32(learningrate_list),
            np.int32(prediction_history)
        )
    
    def _generate_starting_breakpoints(self) -> Tuple[List[int], np.ndarray]:
        """Generate initial breakpoints equally distributed over time."""
        if self.starting_bp is not None:
            return [0] * self.n_seas, np.array(self.starting_bp)
        
        step = self.len_serie // self.n_seas
        breakpoints = []
        
        for i in range(self.n_seas):
            bp = step * (i + 1)
            if bp >= self.len_serie:
                bp = bp - self.len_serie
            breakpoints.append(bp)
        
        return [0] * self.n_seas, np.sort(np.array(breakpoints))
    
    def _update_breakpoints(self, old_breakpoints: np.ndarray) -> Tuple[List[int], np.ndarray]:
        """Update breakpoints with random perturbations."""
        upgrades = []
        new_breakpoints = []
        
        for bp in old_breakpoints:
            upgrade = randint(-self.learning_rate, self.learning_rate)
            new_bp = (bp + upgrade) % self.len_serie
            upgrades.append(upgrade)
            new_breakpoints.append(new_bp)
        
        return upgrades, np.array(new_breakpoints)
    
    def _generate_season_indices(self, breakpoints: np.ndarray) -> List[np.ndarray]:
        """Generate time indices for each season based on breakpoints."""
        if self.n_seas == 1:
            return [np.arange(self.len_serie)]
        
        indices = []
        sorted_bp = np.sort(breakpoints)
        
        for i in range(self.n_seas):
            start = sorted_bp[i]
            end = sorted_bp[(i + 1) % self.n_seas]
            
            if start >= end:  # Wraps around year boundary
                idx = np.concatenate([
                    np.arange(start, self.len_serie),
                    np.arange(0, end)
                ])
            else:
                idx = np.arange(start, end)
            
            indices.append(idx)
        
        return indices
    
    def _check_season_lengths(self, indices: List[np.ndarray]) -> bool:
        """Check if all seasons meet minimum length requirement."""
        return all(len(idx) >= self.min_len for idx in indices)
    
    def _revert_breakpoints(self, breakpoints: np.ndarray, upgrades: List[int]) -> np.ndarray:
        """Revert breakpoints to previous iteration."""
        reverted = []
        for bp, upgrade in zip(breakpoints, upgrades):
            reverted_bp = (bp - upgrade) % self.len_serie
            reverted.append(reverted_bp)
        return np.array(reverted)
    
    def _compute_metrics(self, indices: List[np.ndarray]) -> Tuple[List, List]:
        """Compute centroids and error for each season."""
        centroids, errors = [], []
        
        for idx in indices:
            cluster_data = self.data_to_cluster[idx]
            centroid = np.nanmean(cluster_data, axis=0)
            error = np.nansum(np.power(cluster_data - centroid, 2))
            
            centroids.append(centroid)
            errors.append(error)
        
        return centroids, errors
    
    def _schedule_learning_rate(self) -> int:
        """Reduce learning rate for fine-tuning."""
        return max(1, self.learning_rate // self.scheduling_factor)
    
    def _get_prediction(self) -> np.ndarray:
        """Get season labels for each time step."""
        prediction = np.zeros(self.len_serie)
        indices = self._generate_season_indices(self.breakpoints)
        
        for season_id, idx in enumerate(indices):
            prediction[idx] = season_id
        
        return prediction.astype(int)
    
    # Public methods for accessing results
    def get_prediction(self) -> np.ndarray:
        """Get the final season prediction."""
        if self.breakpoints is None:
            raise RuntimeError("Model must be fitted before getting predictions")
        return self._get_prediction()
    
    def get_final_error(self) -> float:
        """Get the final total error."""
        if self.breakpoints is None:
            raise RuntimeError("Model must be fitted before getting error")
        
        indices = self._generate_season_indices(self.breakpoints)
        _, errors = self._compute_metrics(indices)
        return np.sum(errors)
    
    def get_centroids(self) -> List:
        """Get the final centroids."""
        if self.breakpoints is None:
            raise RuntimeError("Model must be fitted before getting centroids")
        
        indices = self._generate_season_indices(self.breakpoints)
        centroids, _ = self._compute_metrics(indices)
        return centroids
    
    def get_indices(self) -> List[np.ndarray]:
        """Get the final season indices."""
        if self.breakpoints is None:
            raise RuntimeError("Model must be fitted before getting indices")
        return self._generate_season_indices(self.breakpoints)





