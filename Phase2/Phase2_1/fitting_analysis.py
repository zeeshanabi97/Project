"""
Dipole Magnetic Field Fitting Analysis
======================================
A robust framework for fitting dipole models to satellite magnetic field data
with proper train/test splits and comprehensive knee point detection.

Uses dipole_physics module for core physics calculations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cdflib
import warnings
from scipy.optimize import least_squares
from tqdm import tqdm
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List
from pathlib import Path

# Import core physics from dedicated module
from dipole_physics import (
    MU0_OVER_4PI,
    sph_to_cart,
    nec_to_cart,
    dipole_field,
    cart_to_sph_direction
)


warnings.filterwarnings('ignore')

# ==============================================================================
# CONSTANTS
# ==============================================================================

TESLA_TO_NANOTESLA = 1e9
NANOTESLA_TO_TESLA = 1e-9


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class MagneticData:
    """Container for magnetic field data."""
    radius: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    B_NEC: np.ndarray
    B_cart: np.ndarray
    r_cart: np.ndarray
    
    def __len__(self) -> int:
        return len(self.radius)
    
    @property
    def max_field_magnitude(self) -> float:
        """Maximum magnetic field magnitude in nT."""
        return np.max(np.linalg.norm(self.B_cart, axis=1)) * TESLA_TO_NANOTESLA


@dataclass
class FitStatistics:
    """Statistics from dipole model fitting."""
    n_samples: int
    strategy: str
    rms_total: float
    rms_x: float
    rms_y: float
    rms_z: float
    mae_total: float
    relative_error: float
    r_squared: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            'n_samples': self.n_samples,
            'strategy': self.strategy,
            'rms_total': self.rms_total,
            'rms_x': self.rms_x,
            'rms_y': self.rms_y,
            'rms_z': self.rms_z,
            'mae_total': self.mae_total,
            'relative_error': self.relative_error,
            'r_squared': self.r_squared
        }


@dataclass
class KneePoints:
    """Detected knee points from various algorithms."""
    threshold: Optional[float] = None
    curvature: Optional[float] = None
    kneedle: Optional[float] = None
    practical: Optional[float] = None
    
    def has_valid_knee(self) -> bool:
        """Check if any knee point was detected."""
        return any(k is not None and not np.isnan(k) 
                   for k in [self.threshold, self.curvature, self.kneedle, self.practical])


# ==============================================================================
# DIPOLE MODEL
# ==============================================================================

class DipoleModel:
    """Magnetic dipole field model."""
    
    def __init__(self):
        self.dipole_moment: Optional[np.ndarray] = None
        self.dipole_position: Optional[np.ndarray] = None
    
    def fit(self, data: MagneticData, indices: np.ndarray) -> bool:
        """
        Fit dipole model to data subset using physics module.
        
        Args:
            data: Full magnetic data
            indices: Indices of data points to use for fitting
            
        Returns:
            Success status of the fit
        """
        initial_params = np.array([8e22, 8e22, 8e22, 1000, 1000, 1000])
        
        def residuals_wrapper(params):
            m = params[:3]
            r0 = params[3:]
            r_subset = data.r_cart[indices]
            B_model = dipole_field(r_subset, m, r0)
            B_obs = data.B_cart[indices]
            return (B_obs - B_model).flatten()
        
        result = least_squares(
            residuals_wrapper,
            initial_params,
            method='lm',
            ftol=1e-15,
            xtol=1e-15,
            gtol=1e-15,
            max_nfev=1000,
            verbose=0
        )
        
        if result.success:
            self.dipole_moment = result.x[:3]
            self.dipole_position = result.x[3:]
            return True
        
        return False
    
    def predict(self, r: np.ndarray) -> np.ndarray:
        """Predict magnetic field at given positions using physics module."""
        if self.dipole_moment is None or self.dipole_position is None:
            raise ValueError("Model must be fitted before prediction")
        
        return dipole_field(r, self.dipole_moment, self.dipole_position)


# ==============================================================================
# STATISTICS
# ==============================================================================

class StatisticsCalculator:
    """Calculate various error statistics and goodness-of-fit metrics."""
    
    @staticmethod
    def calculate(B_observed: np.ndarray, 
                 B_predicted: np.ndarray,
                 n_samples: int,
                 strategy: str) -> FitStatistics:
        """Calculate comprehensive statistics for model fit."""
        residuals = B_observed - B_predicted
        
        # RMS errors
        rms_total = np.sqrt(np.mean(residuals ** 2)) * TESLA_TO_NANOTESLA
        rms_components = np.sqrt(np.mean(residuals ** 2, axis=0)) * TESLA_TO_NANOTESLA
        
        # MAE
        mae_total = np.mean(np.abs(residuals)) * TESLA_TO_NANOTESLA
        
        # Relative error
        B_magnitude = np.linalg.norm(B_observed, axis=1, keepdims=True)
        residual_magnitude = np.linalg.norm(residuals, axis=1)
        relative_error = np.mean(residual_magnitude / B_magnitude.flatten()) * 100
        
        # R-squared
        ss_residual = np.sum(residuals ** 2)
        ss_total = np.sum((B_observed - np.mean(B_observed, axis=0)) ** 2)
        r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
        
        return FitStatistics(
            n_samples=n_samples,
            strategy=strategy,
            rms_total=rms_total,
            rms_x=rms_components[0],
            rms_y=rms_components[1],
            rms_z=rms_components[2],
            mae_total=mae_total,
            relative_error=relative_error,
            r_squared=r_squared
        )


# ==============================================================================
# KNEE DETECTION
# ==============================================================================

class KneeDetector:
    """Detect knee points in sample size vs error curves."""
    
    def __init__(self, validation_threshold: float = 0.1):
        """
        Args:
            validation_threshold: Fraction of max |B| that error must be below
        """
        self.validation_threshold = validation_threshold
    
    def _validate_knee(self, error_value: float, max_field: float) -> bool:
        """Check if error is below validation threshold."""
        return error_value < self.validation_threshold * max_field
    
    def detect_all(self, df: pd.DataFrame, 
                   max_field: float,
                   metric: str = 'rms_total') -> KneePoints:
        """
        Detect knee points using multiple algorithms.
        
        Args:
            df: DataFrame with results sorted by n_samples
            max_field: Maximum observed field magnitude for validation
            metric: Column name of metric to analyze
            
        Returns:
            KneePoints object with detected knees
        """
        df_sorted = df.sort_values('n_samples').copy()
        
        return KneePoints(
            threshold=self._threshold_method(df_sorted, max_field, metric),
            curvature=self._curvature_method(df_sorted, max_field, metric),
            kneedle=self._kneedle_method(df_sorted, max_field, metric),
            practical=self._practical_method(df_sorted, max_field, metric)
        )
    
    def _threshold_method(self, df: pd.DataFrame, 
                         max_field: float,
                         metric: str,
                         threshold_percent: float = 0.5) -> Optional[float]:
        """Find knee where improvement drops below threshold percentage."""
        if len(df) < 3:
            return None
        
        values = df[metric].values
        n_samples = df['n_samples'].values
        
        for i in range(1, len(values)):
            improvement = (values[i-1] - values[i]) / values[i-1] * 100
            if improvement < threshold_percent:
                if self._validate_knee(values[i], max_field):
                    return n_samples[i]
        
        return None
    
    def _curvature_method(self, df: pd.DataFrame,
                         max_field: float,
                         metric: str) -> Optional[float]:
        """Find knee using maximum curvature in log-log space."""
        if len(df) < 5:
            return None
        
        values = df[metric].values
        n_samples = df['n_samples'].values
        
        log_n = np.log10(n_samples)
        log_values = np.log10(values)
        
        first_deriv = np.gradient(log_values, log_n)
        second_deriv = np.gradient(first_deriv, log_n)
        
        valid_range = range(2, len(second_deriv) - 2) if len(second_deriv) > 4 else range(1, len(second_deriv) - 1)
        
        if len(list(valid_range)) == 0:
            return None
        
        valid_curvatures = second_deriv[list(valid_range)]
        knee_idx = list(valid_range)[np.argmin(valid_curvatures)]
        
        if self._validate_knee(values[knee_idx], max_field):
            return n_samples[knee_idx]
        
        return None
    
    def _kneedle_method(self, df: pd.DataFrame,
                       max_field: float,
                       metric: str,
                       sensitivity: float = 0.3) -> Optional[float]:
        """Find knee using Kneedle algorithm."""
        if len(df) < 3:
            return None
        
        x = df['n_samples'].values
        y = df[metric].values
        
        x_norm = (x - x.min()) / (x.max() - x.min())
        y_norm = (y - y.min()) / (y.max() - y.min())
        y_norm = 1 - y_norm
        
        difference = y_norm - x_norm
        threshold = np.mean(difference) + sensitivity * np.std(difference)
        
        above_threshold = difference > threshold
        if not np.any(above_threshold):
            return None
        
        knee_idx = np.where(above_threshold)[0][0]
        
        if self._validate_knee(y[knee_idx], max_field):
            return x[knee_idx]
        
        return None
    
    def _practical_method(self, df: pd.DataFrame,
                         max_field: float,
                         metric: str,
                         improvement_threshold: float = 0.95) -> Optional[float]:
        """Find point where cumulative improvement reaches 95% of total."""
        if len(df) < 3:
            return None
        
        values = df[metric].values
        n_samples = df['n_samples'].values
        
        total_improvement = values[0] - values[-1]
        
        if total_improvement <= 0:
            return None
        
        for i in range(1, len(values)):
            cumulative_improvement = values[0] - values[i]
            fraction = cumulative_improvement / total_improvement
            
            if fraction >= improvement_threshold:
                if self._validate_knee(values[i], max_field):
                    return n_samples[i]
        
        return None


# ==============================================================================
# SAMPLING STRATEGIES
# ==============================================================================

class SamplingStrategy:
    """Base class for sampling strategies with validation."""
    
    def sample(self, pool_indices: np.ndarray, 
               n_samples: int, 
               data: MagneticData) -> np.ndarray:
        """Sample n_samples from pool_indices."""
        raise NotImplementedError
    
    def validate_sample_size(self, pool_size: int, n_samples: int) -> None:
        """Validate that requested sample size is feasible."""
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        if n_samples > pool_size:
            raise ValueError(
                f"Requested {n_samples} samples but pool only has {pool_size}"
            )


class RandomSampling(SamplingStrategy):
    """
    Random sampling from the data pool.
    
    Mathematical properties:
    - Each point has equal probability of selection: P(select) = n/N
    - Samples are independent (without replacement)
    - Expected to provide good spatial coverage if data is well-distributed
    - No temporal/spatial correlation in the sample
    """
    
    def sample(self, pool_indices: np.ndarray, 
               n_samples: int, 
               data: MagneticData) -> np.ndarray:
        self.validate_sample_size(len(pool_indices), n_samples)
        
        # np.random.choice with replace=False ensures no duplicates
        return np.random.choice(pool_indices, n_samples, replace=False)


class OrbitSampling(SamplingStrategy):
    """
    Sampling consecutive points in the dataset (temporal/orbital).
    
    Mathematical properties:
    - Selects a contiguous block from the orbit
    - Preserves temporal correlation structure
    - May lead to spatial clustering depending on orbit geometry
    - Can have poor generalization if orbit segment is atypical
    
    Implementation notes:
    - Selects random starting point in valid range
    - Extracts consecutive n_samples from that point
    - Handles edge cases where n_samples equals pool size
    """
    
    def sample(self, pool_indices: np.ndarray, 
               n_samples: int, 
               data: MagneticData) -> np.ndarray:
        self.validate_sample_size(len(pool_indices), n_samples)
        
        # Edge case: if requesting all samples, just return them
        if n_samples == len(pool_indices):
            return pool_indices
        
        # Calculate valid range for starting index
        max_start_idx = len(pool_indices) - n_samples
        
        # Randomly select starting position
        # Use randint for cleaner integer selection
        start_idx_local = np.random.randint(0, max_start_idx + 1)
        
        # Extract consecutive block
        selected_indices = pool_indices[start_idx_local:start_idx_local + n_samples]
        
        # Verification (optional, can remove for performance)
        assert len(selected_indices) == n_samples, \
            f"Expected {n_samples} samples, got {len(selected_indices)}"
        
        return selected_indices


class StratifiedSpatialSampling(SamplingStrategy):
    """
    Stratified sampling ensuring coverage across latitude/longitude bins.
    
    Mathematical properties:
    - Divides space into strata (grid cells)
    - Samples proportionally from each stratum
    - Reduces sampling variance compared to pure random
    - Better spatial coverage, especially for uneven data distribution
    
    This is an OPTIONAL enhanced strategy not in your original code.
    """
    
    def __init__(self, n_lat_bins: int = 10, n_lon_bins: int = 10):
        self.n_lat_bins = n_lat_bins
        self.n_lon_bins = n_lon_bins
    
    def sample(self, pool_indices: np.ndarray, 
               n_samples: int, 
               data: MagneticData) -> np.ndarray:
        self.validate_sample_size(len(pool_indices), n_samples)
        
        # Get data for pool
        lats = data.latitude[pool_indices]
        lons = data.longitude[pool_indices]
        
        # Create spatial bins
        lat_bins = np.linspace(-90, 90, self.n_lat_bins + 1)
        lon_bins = np.linspace(-180, 180, self.n_lon_bins + 1)
        
        lat_indices = np.digitize(lats, lat_bins) - 1
        lon_indices = np.digitize(lons, lon_bins) - 1
        
        # Combine into single stratum index
        strata = lat_indices * self.n_lon_bins + lon_indices
        unique_strata = np.unique(strata)
        
        # Calculate samples per stratum (proportional allocation)
        samples_per_stratum = {}
        for s in unique_strata:
            mask = strata == s
            n_in_stratum = np.sum(mask)
            proportion = n_in_stratum / len(pool_indices)
            samples_per_stratum[s] = max(1, int(n_samples * proportion))
        
        # Adjust to hit exact n_samples (due to rounding)
        total_allocated = sum(samples_per_stratum.values())
        if total_allocated > n_samples:
            # Remove from largest strata first
            sorted_strata = sorted(samples_per_stratum.items(), 
                                  key=lambda x: x[1], reverse=True)
            for s, count in sorted_strata:
                if total_allocated <= n_samples:
                    break
                if count > 1:
                    samples_per_stratum[s] -= 1
                    total_allocated -= 1
        elif total_allocated < n_samples:
            # Add to largest strata
            diff = n_samples - total_allocated
            sorted_strata = sorted(samples_per_stratum.items(), 
                                  key=lambda x: x[1], reverse=True)
            for i in range(diff):
                s = sorted_strata[i % len(sorted_strata)][0]
                samples_per_stratum[s] += 1
        
        # Sample from each stratum
        selected = []
        for s in unique_strata:
            mask = strata == s
            stratum_indices = pool_indices[mask]
            n_from_stratum = min(samples_per_stratum[s], len(stratum_indices))
            selected_from_stratum = np.random.choice(
                stratum_indices, n_from_stratum, replace=False
            )
            selected.extend(selected_from_stratum)
        
        selected = np.array(selected[:n_samples])
        return selected
    
# ==============================================================================
# VALIDATION FUNCTIONS
# ==============================================================================

def validate_train_test_split(train_indices: np.ndarray, 
                              test_indices: np.ndarray,
                              total_size: int) -> bool:
    """
    Validate that train/test split has no data leakage.
    
    Checks:
    1. No overlap between train and test
    2. All indices are valid
    3. Union covers expected portion of data
    """
    # Check for overlap (CRITICAL for unbiased evaluation)
    overlap = np.intersect1d(train_indices, test_indices)
    if len(overlap) > 0:
        raise ValueError(f"DATA LEAKAGE: {len(overlap)} samples in both train and test!")
    
    # Check bounds
    if np.any(train_indices < 0) or np.any(train_indices >= total_size):
        raise ValueError("Train indices out of bounds")
    if np.any(test_indices < 0) or np.any(test_indices >= total_size):
        raise ValueError("Test indices out of bounds")
    
    # Check uniqueness
    if len(np.unique(train_indices)) != len(train_indices):
        raise ValueError("Duplicate indices in training set")
    if len(np.unique(test_indices)) != len(test_indices):
        raise ValueError("Duplicate indices in test set")
    
    return True


def validate_sampling_coverage(indices: np.ndarray, 
                               data: MagneticData,
                               min_lat_range: float = 30.0,
                               min_lon_range: float = 60.0) -> Dict[str, float]:
    """
    Validate that sample has adequate spatial coverage.
    
    Returns metrics about the sample's spatial distribution.
    Useful for detecting degenerate samples (e.g., all from one region).
    """
    lats = data.latitude[indices]
    lons = data.longitude[indices]
    
    coverage = {
        'lat_range': lats.max() - lats.min(),
        'lon_range': lons.max() - lons.min(),
        'lat_std': np.std(lats),
        'lon_std': np.std(lons),
        'n_samples': len(indices)
    }
    
    # Warnings for poor coverage
    if coverage['lat_range'] < min_lat_range:
        print(f"⚠️  Warning: Low latitude coverage ({coverage['lat_range']:.1f}°)")
    if coverage['lon_range'] < min_lon_range:
        print(f"⚠️  Warning: Low longitude coverage ({coverage['lon_range']:.1f}°)")
    
    return coverage


# ==============================================================================
# MATHEMATICAL VALIDATION OF FITTING
# ==============================================================================

def validate_dipole_fit(model: 'DipoleModel', 
                       data: MagneticData,
                       indices: np.ndarray) -> Dict[str, any]:
    """
    Validate that fitted dipole parameters are physically reasonable.
    
    Checks:
    1. Dipole moment magnitude is reasonable (~8e22 A⋅m² for Earth)
    2. Dipole position is near Earth's center (within ~500 km)
    3. Residuals are not systematically biased
    """
    validation = {}
    
    # Check dipole moment magnitude
    m_magnitude = np.linalg.norm(model.dipole_moment)
    validation['moment_magnitude'] = m_magnitude
    validation['moment_reasonable'] = 1e22 < m_magnitude < 1e23
    
    # Check dipole position (should be near Earth's center)
    r0_magnitude = np.linalg.norm(model.dipole_position)
    validation['position_offset_km'] = r0_magnitude / 1e3
    validation['position_reasonable'] = r0_magnitude < 500e3  # Within 500 km
    
    # Check for systematic bias in residuals
    B_pred = model.predict(data.r_cart[indices])
    B_obs = data.B_cart[indices]
    residuals = B_obs - B_pred
    
    mean_residual = np.mean(residuals, axis=0)
    mean_residual_magnitude = np.linalg.norm(mean_residual)
    
    validation['mean_residual_nT'] = mean_residual_magnitude * TESLA_TO_NANOTESLA
    validation['unbiased'] = mean_residual_magnitude < 1e-9  # < 1 nT bias
    
    # Check residual distribution (should be roughly normal)
    residual_magnitudes = np.linalg.norm(residuals, axis=1)
    validation['residual_std_nT'] = np.std(residual_magnitudes) * TESLA_TO_NANOTESLA
    
    return validation


# ==============================================================================
# ANALYSIS: EXPECTED SAMPLE SIZE REQUIREMENTS
# ==============================================================================

def theoretical_sample_size_estimate(n_parameters: int = 6,
                                    measurements_per_point: int = 3) -> Dict[str, int]:
    """
    Estimate theoretical minimum sample sizes for dipole fitting.
    
    A dipole model has 6 parameters: m_x, m_y, m_z, r0_x, r0_y, r0_z
    Each measurement provides 3 constraints: B_x, B_y, B_z
    
    Theoretical minimums:
    1. Determined system: n_points ≥ n_parameters / measurements_per_point = 2
    2. Overdetermined (recommended): n_points ≥ 10 * n_parameters = 60
    3. Statistical stability: n_points ≥ 100-1000 (rule of thumb)
    
    In practice, we expect convergence around 1000-5000 samples based on:
    - Noise in real data
    - Nonlinear optimization challenges
    - Spatial coverage requirements
    """
    return {
        'theoretical_minimum': n_parameters // measurements_per_point,
        'overdetermined_minimum': 10 * n_parameters,
        'practical_minimum': 100,
        'expected_convergence': 1000,
        'high_confidence': 5000
    }


# ==============================================================================
# DATA LOADER
# ==============================================================================

class SwarmDataLoader:
    """Load and preprocess Swarm satellite magnetic field data."""
    
    @staticmethod
    def load_from_cdf(filepath: str) -> MagneticData:
        cdf = cdflib.CDF(filepath)
        info = cdf.cdf_info()
        
        variables = info.zVariables if info.zVariables else info.rVariables
        target_vars = [v for v in variables 
                      if any(x in v for x in ['Radius', 'Lat', 'Long', 'B_NEC'])]
        
        data_dict = {var: cdf.varget(var) for var in target_vars}
        
        radius = np.array(data_dict['Radius'])
        latitude = np.array(data_dict['Latitude'])
        longitude = np.array(data_dict['Longitude'])
        B_NEC = np.array(data_dict['B_NEC']) * NANOTESLA_TO_TESLA
        
        # ✅ Use physics functions directly
        B_cart = nec_to_cart(B_NEC, latitude, longitude)
        r_cart = sph_to_cart(radius, latitude, longitude)
        
        return MagneticData(
            radius=radius,
            latitude=latitude,
            longitude=longitude,
            B_NEC=B_NEC,
            B_cart=B_cart,
            r_cart=r_cart
        )


# ==============================================================================
# EXPERIMENT RUNNER
# ==============================================================================

class ExperimentRunner:
    """Run dipole fitting experiments with different sampling strategies."""
    
    def __init__(self, data: MagneticData, test_fraction: float = 0.2):
        """
        Args:
            data: Full magnetic field dataset
            test_fraction: Fraction of data to reserve for testing
        """
        self.data = data
        self.test_fraction = test_fraction
        
        self._create_train_test_split()
        
        self.model = DipoleModel()
        self.stats_calculator = StatisticsCalculator()
        self.strategies = {
            'random': RandomSampling(),
            'orbital': OrbitSampling()
        }

    
    def _create_train_test_split(self):
        """Create fixed train/test split with no data leakage."""
        total_samples = len(self.data)
        num_test = int(total_samples * self.test_fraction)
        
        all_indices = np.arange(total_samples)
        self.test_indices = np.random.choice(total_samples, num_test, replace=False)
        self.train_pool = np.setdiff1d(all_indices, self.test_indices)
        
        print(f"\n{'='*70}")
        print("TRAIN/TEST SPLIT")
        print(f"{'='*70}")
        print(f"Training pool: {len(self.train_pool):,} samples")
        print(f"Test set: {len(self.test_indices):,} samples")
        print(f"Test fraction: {self.test_fraction:.1%}")
        print(f"{'='*70}\n")
    
    def run(self, sample_sizes: np.ndarray) -> pd.DataFrame:
        """
        Run experiments for all strategies and sample sizes.
        
        Args:
            sample_sizes: Array of sample sizes to test
            
        Returns:
            DataFrame with results
        """
        results = []
        
        print(f"Running experiments: {len(sample_sizes)} sizes × {len(self.strategies)} strategies")
        
        for n_samples in tqdm(sample_sizes, desc="Progress"):
            if n_samples > len(self.train_pool):
                continue
            
            for strategy_name, strategy in self.strategies.items():
                try:
                    train_indices = strategy.sample(self.train_pool, n_samples, self.data)
                    success = self.model.fit(self.data, train_indices)
                    
                    if success:
                        r_test = self.data.r_cart[self.test_indices]
                        B_test = self.data.B_cart[self.test_indices]
                        B_pred = self.model.predict(r_test)
                        
                        stats = self.stats_calculator.calculate(
                            B_test, B_pred, n_samples, strategy_name
                        )
                        results.append(stats.to_dict())
                
                except Exception as e:
                    print(f"\nWarning: Failed for {strategy_name} with n={n_samples}: {e}")
                    continue
        
        return pd.DataFrame(results)


# ==============================================================================
# VISUALIZATION
# ==============================================================================

class ResultVisualizer:
    """Create visualizations of experimental results."""
    
    COLORS = {
    'random': '#2E86AB',
    'orbital': '#A23B72'
    }

    
    KNEE_STYLES = {
        'threshold': {'color': 'red', 'linestyle': '--', 'linewidth': 2, 'alpha': 0.7},
        'curvature': {'color': 'orange', 'linestyle': '-.', 'linewidth': 2, 'alpha': 0.7},
        'kneedle': {'color': 'green', 'linestyle': ':', 'linewidth': 2, 'alpha': 0.7},
        'practical': {'color': 'purple', 'linestyle': '-', 'linewidth': 2.5, 'alpha': 0.9}
    }
    
    @staticmethod
    def plot_strategy_analysis(df: pd.DataFrame, 
                               strategy: str,
                               knee_rms: KneePoints,
                               knee_mae: KneePoints,
                               threshold_10pc: float,
                               output_path: Optional[str] = None):
        """Create comprehensive analysis plots for a single strategy."""
        df_strat = df[df['strategy'] == strategy].copy()
        
        if len(df_strat) == 0:
            print(f"No data for strategy: {strategy}")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{strategy.upper()} Sampling - Test Set Evaluation', 
                    fontsize=16, fontweight='bold')
        
        color = ResultVisualizer.COLORS.get(strategy, 'blue')
        
        ax = axes[0, 0]
        ax.plot(df_strat['n_samples'], df_strat['rms_total'], 
               'o-', color=color, linewidth=2.5, markersize=7, label=strategy)
        ax.axhline(threshold_10pc, color='gray', linestyle='--', linewidth=1.5,
                  label=f'10% Max |B|: {threshold_10pc:.1f} nT', alpha=0.7)
        ResultVisualizer._add_knee_lines(ax, knee_rms, 'RMS')
        ax.set_xscale('log')
        ax.set_xlabel('Training Samples', fontsize=11)
        ax.set_ylabel('RMS Error (nT)', fontsize=11)
        ax.set_title('RMS Error vs Sample Size')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        ax.plot(df_strat['n_samples'], df_strat['rms_x'], 'o-', label='X', markersize=5)
        ax.plot(df_strat['n_samples'], df_strat['rms_y'], 's-', label='Y', markersize=5)
        ax.plot(df_strat['n_samples'], df_strat['rms_z'], '^-', label='Z', markersize=5)
        ax.set_xscale('log')
        ax.set_xlabel('Training Samples', fontsize=11)
        ax.set_ylabel('RMS Error (nT)', fontsize=11)
        ax.set_title('Component-wise RMS')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        ax.plot(df_strat['n_samples'], df_strat['r_squared'], 
               'o-', color=color, linewidth=2.5, markersize=7)
        ax.set_xscale('log')
        ax.set_xlabel('Training Samples', fontsize=11)
        ax.set_ylabel('R² Score', fontsize=11)
        ax.set_title('Model Fit Quality')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        ax.plot(df_strat['n_samples'], df_strat['mae_total'], 
               'o-', color=color, linewidth=2.5, markersize=7, label=strategy)
        ax.axhline(threshold_10pc, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        ResultVisualizer._add_knee_lines(ax, knee_mae, 'MAE')
        ax.set_xscale('log')
        ax.set_xlabel('Training Samples', fontsize=11)
        ax.set_ylabel('MAE (nT)', fontsize=11)
        ax.set_title('Mean Absolute Error')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_path}")
        
        plt.show()
    
    @staticmethod
    def _add_knee_lines(ax, knees: KneePoints, prefix: str):
        """Add vertical lines for detected knee points."""
        if knees.threshold is not None and not np.isnan(knees.threshold):
            style = ResultVisualizer.KNEE_STYLES['threshold']
            ax.axvline(knees.threshold, label=f'{prefix} Threshold: {int(knees.threshold)}', **style)
        
        if knees.curvature is not None and not np.isnan(knees.curvature):
            style = ResultVisualizer.KNEE_STYLES['curvature']
            ax.axvline(knees.curvature, label=f'{prefix} Curvature: {int(knees.curvature)}', **style)
        
        if knees.kneedle is not None and not np.isnan(knees.kneedle):
            style = ResultVisualizer.KNEE_STYLES['kneedle']
            ax.axvline(knees.kneedle, label=f'{prefix} Kneedle: {int(knees.kneedle)}', **style)
        
        if knees.practical is not None and not np.isnan(knees.practical):
            style = ResultVisualizer.KNEE_STYLES['practical']
            ax.axvline(knees.practical, label=f'{prefix} Practical: {int(knees.practical)}', **style)
    
    @staticmethod
    def plot_comparison(df: pd.DataFrame, 
                       threshold_10pc: float,
                       output_path: Optional[str] = None):
        """Create comparison plots for all strategies."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Comparison: Random vs Clustered Sampling', 
                    fontsize=16, fontweight='bold')
        
        for strategy in ['random', 'orbital']:
            df_strat = df[df['strategy'] == strategy]
            if len(df_strat) == 0:
                continue
            
            color = ResultVisualizer.COLORS.get(strategy, 'blue')
            
            axes[0, 0].plot(df_strat['n_samples'], df_strat['rms_total'],
                          'o-', color=color, linewidth=2.5, markersize=7, label=strategy)
            axes[0, 1].plot(df_strat['n_samples'], df_strat['r_squared'],
                          'o-', color=color, linewidth=2.5, markersize=7, label=strategy)
            axes[1, 0].plot(df_strat['n_samples'], df_strat['mae_total'],
                          'o-', color=color, linewidth=2.5, markersize=7, label=strategy)
            axes[1, 1].plot(df_strat['n_samples'], df_strat['relative_error'],
                          'o-', color=color, linewidth=2.5, markersize=7, label=strategy)

        
        axes[0, 0].axhline(threshold_10pc, color='gray', linestyle='--', 
                          linewidth=1.5, label=f'10% Max |B|: {threshold_10pc:.1f} nT', alpha=0.7)
        axes[1, 0].axhline(threshold_10pc, color='gray', linestyle='--', 
                          linewidth=1.5, alpha=0.7)
        
        for ax, ylabel, title in zip(
            axes.flat,
            ['RMS Error (nT)', 'R² Score', 'MAE (nT)', 'Relative Error (%)'],
            ['RMS Error', 'R² Score', 'Mean Absolute Error', 'Relative Error']
        ):
            ax.set_xscale('log')
            ax.set_xlabel('Training Samples', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_path}")
        
        plt.show()


# ==============================================================================
# REPORT GENERATOR
# ==============================================================================

class ReportGenerator:
    """Generate text reports of experimental results."""
    
    @staticmethod
    def print_data_summary(data: MagneticData):
        """Print summary of loaded data."""
        print(f"\n{'='*70}")
        print("DATA SUMMARY")
        print(f"{'='*70}")
        print(f"Total samples: {len(data):,}")
        print(f"Altitude range: {(data.radius.min()-6371e3)/1e3:.0f} - {(data.radius.max()-6371e3)/1e3:.0f} km")
        print(f"Latitude range: {data.latitude.min():.1f}° to {data.latitude.max():.1f}°")
        print(f"Longitude range: {data.longitude.min():.1f}° to {data.longitude.max():.1f}°")
        print(f"Max |B| observed: {data.max_field_magnitude:.2f} nT")
        print(f"10% threshold: {0.1 * data.max_field_magnitude:.2f} nT")
        print(f"{'='*70}\n")
    
    @staticmethod
    def print_strategy_results(df: pd.DataFrame, 
                              strategy: str,
                              knee_rms: KneePoints,
                              knee_mae: KneePoints,
                              threshold_10pc: float):
        """Print detailed results for a strategy."""
        df_strat = df[df['strategy'] == strategy]
        
        if len(df_strat) == 0:
            return
        
        print(f"\n{'='*70}")
        print(f"{strategy.upper()} SAMPLING RESULTS")
        print(f"{'='*70}")
        
        print(f"\nError Statistics:")
        print(f"  Mean RMS:  {df_strat['rms_total'].mean():.2f} nT")
        print(f"  Best RMS:  {df_strat['rms_total'].min():.2f} nT "
              f"(n={df_strat.loc[df_strat['rms_total'].idxmin(), 'n_samples']:.0f})")
        print(f"  Mean MAE:  {df_strat['mae_total'].mean():.2f} nT")
        print(f"  Best MAE:  {df_strat['mae_total'].min():.2f} nT "
              f"(n={df_strat.loc[df_strat['mae_total'].idxmin(), 'n_samples']:.0f})")
        print(f"  Mean R²:   {df_strat['r_squared'].mean():.6f}")
        print(f"  Best R²:   {df_strat['r_squared'].max():.6f}")
        
        print(f"\nRMS-based Knee Points (must be < {threshold_10pc:.1f} nT):")
        ReportGenerator._print_knee_value("Threshold", knee_rms.threshold, threshold_10pc)
        ReportGenerator._print_knee_value("Curvature", knee_rms.curvature, threshold_10pc)
        ReportGenerator._print_knee_value("Kneedle", knee_rms.kneedle, threshold_10pc)
        ReportGenerator._print_knee_value("Practical (95%)", knee_rms.practical, threshold_10pc)
        
        print(f"\nMAE-based Knee Points (must be < {threshold_10pc:.1f} nT):")
        ReportGenerator._print_knee_value("Threshold", knee_mae.threshold, threshold_10pc)
        ReportGenerator._print_knee_value("Curvature", knee_mae.curvature, threshold_10pc)
        ReportGenerator._print_knee_value("Kneedle", knee_mae.kneedle, threshold_10pc)
        ReportGenerator._print_knee_value("Practical (95%)", knee_mae.practical, threshold_10pc)
    
    @staticmethod
    def _print_knee_value(name: str, value: Optional[float], threshold: float):
        """Print a single knee point value."""
        if value is not None and not np.isnan(value):
            print(f"  {name:20s}: {int(value):,} samples")
        else:
            print(f"  {name:20s}: NOT FOUND (error must be < {threshold:.1f} nT)")
    
    @staticmethod
    def print_final_summary():
        """Print final summary of analysis."""
        print(f"\n{'='*70}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print("✓ All evaluations performed on fixed test set")
        print("✓ Zero data leakage - completely unbiased assessment")
        print("✓ Knee validation: RMS and MAE must be < 10% of max |B|")
        print("✓ Practical knee: 95% of total possible improvement")
        print("✓ Physics calculations use dipole_physics module")
        print(f"{'='*70}\n")


# ==============================================================================
# MAIN ANALYSIS PIPELINE
# ==============================================================================

def main():
    """Main analysis pipeline."""
    
    print(f"\n{'='*70}")
    print("DIPOLE FITTING ANALYSIS")
    print("Robust Train/Test Split with Knee Point Detection")
    print("Using dipole_physics module for core calculations")
    print(f"{'='*70}\n")
    
    print("Loading data...")
    cdf_file = "D:\Study\Masters\Sem-3\EPM\Project\Phase2\SW_PREL_MAGA_LR_1B_20181212T000000_20181212T235959_0603_MDR_MAG_LR.cdf"
    
    if not Path(cdf_file).exists():
        print(f"Error: File not found: {cdf_file}")
        return
    
    data = SwarmDataLoader.load_from_cdf(cdf_file)
    ReportGenerator.print_data_summary(data)
    
    runner = ExperimentRunner(data, test_fraction=0.2)
    
    max_samples = min(60000, len(runner.train_pool))
    sample_sizes = np.unique(
        np.logspace(np.log10(6), np.log10(max_samples), 25).astype(int)
    )
    
    print(f"Testing {len(sample_sizes)} sample sizes: {sample_sizes[0]} to {sample_sizes[-1]}")
    
    results_df = runner.run(sample_sizes)
    
    if len(results_df) == 0:
        print("Error: No successful fits")
        return
    
    detector = KneeDetector(validation_threshold=0.1)
    max_field = data.max_field_magnitude
    threshold_10pc = 0.1 * max_field
    
    for strategy in ['random', 'orbital']:
        df_strat = results_df[results_df['strategy'] == strategy]
        
        if len(df_strat) == 0:
            continue
        
        knee_rms = detector.detect_all(df_strat, max_field, metric='rms_total')
        knee_mae = detector.detect_all(df_strat, max_field, metric='mae_total')
        
        ReportGenerator.print_strategy_results(
            results_df, strategy, knee_rms, knee_mae, threshold_10pc
        )
        
        ResultVisualizer.plot_strategy_analysis(
            results_df, strategy, knee_rms, knee_mae, threshold_10pc,
            output_path=f'{strategy}_analysis.png'
        )

    
    ResultVisualizer.plot_comparison(
        results_df, threshold_10pc, output_path='comparison.png'
    )
    
    results_df.to_csv('results.csv', index=False)
    print("Saved results to: results.csv")
    
    ReportGenerator.print_final_summary()


if __name__ == "__main__":
    # Print theoretical expectations
    print("Theoretical Sample Size Requirements:")
    print("=" * 50)
    estimates = theoretical_sample_size_estimate()
    for key, value in estimates.items():
        print(f"{key:30s}: {value:6d} samples")
    
    print("\n" + "=" * 50)
    print("Key Mathematical Properties:")
    print("=" * 50)
    print("""
    Random Sampling:
    - Unbiased estimator of population parameters
    - Variance ∝ 1/n (decreases with sample size)
    - Good for i.i.d. assumptions
    
    Orbital Sampling:
    - Preserves temporal autocorrelation
    - May violate i.i.d. assumption
    - Better for time-series analysis
    - Risk: poor generalization if orbit segment is atypical
    
    Expected Behavior:
    - Random should converge faster (better coverage)
    - Orbital may need more samples due to clustering
    - Both should converge to similar final error
    - Knee point expected around 1000-5000 samples
    """)
    # THEN run the actual analysis
    main()