"""
Dipole Magnetic Field Fitting Analysis
======================================
A robust framework for fitting dipole models to satellite magnetic field data
with proper train/test splits and comprehensive knee point detection.

Author: Refactored for clarity and maintainability
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

warnings.filterwarnings('ignore')

# ==============================================================================
# CONSTANTS
# ==============================================================================

MU0_OVER_4PI = 1e-7  # Magnetic constant: T·m/A
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
# COORDINATE TRANSFORMATIONS
# ==============================================================================

class CoordinateTransformer:
    """Handles coordinate system transformations."""
    
    @staticmethod
    def spherical_to_cartesian(radius: np.ndarray, 
                               latitude: np.ndarray, 
                               longitude: np.ndarray) -> np.ndarray:
        """
        Convert spherical to Cartesian coordinates.
        
        Args:
            radius: Radial distance from Earth's center (m)
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            
        Returns:
            Cartesian coordinates (x, y, z) in meters
        """
        lat_rad = np.deg2rad(latitude)
        lon_rad = np.deg2rad(longitude)
        
        x = radius * np.cos(lat_rad) * np.cos(lon_rad)
        y = radius * np.cos(lat_rad) * np.sin(lon_rad)
        z = radius * np.sin(lat_rad)
        
        return np.column_stack([x, y, z])
    
    @staticmethod
    def nec_to_cartesian(B_nec: np.ndarray, 
                        latitude: np.ndarray, 
                        longitude: np.ndarray) -> np.ndarray:
        """
        Transform magnetic field from NEC (North-East-Center) to Cartesian.
        
        Args:
            B_nec: Magnetic field in NEC coordinates (N x 3)
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            
        Returns:
            Magnetic field in Cartesian coordinates (N x 3)
        """
        lat_rad = np.deg2rad(latitude)
        lon_rad = np.deg2rad(longitude)
        B_cart = np.zeros_like(B_nec)
        
        for i in range(len(latitude)):
            cos_lat, sin_lat = np.cos(lat_rad[i]), np.sin(lat_rad[i])
            cos_lon, sin_lon = np.cos(lon_rad[i]), np.sin(lon_rad[i])
            
            # NEC basis vectors in Cartesian coordinates
            north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
            east = np.array([-sin_lon, cos_lon, 0])
            center = np.array([-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat])
            
            B_cart[i] = (B_nec[i, 0] * north + 
                        B_nec[i, 1] * east + 
                        B_nec[i, 2] * center)
        
        return B_cart


# ==============================================================================
# DIPOLE MODEL
# ==============================================================================

class DipoleModel:
    """Magnetic dipole field model."""
    
    def __init__(self):
        self.dipole_moment: Optional[np.ndarray] = None
        self.dipole_position: Optional[np.ndarray] = None
    
    @staticmethod
    def calculate_field(r: np.ndarray, 
                       dipole_moment: np.ndarray, 
                       dipole_position: np.ndarray) -> np.ndarray:
        """
        Calculate magnetic field from a dipole source.
        
        B = (μ₀/4π) * [3(m·r̂)r̂ - m] / r³
        
        Args:
            r: Position vectors where field is calculated (N x 3)
            dipole_moment: Magnetic dipole moment vector (1 x 3)
            dipole_position: Position of dipole center (1 x 3)
            
        Returns:
            Magnetic field vectors (N x 3)
        """
        m = np.atleast_2d(dipole_moment)
        r0 = np.atleast_2d(dipole_position)
        
        # Vector from dipole to field point
        R = r - r0
        R_magnitude = np.linalg.norm(R, axis=1, keepdims=True)
        R_hat = R / R_magnitude
        
        # Dipole field formula
        dot_product = np.sum(m * R_hat, axis=1, keepdims=True)
        B = MU0_OVER_4PI * (3 * dot_product * R_hat - m) / (R_magnitude ** 3)
        
        return B
    
    def fit(self, data: MagneticData, indices: np.ndarray) -> bool:
        """
        Fit dipole model to data subset.
        
        Args:
            data: Full magnetic data
            indices: Indices of data points to use for fitting
            
        Returns:
            Success status of the fit
        """
        # Initial guess for parameters
        initial_params = np.array([8e22, 8e22, 8e22, 1000, 1000, 1000])
        
        def residuals(params):
            m = params[:3]
            r0 = params[3:]
            r_subset = data.r_cart[indices]
            B_model = self.calculate_field(r_subset, m, r0)
            B_obs = data.B_cart[indices]
            return (B_obs - B_model).flatten()
        
        result = least_squares(
            residuals,
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
        """Predict magnetic field at given positions."""
        if self.dipole_moment is None or self.dipole_position is None:
            raise ValueError("Model must be fitted before prediction")
        
        return self.calculate_field(r, self.dipole_moment, self.dipole_position)


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
        """
        Calculate comprehensive statistics for model fit.
        
        Args:
            B_observed: Observed magnetic field (N x 3)
            B_predicted: Predicted magnetic field (N x 3)
            n_samples: Number of training samples used
            strategy: Sampling strategy name
            
        Returns:
            FitStatistics object with all metrics
        """
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
        
        # Calculate curvature via second derivative
        first_deriv = np.gradient(log_values, log_n)
        second_deriv = np.gradient(first_deriv, log_n)
        
        # Use interior points to avoid edge effects
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
        
        # Normalize to [0, 1]
        x_norm = (x - x.min()) / (x.max() - x.min())
        y_norm = (y - y.min()) / (y.max() - y.min())
        
        # For decreasing curves, flip y
        y_norm = 1 - y_norm
        
        # Find where normalized curve exceeds diagonal by threshold
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
        """Find point where cumulative improvement reaches threshold of total."""
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
    """Base class for sampling strategies."""
    
    def sample(self, pool_indices: np.ndarray, 
               n_samples: int, 
               data: MagneticData) -> np.ndarray:
        """Sample n_samples from pool_indices."""
        raise NotImplementedError


class RandomSampling(SamplingStrategy):
    """Random sampling from the data pool."""
    
    def sample(self, pool_indices: np.ndarray, 
               n_samples: int, 
               data: MagneticData) -> np.ndarray:
        return np.random.choice(pool_indices, n_samples, replace=False)


class ClusteredSampling(SamplingStrategy):
    """Spatially clustered sampling."""
    
    def sample(self, pool_indices: np.ndarray, 
               n_samples: int, 
               data: MagneticData) -> np.ndarray:
        # Select random center point
        center_idx_local = np.random.choice(len(pool_indices))
        center_idx = pool_indices[center_idx_local]
        
        # Calculate distances in lat/lon space
        distances = np.sqrt(
            (data.latitude[pool_indices] - data.latitude[center_idx]) ** 2 +
            (data.longitude[pool_indices] - data.longitude[center_idx]) ** 2
        )
        
        # Select nearest points
        nearest_indices = np.argsort(distances)[:n_samples]
        return pool_indices[nearest_indices]


# ==============================================================================
# DATA LOADER
# ==============================================================================

class SwarmDataLoader:
    """Load and preprocess Swarm satellite magnetic field data."""
    
    @staticmethod
    def load_from_cdf(filepath: str) -> MagneticData:
        """
        Load data from CDF file.
        
        Args:
            filepath: Path to CDF file
            
        Returns:
            MagneticData object with loaded data
        """
        cdf = cdflib.CDF(filepath)
        info = cdf.cdf_info()
        
        # Find relevant variables
        variables = info.zVariables if info.zVariables else info.rVariables
        target_vars = [v for v in variables 
                      if any(x in v for x in ['Radius', 'Lat', 'Long', 'B_NEC'])]
        
        # Load data
        data_dict = {var: cdf.varget(var) for var in target_vars}
        
        radius = np.array(data_dict['Radius'])
        latitude = np.array(data_dict['Latitude'])
        longitude = np.array(data_dict['Longitude'])
        B_NEC = np.array(data_dict['B_NEC']) * NANOTESLA_TO_TESLA
        
        # Transform coordinates
        transformer = CoordinateTransformer()
        B_cart = transformer.nec_to_cartesian(B_NEC, latitude, longitude)
        r_cart = transformer.spherical_to_cartesian(radius, latitude, longitude)
        
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
        
        # Create fixed train/test split
        self._create_train_test_split()
        
        # Initialize components
        self.model = DipoleModel()
        self.stats_calculator = StatisticsCalculator()
        self.strategies = {
            'random': RandomSampling(),
            'clustered': ClusteredSampling()
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
                    # Sample training data
                    train_indices = strategy.sample(self.train_pool, n_samples, self.data)
                    
                    # Fit model
                    success = self.model.fit(self.data, train_indices)
                    
                    if success:
                        # Evaluate on test set
                        r_test = self.data.r_cart[self.test_indices]
                        B_test = self.data.B_cart[self.test_indices]
                        B_pred = self.model.predict(r_test)
                        
                        # Calculate statistics
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
        'clustered': '#A23B72'
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
        
        # Plot 1: RMS Error with knee points
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
        
        # Plot 2: Component-wise RMS
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
        
        # Plot 3: R² Score
        ax = axes[1, 0]
        ax.plot(df_strat['n_samples'], df_strat['r_squared'], 
               'o-', color=color, linewidth=2.5, markersize=7)
        ax.set_xscale('log')
        ax.set_xlabel('Training Samples', fontsize=11)
        ax.set_ylabel('R² Score', fontsize=11)
        ax.set_title('Model Fit Quality')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: MAE with knee points
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
        
        for strategy in ['random', 'clustered']:
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
        
        # Add threshold lines
        axes[0, 0].axhline(threshold_10pc, color='gray', linestyle='--', 
                          linewidth=1.5, label=f'10% Max |B|: {threshold_10pc:.1f} nT', alpha=0.7)
        axes[1, 0].axhline(threshold_10pc, color='gray', linestyle='--', 
                          linewidth=1.5, alpha=0.7)
        
        # Configure axes
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
        
        # Summary statistics
        print(f"\nError Statistics:")
        print(f"  Mean RMS:  {df_strat['rms_total'].mean():.2f} nT")
        print(f"  Best RMS:  {df_strat['rms_total'].min():.2f} nT "
              f"(n={df_strat.loc[df_strat['rms_total'].idxmin(), 'n_samples']:.0f})")
        print(f"  Mean MAE:  {df_strat['mae_total'].mean():.2f} nT")
        print(f"  Best MAE:  {df_strat['mae_total'].min():.2f} nT "
              f"(n={df_strat.loc[df_strat['mae_total'].idxmin(), 'n_samples']:.0f})")
        print(f"  Mean R²:   {df_strat['r_squared'].mean():.6f}")
        print(f"  Best R²:   {df_strat['r_squared'].max():.6f}")
        
        # RMS-based knees
        print(f"\nRMS-based Knee Points (must be < {threshold_10pc:.1f} nT):")
        ReportGenerator._print_knee_value("Threshold", knee_rms.threshold, threshold_10pc)
        ReportGenerator._print_knee_value("Curvature", knee_rms.curvature, threshold_10pc)
        ReportGenerator._print_knee_value("Kneedle", knee_rms.kneedle, threshold_10pc)
        ReportGenerator._print_knee_value("Practical (95%)", knee_rms.practical, threshold_10pc)
        
        # MAE-based knees
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
        print(f"{'='*70}\n")


# ==============================================================================
# MAIN ANALYSIS PIPELINE
# ==============================================================================

def main():
    """Main analysis pipeline."""
    
    print(f"\n{'='*70}")
    print("DIPOLE FITTING ANALYSIS")
    print("Robust Train/Test Split with Knee Point Detection")
    print(f"{'='*70}\n")
    
    # 1. Load data
    print("Loading data...")
    cdf_file = "SW_PREL_MAGA_LR_1B_20181212T000000_20181212T235959_0603_MDR_MAG_LR.cdf"
    
    if not Path(cdf_file).exists():
        print(f"Error: File not found: {cdf_file}")
        return
    
    data = SwarmDataLoader.load_from_cdf(cdf_file)
    ReportGenerator.print_data_summary(data)
    
    # 2. Set up experiment
    runner = ExperimentRunner(data, test_fraction=0.2)
    
    # 3. Define sample sizes (logarithmic spacing)
    max_samples = min(60000, len(runner.train_pool))
    sample_sizes = np.unique(
        np.logspace(np.log10(6), np.log10(max_samples), 25).astype(int)
    )
    
    print(f"Testing {len(sample_sizes)} sample sizes: {sample_sizes[0]} to {sample_sizes[-1]}")
    
    # 4. Run experiments
    results_df = runner.run(sample_sizes)
    
    if len(results_df) == 0:
        print("Error: No successful fits")
        return
    
    # 5. Detect knee points
    detector = KneeDetector(validation_threshold=0.1)
    max_field = data.max_field_magnitude
    threshold_10pc = 0.1 * max_field
    
    # 6. Generate reports and visualizations
    for strategy in ['random', 'clustered']:
        df_strat = results_df[results_df['strategy'] == strategy]
        
        if len(df_strat) == 0:
            continue
        
        # Detect knees
        knee_rms = detector.detect_all(df_strat, max_field, metric='rms_total')
        knee_mae = detector.detect_all(df_strat, max_field, metric='mae_total')
        
        # Print results
        ReportGenerator.print_strategy_results(
            results_df, strategy, knee_rms, knee_mae, threshold_10pc
        )
        
        # Create plots
        ResultVisualizer.plot_strategy_analysis(
            results_df, strategy, knee_rms, knee_mae, threshold_10pc,
            output_path=f'{strategy}_analysis.png'
        )
    
    # 7. Comparison plot
    ResultVisualizer.plot_comparison(
        results_df, threshold_10pc, output_path='comparison.png'
    )
    
    # 8. Save results
    results_df.to_csv('results.csv', index=False)
    print("Saved results to: results.csv")
    
    # 9. Final summary
    ReportGenerator.print_final_summary()


if __name__ == "__main__":
    main()