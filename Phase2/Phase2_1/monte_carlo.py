"""
Dipole Model Validation Framework - CORRECTED
==============================================
Fixed to ensure proper train/test split with no data leakage:
- Global test set (20%) reserved at initialization
- Training samples drawn only from remaining 80%
- Test evaluation always on the global holdout set

MODIFIED: Now stores full moment and position vectors (all 3 components)
FIXED: Uses correct attribute names (model.dipole_moment, model.dipole_position)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import List, Dict, Optional
from sklearn.model_selection import KFold
from tqdm import tqdm

# =============================================================================
# IMPORT FROM IMPROVED fitting_analysis MODULE
# =============================================================================

from fitting_analysis import (
    MagneticData,
    DipoleModel,
    StatisticsCalculator,
    SwarmDataLoader,
    RandomSampling,
    OrbitSampling,
    StratifiedSpatialSampling,
    validate_train_test_split,
    validate_sampling_coverage,
    validate_dipole_fit,
    TESLA_TO_NANOTESLA
)

from dipole_model import (
    setup_global_map,
    plot_continuous_field_on_earth,
    plot_scatter_field_on_earth,
    create_global_grid
)

from dipole_physics import (
    sph_to_cart,
    nec_to_cart,
    cart_to_nec,
    cart_to_nec_grid,
    dipole_field,
    cart_to_sph_direction
)

# =============================================================================
# VALIDATION RESULT CONTAINER
# =============================================================================

@dataclass
class ValidationResult:
    """Container for single validation run results."""
    strategy: str
    sample_size: int
    run: int
    fold: int
    rms: float
    mae: float
    r2: float
    # Store full moment vector components
    moment_x: Optional[float] = None
    moment_y: Optional[float] = None
    moment_z: Optional[float] = None
    moment_magnitude: Optional[float] = None
    # Store full position vector components
    position_x_km: Optional[float] = None
    position_y_km: Optional[float] = None
    position_z_km: Optional[float] = None
    position_offset_km: Optional[float] = None
    mean_residual_nT: Optional[float] = None


# =============================================================================
# ENHANCED VALIDATOR WITH PROPER TRAIN/TEST SPLIT
# =============================================================================

class DipoleValidator:
    """
    Run comprehensive validation with multiple strategies and resampling.

    CRITICAL FIX: Test set is reserved globally at initialization.
    Training samples are ONLY drawn from the remaining data pool.
    """

    def __init__(
        self,
        data: MagneticData,
        sample_sizes: List[int],
        n_mc: int = 100,
        k_folds: int = 5,
        test_fraction: float = 0.2,
        seed: int = 42,
        enable_validation_checks: bool = True
    ):
        """
        Args:
            data: Magnetic field data
            sample_sizes: List of sample sizes to test
            n_mc: Number of Monte Carlo runs per (strategy, size) pair
            k_folds: Number of CV folds (applied to training set only)
            test_fraction: Fraction reserved for testing (0.2 = 20%)
            seed: Random seed for reproducibility
            enable_validation_checks: Enable physical validation checks
        """
        self.data = data
        self.sample_sizes = sample_sizes
        self.n_mc = n_mc
        self.k_folds = k_folds
        self.test_fraction = test_fraction
        self.enable_validation_checks = enable_validation_checks

        np.random.seed(seed)

        # =================================================================
        # CRITICAL: RESERVE GLOBAL TEST SET AT INITIALIZATION
        # =================================================================
        all_indices = np.arange(len(self.data))
        num_global_test = int(len(self.data) * test_fraction)

        # Randomly select test indices
        self.global_test_indices = np.random.choice(
            all_indices, num_global_test, replace=False
        )

        # Remaining indices form the training pool
        self.train_pool = np.setdiff1d(all_indices, self.global_test_indices)

        # Use improved sampling strategies
        self.strategies = {
            "random": RandomSampling(),
            "orbital": OrbitSampling(),
            "stratified": StratifiedSpatialSampling(n_lat_bins=10, n_lon_bins=10)
        }

        print(f"\n{'='*70}")
        print("DIPOLE VALIDATION FRAMEWORK - CORRECTED")
        print(f"{'='*70}")
        print(f"Total data points: {len(data):,}")
        print(f"Global test set: {len(self.global_test_indices):,} ({test_fraction*100:.0f}%)")
        print(f"Training pool: {len(self.train_pool):,} ({(1-test_fraction)*100:.0f}%)")
        print(f"\nStrategies: {list(self.strategies.keys())}")
        print(f"Sample sizes: {len(sample_sizes)} from {min(sample_sizes)} to {max(sample_sizes)}")
        print(f"Monte Carlo runs: {n_mc}")
        print(f"CV folds: {k_folds}")
        print(f"Physical validation: {'ON' if enable_validation_checks else 'OFF'}")
        print(f"{'='*70}\n")

    def run(self) -> pd.DataFrame:
        """
        Run full validation experiment.

        Returns:
            DataFrame with all validation results
        """
        results: List[Dict] = []

        total_iterations = len(self.strategies) * len(self.sample_sizes) * self.n_mc
        print(f"Total iterations: {total_iterations:,}\n")

        for strategy_name, strategy in self.strategies.items():
            print(f"\n{'='*70}")
            print(f"Strategy: {strategy_name.upper()}")
            print(f"{'='*70}")

            for N in self.sample_sizes:
                print(f"\n  Sample size: {N}")

                # Skip if too small or too large for training pool
                if N < 10:
                    print(f"    ⚠️  Skipping (too small, need ≥10)")
                    continue

                if N > len(self.train_pool):
                    print(f"    ⚠️  Skipping (exceeds training pool size: {len(self.train_pool)})")
                    continue

                successful_runs = 0
                failed_runs = 0

                for mc in tqdm(range(self.n_mc), desc=f"  MC runs"):
                    try:
                        # =================================================================
                        # KEY FIX: Sample ONLY from training pool (not entire dataset)
                        # =================================================================
                        train_idx = strategy.sample(self.train_pool, N, self.data)

                        # Verify no overlap with test set
                        assert len(np.intersect1d(train_idx, self.global_test_indices)) == 0, \
                            "Data leakage detected! Training set overlaps with test set!"

                        # Optional: Check spatial coverage
                        if self.enable_validation_checks and N >= 50:
                            coverage = validate_sampling_coverage(
                                train_idx, self.data,
                                min_lat_range=20.0, min_lon_range=40.0
                            )

                        # -----------------------------
                        # k-FOLD CV (TRAIN ONLY)
                        # -----------------------------
                        k_eff = min(self.k_folds, len(train_idx))
                        if k_eff < 2:
                            k_eff = 1  # Just use all training data

                        if k_eff > 1:
                            kf = KFold(n_splits=k_eff, shuffle=True, random_state=None)
                            splits = list(kf.split(train_idx))
                        else:
                            # No CV, use all training data
                            splits = [(np.arange(len(train_idx)), [])]

                        for fold, (tr, _) in enumerate(splits):
                            tr_idx = train_idx[tr]

                            if len(tr_idx) < 6:  # Need minimum samples for 6 parameters
                                failed_runs += 1
                                continue

                            # Fit model on training fold
                            model = DipoleModel()
                            success = model.fit(self.data, tr_idx)

                            if not success:
                                failed_runs += 1
                                continue

                            # =================================================================
                            # FIXED: Extract full moment and position vectors using correct attributes
                            # =================================================================
                            validation_info = {}
                            if self.enable_validation_checks:
                                validation = validate_dipole_fit(
                                    model, self.data, tr_idx
                                )

                                # Extract full moment vector (CORRECTED: model.dipole_moment)
                                moment_vec = model.dipole_moment
                                moment_mag = np.linalg.norm(moment_vec)

                                # Extract full position vector (CORRECTED: model.dipole_position)
                                position_vec = model.dipole_position / 1e3  # Convert to km
                                position_offset = np.linalg.norm(position_vec)

                                validation_info = {
                                    'moment_x': moment_vec[0],
                                    'moment_y': moment_vec[1],
                                    'moment_z': moment_vec[2],
                                    'moment_magnitude': moment_mag,
                                    'position_x_km': position_vec[0],
                                    'position_y_km': position_vec[1],
                                    'position_z_km': position_vec[2],
                                    'position_offset_km': position_offset,
                                    'mean_residual_nT': validation['mean_residual_nT']
                                }

                                # Skip if non-physical
                                if not validation['moment_reasonable']:
                                    failed_runs += 1
                                    continue
                            else:
                                # If validation disabled, still extract moment and position
                                # CORRECTED: Use model.dipole_moment and model.dipole_position
                                moment_vec = model.dipole_moment
                                position_vec = model.dipole_position / 1e3

                                validation_info = {
                                    'moment_x': moment_vec[0],
                                    'moment_y': moment_vec[1],
                                    'moment_z': moment_vec[2],
                                    'moment_magnitude': np.linalg.norm(moment_vec),
                                    'position_x_km': position_vec[0],
                                    'position_y_km': position_vec[1],
                                    'position_z_km': position_vec[2],
                                    'position_offset_km': np.linalg.norm(position_vec),
                                    'mean_residual_nT': None
                                }

                            # =================================================================
                            # KEY FIX: TEST ON GLOBAL HOLDOUT SET (never seen during training)
                            # =================================================================
                            B_obs = self.data.B_cart[self.global_test_indices]
                            r_test = self.data.r_cart[self.global_test_indices]
                            B_pred = model.predict(r_test)

                            stats = StatisticsCalculator.calculate(
                                B_obs,
                                B_pred,
                                n_samples=len(tr_idx),
                                strategy=strategy_name
                            )

                            results.append({
                                "strategy": strategy_name,
                                "sample_size": N,
                                "run": mc,
                                "fold": fold,
                                "rms": stats.rms_total,
                                "mae": stats.mae_total,
                                "r2": stats.r_squared,
                                **validation_info
                            })

                            successful_runs += 1

                    except Exception as e:
                        failed_runs += 1
                        if failed_runs < 3:  # Only print first few errors
                            print(f"\n    ⚠️  Error in run {mc}: {str(e)[:80]}")
                        continue

                print(f"    ✓ Successful: {successful_runs}, Failed: {failed_runs}")

        df = pd.DataFrame(results)

        print(f"\n{'='*70}")
        print("VALIDATION COMPLETE")
        print(f"{'='*70}")
        print(f"Total results collected: {len(df):,}")
        print(f"Strategies evaluated: {df['strategy'].nunique()}")
        print(f"Sample sizes tested: {df['sample_size'].nunique()}")
        print(f"{'='*70}\n")

        return df


# =============================================================================
# VISUALIZATION FUNCTIONS (unchanged)
# =============================================================================

def plot_sampling_examples_3d(data: MagneticData, N: int = 500):
    """Plot one example of each sampling strategy in 3D with Earth."""
    fig = plt.figure(figsize=(18, 6))
    strategies = {
        "Random": RandomSampling(),
        "Orbital": OrbitSampling(),
        "Stratified": StratifiedSpatialSampling(n_lat_bins=10, n_lon_bins=10)
    }

    indices = np.arange(len(data))

    for i, (name, strategy) in enumerate(strategies.items(), 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")

        idx = strategy.sample(indices, N, data)
        r = data.r_cart[idx]

        # Color by altitude
        altitudes = (data.radius[idx] - 6371e3) / 1e3  # km
        scatter = ax.scatter(r[:, 0] / 1e3, r[:, 1] / 1e3, r[:, 2] / 1e3, 
                           c=altitudes, s=3, cmap='viridis', alpha=0.6)

        ax.set_title(f'{name}\n({len(idx)} points)', fontsize=12, fontweight='bold')

        # Earth sphere
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 30)
        R = 6371  # km
        x = R * np.outer(np.cos(u), np.sin(v))
        y = R * np.outer(np.sin(u), np.sin(v))
        z = R * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(x, y, z, color="gray", alpha=0.2, linewidth=0.5)

        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_box_aspect([1, 1, 1])

        plt.colorbar(scatter, ax=ax, label='Altitude (km)', shrink=0.6)

    plt.suptitle('Sampling Strategy Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sampling_strategies_3d.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: sampling_strategies_3d.png")
    plt.show()


def plot_sampling_examples_global(data: MagneticData, N: int = 6000):
    """Plot sampling strategies on global map."""
    strategies = {
        "Random": RandomSampling(),
        "Orbital": OrbitSampling(),
        "Stratified": StratifiedSpatialSampling(n_lat_bins=10, n_lon_bins=10)
    }

    indices = np.arange(len(data))

    for name, strategy in strategies.items():
        idx = strategy.sample(indices, N, data)

        lats = data.latitude[idx]
        lons = data.longitude[idx]

        fig, ax = setup_global_map(f'{name} Sampling Strategy ({N} points)')

        ax.scatter(lons, lats, c='red', s=1, alpha=0.5, 
                  transform=plt.gca().projection.as_geodetic())

        plt.tight_layout()
        plt.savefig(f'sampling_{name.lower()}_global.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: sampling_{name.lower()}_global.png")
        plt.show()


def plot_validation_results(df: pd.DataFrame, output_path: str = 'validation_results.png'):
    """Create comprehensive plots of validation results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Validation Results: Monte Carlo + Cross-Validation', 
                 fontsize=14, fontweight='bold')

    agg = df.groupby(['strategy', 'sample_size']).agg({
        'rms': ['mean', 'std'],
        'mae': ['mean', 'std'],
        'r2': ['mean', 'std']
    }).reset_index()

    colors = {
        'random': '#2E86AB', 
        'orbital': '#A23B72', 
        'stratified': '#F18F01'
    }

    # RMS Error
    ax = axes[0, 0]
    for strategy in agg['strategy'].unique():
        data = agg[agg['strategy'] == strategy]
        x = data['sample_size']
        y_mean = data['rms']['mean']
        y_std = data['rms']['std']

        color = colors.get(strategy, 'gray')
        ax.plot(x, y_mean, 'o-', label=strategy, color=color, linewidth=2)
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2, color=color)

    ax.set_xscale('log')
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('RMS Error (nT)')
    ax.set_title('RMS Error vs Sample Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MAE
    ax = axes[0, 1]
    for strategy in agg['strategy'].unique():
        data = agg[agg['strategy'] == strategy]
        x = data['sample_size']
        y_mean = data['mae']['mean']
        y_std = data['mae']['std']

        color = colors.get(strategy, 'gray')
        ax.plot(x, y_mean, 's-', label=strategy, color=color, linewidth=2)
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2, color=color)

    ax.set_xscale('log')
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('MAE (nT)')
    ax.set_title('Mean Absolute Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # R²
    ax = axes[1, 0]
    for strategy in agg['strategy'].unique():
        data = agg[agg['strategy'] == strategy]
        x = data['sample_size']
        y_mean = data['r2']['mean']
        y_std = data['r2']['std']

        color = colors.get(strategy, 'gray')
        ax.plot(x, y_mean, '^-', label=strategy, color=color, linewidth=2)
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2, color=color)

    ax.set_xscale('log')
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('R² Score')
    ax.set_title('Model Fit Quality')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Variance across runs
    ax = axes[1, 1]
    for strategy in agg['strategy'].unique():
        data = agg[agg['strategy'] == strategy]
        x = data['sample_size']
        y = data['rms']['std']

        color = colors.get(strategy, 'gray')
        ax.plot(x, y, 'o-', label=strategy, color=color, linewidth=2)

    ax.set_xscale('log')
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('RMS Std Dev (nT)')
    ax.set_title('Uncertainty (Variance Across Runs)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main validation pipeline."""

    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)

    data = SwarmDataLoader.load_from_cdf(
        r"D:\Study\Masters\Sem-3\EPM\Project\Phase2\SW_PREL_MAGA_LR_1B_20181212T000000_20181212T235959_0603_MDR_MAG_LR.cdf"
    )

    print(f"Total samples loaded: {len(data):,}")
    print(f"Altitude range: {(data.radius.min()-6371e3)/1e3:.0f} - {(data.radius.max()-6371e3)/1e3:.0f} km")
    print(f"Max |B|: {data.max_field_magnitude:.2f} nT")

    # Visualize sampling strategies
    print("\n" + "="*70)
    print("GENERATING SAMPLING VISUALIZATIONS (3D)")
    print("="*70)
    plot_sampling_examples_3d(data, N=6000)

    print("\n" + "="*70)
    print("GENERATING SAMPLING VISUALIZATIONS (Global Maps)")
    print("="*70)
    plot_sampling_examples_global(data, N=6000)

    # Run validation
    print("\n" + "="*70)
    print("STARTING VALIDATION EXPERIMENT")
    print("="*70)

    validator = DipoleValidator(
        data=data,
        sample_sizes=np.unique(
            np.logspace(np.log10(20), np.log10(10000), 25).astype(int)
        ),
        n_mc=50,
        k_folds=5,
        test_fraction=0.2,  # 20% reserved for testing
        enable_validation_checks=True
    )

    df = validator.run()

    # Save results
    output_file = "validation_results_mc_cv.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")

    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    summary = df.groupby(["strategy", "sample_size"])[["rms", "mae", "r2"]].agg(['mean', 'std'])
    print(summary)

    # Plot results
    print("\n" + "="*70)
    print("GENERATING VALIDATION PLOTS")
    print("="*70)
    plot_validation_results(df)

    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
    print("\nAll visualizations generated:")
    print("  ✓ sampling_strategies_3d.png")
    print("  ✓ sampling_random_global.png")
    print("  ✓ sampling_orbital_global.png")
    print("  ✓ sampling_stratified_global.png")
    print("  ✓ validation_results.png")
    print("  ✓ validation_results_mc_cv.csv")
    print("\nCSV now includes full moment and position vectors (all 3 components)")


if __name__ == "__main__":
    main()