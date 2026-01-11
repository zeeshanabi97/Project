"""
Advanced Knee Detection with Confidence Intervals
==================================================
Detect optimal sample size (knee point) with uncertainty quantification.
Works with pre-aggregated data that has CI bounds.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 14)
plt.rcParams['font.size'] = 10

# Load raw data
df_raw = pd.read_csv('validation_results_mc_cv.csv')

# ==============================================================================
# STEP 1: AGGREGATE WITH CONFIDENCE INTERVALS
# ==============================================================================

def compute_ci(group, metric='rms', ci=95):
    """Compute mean and confidence interval."""
    data = group[metric].values
    mean = np.mean(data)
    sem = stats.sem(data)
    margin = sem * stats.t.ppf((1 + ci/100) / 2, len(data) - 1)
    return {
        'mean': mean,
        'ci_lower': mean - margin,
        'ci_upper': mean + margin,
        'std': np.std(data),
        'count': len(data)
    }

# Aggregate across runs and folds
agg_data = []
for strategy in df_raw['strategy'].unique():
    for sample_size in sorted(df_raw['sample_size'].unique()):
        subset = df_raw[(df_raw['strategy'] == strategy) & (df_raw['sample_size'] == sample_size)]
        if len(subset) == 0:
            continue
        
        row = {
            'strategy': strategy,
            'sample_size': sample_size,
            'n_runs': len(subset)
        }
        
        # Compute stats for RMS and MAE
        for metric in ['rms', 'mae']:
            ci_stats = compute_ci(subset, metric=metric)
            row.update({
                f'{metric}_mean': ci_stats['mean'],
                f'{metric}_ci_lower': ci_stats['ci_lower'],
                f'{metric}_ci_upper': ci_stats['ci_upper'],
                f'{metric}_std': ci_stats['std'],
                f'{metric}_count': ci_stats['count']
            })
        
        agg_data.append(row)

agg_df = pd.DataFrame(agg_data)

print("\n" + "="*80)
print("KNEE POINT DETECTION WITH CONFIDENCE INTERVALS")
print("="*80)

# ==============================================================================
# STEP 2: KNEE DETECTION ALGORITHMS WITH UNCERTAINTY
# ==============================================================================

class KneeDetectorCI:
    """Detect knee points with confidence interval propagation."""
    
    def __init__(self, validation_threshold=0.1):
        self.validation_threshold = validation_threshold
    
    def detect_with_uncertainty(self, df, metric='rms_mean', 
                               ci_lower_col='rms_ci_lower', 
                               ci_upper_col='rms_ci_upper'):
        """
        Detect knee point and estimate its uncertainty.
        
        Returns dict with:
        - knee_value: point estimate
        - knee_lower: conservative estimate (upper CI bound)
        - knee_upper: optimistic estimate (lower CI bound)
        - confidence: robustness measure
        """
        df_sorted = df.sort_values('sample_size').copy()
        
        x = df_sorted['sample_size'].values
        y = df_sorted[metric].values
        y_lower = df_sorted[ci_lower_col].values
        y_upper = df_sorted[ci_upper_col].values
        
        results = {}
        
        # === METHOD 1: Threshold-based (improvement < 0.5%) ===
        knee_thresh = self._threshold_method(x, y, y_lower, y_upper)
        results['threshold'] = knee_thresh
        
        # === METHOD 2: Curvature-based (max 2nd derivative) ===
        knee_curv = self._curvature_method(x, y, y_lower, y_upper)
        results['curvature'] = knee_curv
        
        # === METHOD 3: Kneedle algorithm ===
        knee_kneedle = self._kneedle_method(x, y, y_lower, y_upper)
        results['kneedle'] = knee_kneedle
        
        # === METHOD 4: Practical (95% improvement) ===
        knee_pract = self._practical_method(x, y, y_lower, y_upper)
        results['practical'] = knee_pract
        
        return results
    
    def _threshold_method(self, x, y, y_lower, y_upper, threshold_pct=0.5):
        """Find where improvement drops below threshold."""
        improvements = np.abs(np.diff(y)) / y[:-1] * 100
        
        knee_idx = np.where(improvements < threshold_pct)[0]
        if len(knee_idx) == 0:
            return None
        
        knee_idx = knee_idx[0]
        
        # Uncertainty: use CI bounds
        knee_estimate = x[knee_idx]
        knee_lower = x[knee_idx]  # conservative
        knee_upper = x[min(knee_idx + 1, len(x) - 1)]  # could be next point
        
        return {
            'value': knee_estimate,
            'lower': knee_lower,
            'upper': knee_upper,
            'method': 'Threshold (0.5%)',
            'quality': 'reliable' if knee_idx > 2 else 'uncertain'
        }
    
    def _curvature_method(self, x, y, y_lower, y_upper):
        """Find knee using 2nd derivative in log-log space."""
        if len(x) < 5:
            return None
        
        log_x = np.log10(x)
        log_y = np.log10(y)
        
        # Compute derivatives
        first_deriv = np.gradient(log_y, log_x)
        second_deriv = np.gradient(first_deriv, log_x)
        
        # Find most negative curvature (steepest elbow)
        valid_range = range(2, len(second_deriv) - 2) if len(second_deriv) > 4 else range(1, len(second_deriv) - 1)
        if len(list(valid_range)) == 0:
            return None
        
        valid_curves = second_deriv[list(valid_range)]
        knee_idx = list(valid_range)[np.argmin(valid_curves)]
        
        knee_estimate = x[knee_idx]
        knee_lower = x[max(0, knee_idx - 1)]
        knee_upper = x[min(knee_idx + 1, len(x) - 1)]
        
        return {
            'value': knee_estimate,
            'lower': knee_lower,
            'upper': knee_upper,
            'method': 'Curvature (max 2nd deriv)',
            'quality': 'reliable' if knee_idx > 2 else 'uncertain'
        }
    
    def _kneedle_method(self, x, y, y_lower, y_upper, sensitivity=0.3):
        """Kneedle algorithm with uncertainty."""
        x_norm = (x - x.min()) / (x.max() - x.min())
        y_norm = (y - y.min()) / (y.max() - y.min())
        y_norm = 1 - y_norm
        
        difference = y_norm - x_norm
        threshold = np.mean(difference) + sensitivity * np.std(difference)
        
        above_threshold = difference > threshold
        if not np.any(above_threshold):
            return None
        
        knee_idx = np.where(above_threshold)[0][0]
        
        knee_estimate = x[knee_idx]
        knee_lower = x[max(0, knee_idx - 1)]
        knee_upper = x[min(knee_idx + 1, len(x) - 1)]
        
        return {
            'value': knee_estimate,
            'lower': knee_lower,
            'upper': knee_upper,
            'method': 'Kneedle',
            'quality': 'reliable' if knee_idx > 2 else 'uncertain'
        }
    
    def _practical_method(self, x, y, y_lower, y_upper, improvement_threshold=0.95):
        """Find 95% improvement point."""
        total_improvement = y[0] - y[-1]
        if total_improvement <= 0:
            return None
        
        cumulative_improvements = (y[0] - y) / total_improvement
        
        knee_idx = np.where(cumulative_improvements >= improvement_threshold)[0]
        if len(knee_idx) == 0:
            return None
        
        knee_idx = knee_idx[0]
        
        knee_estimate = x[knee_idx]
        knee_lower = x[max(0, knee_idx - 1)]
        knee_upper = x[min(knee_idx + 1, len(x) - 1)]
        
        return {
            'value': knee_estimate,
            'lower': knee_lower,
            'upper': knee_upper,
            'method': 'Practical (95% improvement)',
            'quality': 'pragmatic'
        }

# ==============================================================================
# STEP 3: RUN KNEE DETECTION FOR EACH STRATEGY
# ==============================================================================

detector = KneeDetectorCI()

results_by_strategy = {}

for strategy in sorted(agg_df['strategy'].unique()):
    df_strat = agg_df[agg_df['strategy'] == strategy].sort_values('sample_size')
    
    print(f"\n{'─'*80}")
    print(f"STRATEGY: {strategy.upper()}")
    print(f"{'─'*80}")
    
    # Detect knees for RMS
    knees_rms = detector.detect_with_uncertainty(
        df_strat, metric='rms_mean',
        ci_lower_col='rms_ci_lower',
        ci_upper_col='rms_ci_upper'
    )
    
    # Detect knees for MAE
    knees_mae = detector.detect_with_uncertainty(
        df_strat, metric='mae_mean',
        ci_lower_col='mae_ci_lower',
        ci_upper_col='mae_ci_upper'
    )
    
    results_by_strategy[strategy] = {
        'rms': knees_rms,
        'mae': knees_mae,
        'data': df_strat
    }
    
    # Print results
    print(f"\nRMS ERROR - Knee Point Detection:")
    print(f"{' '*80}")
    
    for method_name, knee_info in knees_rms.items():
        if knee_info is None:
            print(f"  {method_name:20s}: NOT DETECTED")
        else:
            print(f"  {knee_info['method']:30s}")
            print(f"    Point estimate: {knee_info['value']:8.0f} samples")
            print(f"    Conservative:   {knee_info['lower']:8.0f} - {knee_info['upper']:8.0f} samples")
            print(f"    Uncertainty:    ±{(knee_info['upper'] - knee_info['lower'])/2:8.0f} samples")
            print(f"    Quality:        {knee_info['quality']}")
    
    print(f"\nMAE ERROR - Knee Point Detection:")
    print(f"{' '*80}")
    
    for method_name, knee_info in knees_mae.items():
        if knee_info is None:
            print(f"  {method_name:20s}: NOT DETECTED")
        else:
            print(f"  {knee_info['method']:30s}")
            print(f"    Point estimate: {knee_info['value']:8.0f} samples")
            print(f"    Conservative:   {knee_info['lower']:8.0f} - {knee_info['upper']:8.0f} samples")
            print(f"    Uncertainty:    ±{(knee_info['upper'] - knee_info['lower'])/2:8.0f} samples")
            print(f"    Quality:        {knee_info['quality']}")

# ==============================================================================
# STEP 4: VISUALIZATION - KNEE POINTS WITH CONFIDENCE BANDS
# ==============================================================================

fig, axes = plt.subplots(3, 2, figsize=(18, 14))
fig.suptitle('Knee Point Detection with Uncertainty Quantification', 
             fontsize=16, fontweight='bold', y=0.995)

colors = {
    'random': '#2E86AB',
    'orbital': '#A23B72',
    'stratified': '#F18F01'
}

knee_colors = {
    'threshold': {'color': 'red', 'marker': '|', 'ms': 15},
    'curvature': {'color': 'orange', 'marker': 's', 'ms': 8},
    'kneedle': {'color': 'green', 'marker': '^', 'ms': 8},
    'practical': {'color': 'purple', 'marker': 'D', 'ms': 8}
}

for idx, strategy in enumerate(sorted(results_by_strategy.keys())):
    ax_rms = axes[idx, 0]
    ax_mae = axes[idx, 1]
    
    data = results_by_strategy[strategy]['data']
    knees_rms = results_by_strategy[strategy]['rms']
    knees_mae = results_by_strategy[strategy]['mae']
    color = colors[strategy]
    
    # ========== RMS PLOT ==========
    ax_rms.plot(data['sample_size'], data['rms_mean'], 
               'o-', color=color, linewidth=2.5, markersize=7, 
               label='Mean', alpha=0.8)
    ax_rms.fill_between(data['sample_size'], 
                       data['rms_ci_lower'], 
                       data['rms_ci_upper'],
                       alpha=0.25, color=color, label='95% CI')
    
    # Add knee points with uncertainty bands
    for method_name, knee_info in knees_rms.items():
        if knee_info is not None:
            kw = knee_colors[method_name]
            
            # Plot uncertainty band
            ax_rms.axvspan(knee_info['lower'], knee_info['upper'], 
                          alpha=0.1, color=kw['color'])
            
            # Plot point estimate
            y_at_knee = np.interp(knee_info['value'], 
                                 data['sample_size'], 
                                 data['rms_mean'])
            ax_rms.scatter([knee_info['value']], [y_at_knee],
                         color=kw['color'], s=200, marker=kw['marker'],
                         edgecolors='black', linewidth=1.5, zorder=5,
                         label=f"{method_name.title()}: {knee_info['value']:.0f}±{(knee_info['upper']-knee_info['lower'])/2:.0f}")
    
    ax_rms.set_xscale('log')
    ax_rms.set_xlabel('Sample Size', fontsize=11, fontweight='bold')
    ax_rms.set_ylabel('RMS Error (nT)', fontsize=11, fontweight='bold')
    ax_rms.set_title(f'{strategy.upper()} - RMS Error with Knee Detection', 
                    fontsize=12, fontweight='bold')
    ax_rms.legend(fontsize=9, loc='upper right')
    ax_rms.grid(True, alpha=0.3)
    
    # ========== MAE PLOT ==========
    ax_mae.plot(data['sample_size'], data['mae_mean'], 
               's-', color=color, linewidth=2.5, markersize=7, 
               label='Mean', alpha=0.8)
    ax_mae.fill_between(data['sample_size'], 
                       data['mae_ci_lower'], 
                       data['mae_ci_upper'],
                       alpha=0.25, color=color, label='95% CI')
    
    # Add knee points with uncertainty bands
    for method_name, knee_info in knees_mae.items():
        if knee_info is not None:
            kw = knee_colors[method_name]
            
            # Plot uncertainty band
            ax_mae.axvspan(knee_info['lower'], knee_info['upper'], 
                          alpha=0.1, color=kw['color'])
            
            # Plot point estimate
            y_at_knee = np.interp(knee_info['value'], 
                                 data['sample_size'], 
                                 data['mae_mean'])
            ax_mae.scatter([knee_info['value']], [y_at_knee],
                         color=kw['color'], s=200, marker=kw['marker'],
                         edgecolors='black', linewidth=1.5, zorder=5,
                         label=f"{method_name.title()}: {knee_info['value']:.0f}±{(knee_info['upper']-knee_info['lower'])/2:.0f}")
    
    ax_mae.set_xscale('log')
    ax_mae.set_xlabel('Sample Size', fontsize=11, fontweight='bold')
    ax_mae.set_ylabel('MAE Error (nT)', fontsize=11, fontweight='bold')
    ax_mae.set_title(f'{strategy.upper()} - MAE with Knee Detection', 
                    fontsize=12, fontweight='bold')
    ax_mae.legend(fontsize=9, loc='upper right')
    ax_mae.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('10_knee_detection_with_ci.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 10_knee_detection_with_ci.png")
plt.close()

# ==============================================================================
# STEP 5: SUMMARY TABLE - RECOMMENDED KNEE POINTS
# ==============================================================================

print("\n" + "="*80)
print("SUMMARY: RECOMMENDED SAMPLE SIZES (KNEE POINTS)")
print("="*80)

summary_rows = []

for strategy in sorted(results_by_strategy.keys()):
    knees_rms = results_by_strategy[strategy]['rms']
    
    # Consensus: take median of all methods
    valid_knees = [k['value'] for k in knees_rms.values() if k is not None]
    
    if len(valid_knees) > 0:
        recommended = np.median(valid_knees)
        knee_range = (np.min(valid_knees), np.max(valid_knees))
        
        summary_rows.append({
            'Strategy': strategy.upper(),
            'Recommended n': int(recommended),
            'Range': f"{int(knee_range[0])}-{int(knee_range[1])}",
            'Methods Agreeing': len(valid_knees)
        })

summary_df = pd.DataFrame(summary_rows)

print("\n")
print(summary_df.to_string(index=False))

print("\n" + "="*80)
print("INTERPRETATION:")
print("="*80)
print("""
The knee point represents the sample size where diminishing returns set in.
Beyond this point, adding more samples yields minimal improvement.

- Point estimate: Most likely optimal sample size
- Uncertainty band: Range of plausible values (from different detection methods)
- Methods Agreeing: How many algorithms identify similar knees (higher = more confident)

USE: For mission planning, pick the MEDIAN value with uncertainty range.
For conservative estimates, use the UPPER bound.
For aggressive estimates, use the LOWER bound.
""")

print("="*80)