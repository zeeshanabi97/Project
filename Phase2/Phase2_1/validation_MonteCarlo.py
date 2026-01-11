"""
Comprehensive Validation Results Plotting
==========================================
Plot random, orbital, and stratified sampling strategies with confidence intervals
and comprehensive statistical analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 11

# Load data
df = pd.read_csv('validation_results_mc_cv.csv')

print("Data shape:", df.shape)
print("Strategies:", df['strategy'].unique())
print("Sample sizes range:", df['sample_size'].min(), "-", df['sample_size'].max())

# ==============================================================================
# 1. AGGREGATE BY STRATEGY AND SAMPLE SIZE (for mean/CI)
# ==============================================================================

def compute_ci(group, metric='rms', ci=95):
    """Compute mean and confidence interval."""
    data = group[metric].values
    mean = np.mean(data)
    sem = stats.sem(data)
    margin = sem * stats.t.ppf((1 + ci/100) / 2, len(data) - 1)
    return pd.Series({
        'mean': mean,
        'ci_lower': mean - margin,
        'ci_upper': mean + margin,
        'std': np.std(data),
        'count': len(data)
    })

# Aggregate across runs and folds
agg_data = []
for strategy in df['strategy'].unique():
    for sample_size in sorted(df['sample_size'].unique()):
        subset = df[(df['strategy'] == strategy) & (df['sample_size'] == sample_size)]
        if len(subset) == 0:
            continue
        
        row = {
            'strategy': strategy,
            'sample_size': sample_size,
            'n_runs': len(subset)
        }
        
        # Compute stats for each metric
        for metric in ['rms', 'mae', 'r2']:
            ci = compute_ci(subset, metric=metric)
            row.update({
                f'{metric}_mean': ci['mean'],
                f'{metric}_ci_lower': ci['ci_lower'],
                f'{metric}_ci_upper': ci['ci_upper'],
                f'{metric}_std': ci['std']
            })
        
        agg_data.append(row)

agg_df = pd.DataFrame(agg_data)

# ==============================================================================
# FIGURE 1: RMS ERROR WITH CONFIDENCE INTERVALS
# ==============================================================================

fig, axes = plt.subplots(1, 1, figsize=(14, 8))

strategies = sorted(df['strategy'].unique())
colors = {
    'random': '#2E86AB',
    'orbital': '#A23B72',
    'stratified': '#F18F01'
}

for strategy in strategies:
    data = agg_df[agg_df['strategy'] == strategy].sort_values('sample_size')
    
    axes.plot(data['sample_size'], data['rms_mean'], 
             'o-', label=strategy.upper(), color=colors[strategy], 
             linewidth=2.5, markersize=8)
    
    axes.fill_between(data['sample_size'], 
                     data['rms_ci_lower'], 
                     data['rms_ci_upper'],
                     alpha=0.2, color=colors[strategy])

axes.set_xscale('log')
axes.set_xlabel('Training Sample Size', fontsize=12, fontweight='bold')
axes.set_ylabel('RMS Error (nT)', fontsize=12, fontweight='bold')
axes.set_title('RMS Error vs Sample Size with 95% Confidence Intervals', 
               fontsize=14, fontweight='bold')
axes.legend(fontsize=11, loc='upper right')
axes.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('01_rms_error_ci.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_rms_error_ci.png")
plt.close()

# ==============================================================================
# FIGURE 2: MAE WITH CONFIDENCE INTERVALS
# ==============================================================================

fig, axes = plt.subplots(1, 1, figsize=(14, 8))

for strategy in strategies:
    data = agg_df[agg_df['strategy'] == strategy].sort_values('sample_size')
    
    axes.plot(data['sample_size'], data['mae_mean'], 
             's-', label=strategy.upper(), color=colors[strategy], 
             linewidth=2.5, markersize=8)
    
    axes.fill_between(data['sample_size'], 
                     data['mae_ci_lower'], 
                     data['mae_ci_upper'],
                     alpha=0.2, color=colors[strategy])

axes.set_xscale('log')
axes.set_xlabel('Training Sample Size', fontsize=12, fontweight='bold')
axes.set_ylabel('Mean Absolute Error (nT)', fontsize=12, fontweight='bold')
axes.set_title('MAE vs Sample Size with 95% Confidence Intervals', 
               fontsize=14, fontweight='bold')
axes.legend(fontsize=11, loc='upper right')
axes.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('02_mae_ci.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_mae_ci.png")
plt.close()

# ==============================================================================
# FIGURE 3: R² SCORE WITH CONFIDENCE INTERVALS
# ==============================================================================

fig, axes = plt.subplots(1, 1, figsize=(14, 8))

for strategy in strategies:
    data = agg_df[agg_df['strategy'] == strategy].sort_values('sample_size')
    
    axes.plot(data['sample_size'], data['r2_mean'], 
             '^-', label=strategy.upper(), color=colors[strategy], 
             linewidth=2.5, markersize=8)
    
    axes.fill_between(data['sample_size'], 
                     data['r2_ci_lower'], 
                     data['r2_ci_upper'],
                     alpha=0.2, color=colors[strategy])

axes.set_xscale('log')
axes.set_xlabel('Training Sample Size', fontsize=12, fontweight='bold')
axes.set_ylabel('R² Score', fontsize=12, fontweight='bold')
axes.set_title('Model Fit Quality (R²) vs Sample Size with 95% CI', 
               fontsize=14, fontweight='bold')
axes.legend(fontsize=11, loc='lower right')
axes.grid(True, alpha=0.3)
axes.set_ylim([0.96, 0.985])
plt.tight_layout()
plt.savefig('03_r2_score_ci.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_r2_score_ci.png")
plt.close()

# ==============================================================================
# FIGURE 4: INDIVIDUAL STRATEGY DASHBOARDS (3 SUBPLOTS EACH)
# ==============================================================================

for strategy in strategies:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{strategy.upper()} Sampling Strategy - Comprehensive Analysis', 
                fontsize=14, fontweight='bold', y=1.02)
    
    data_strat = agg_df[agg_df['strategy'] == strategy].sort_values('sample_size')
    color = colors[strategy]
    
    # RMS
    axes[0].plot(data_strat['sample_size'], data_strat['rms_mean'], 
                'o-', color=color, linewidth=2.5, markersize=8, label='Mean')
    axes[0].fill_between(data_strat['sample_size'], 
                        data_strat['rms_ci_lower'], 
                        data_strat['rms_ci_upper'],
                        alpha=0.3, color=color, label='95% CI')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Sample Size', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('RMS Error (nT)', fontsize=11, fontweight='bold')
    axes[0].set_title('RMS Error', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(data_strat['sample_size'], data_strat['mae_mean'], 
                's-', color=color, linewidth=2.5, markersize=8, label='Mean')
    axes[1].fill_between(data_strat['sample_size'], 
                        data_strat['mae_ci_lower'], 
                        data_strat['mae_ci_upper'],
                        alpha=0.3, color=color, label='95% CI')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Sample Size', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('MAE (nT)', fontsize=11, fontweight='bold')
    axes[1].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # R²
    axes[2].plot(data_strat['sample_size'], data_strat['r2_mean'], 
                '^-', color=color, linewidth=2.5, markersize=8, label='Mean')
    axes[2].fill_between(data_strat['sample_size'], 
                        data_strat['r2_ci_lower'], 
                        data_strat['r2_ci_upper'],
                        alpha=0.3, color=color, label='95% CI')
    axes[2].set_xscale('log')
    axes[2].set_xlabel('Sample Size', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('R² Score', fontsize=11, fontweight='bold')
    axes[2].set_title('Model Quality', fontsize=12, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'04_{strategy}_dashboard.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 04_{strategy}_dashboard.png")
    plt.close()

# ==============================================================================
# FIGURE 5: COMPARATIVE BOX PLOTS AT KEY SAMPLE SIZES
# ==============================================================================

key_samples = sorted(df['sample_size'].unique())[::len(df['sample_size'].unique())//4]
if len(key_samples) > 4:
    key_samples = key_samples[:4]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, sample_size in enumerate(key_samples[:4]):
    data_sample = df[df['sample_size'] == sample_size]
    
    # RMS
    box_data = [data_sample[data_sample['strategy'] == s]['rms'].values for s in strategies]
    bp = axes[idx].boxplot(box_data, labels=[s.upper() for s in strategies],
                           patch_artist=True, showmeans=True)
    
    for patch, strategy in zip(bp['boxes'], strategies):
        patch.set_facecolor(colors[strategy])
        patch.set_alpha(0.6)
    
    axes[idx].set_ylabel('RMS Error (nT)', fontsize=11, fontweight='bold')
    axes[idx].set_title(f'RMS Distribution at n={sample_size}', 
                       fontsize=12, fontweight='bold')
    axes[idx].grid(True, alpha=0.3, axis='y')

plt.suptitle('RMS Error Distribution Comparison Across Strategies', 
            fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('05_boxplot_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_boxplot_comparison.png")
plt.close()

# ==============================================================================
# FIGURE 6: VARIANCE (STD DEV) ACROSS STRATEGIES
# ==============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# RMS Variance
for strategy in strategies:
    data = agg_df[agg_df['strategy'] == strategy].sort_values('sample_size')
    axes[0].plot(data['sample_size'], data['rms_std'], 
                'o-', label=strategy.upper(), color=colors[strategy],
                linewidth=2.5, markersize=8)

axes[0].set_xscale('log')
axes[0].set_xlabel('Sample Size', fontsize=12, fontweight='bold')
axes[0].set_ylabel('RMS Std Dev (nT)', fontsize=12, fontweight='bold')
axes[0].set_title('RMS Error Variance Across Runs', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# MAE Variance
for strategy in strategies:
    data = agg_df[agg_df['strategy'] == strategy].sort_values('sample_size')
    axes[1].plot(data['sample_size'], data['mae_std'], 
                's-', label=strategy.upper(), color=colors[strategy],
                linewidth=2.5, markersize=8)

axes[1].set_xscale('log')
axes[1].set_xlabel('Sample Size', fontsize=12, fontweight='bold')
axes[1].set_ylabel('MAE Std Dev (nT)', fontsize=12, fontweight='bold')
axes[1].set_title('MAE Variance Across Runs', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.suptitle('Stability Analysis: Variance vs Sample Size', 
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('06_variance_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 06_variance_analysis.png")
plt.close()

# ==============================================================================
# FIGURE 7: CONVERGENCE ANALYSIS (% IMPROVEMENT FROM SMALLEST SIZE)
# ==============================================================================

fig, axes = plt.subplots(1, 1, figsize=(14, 8))

for strategy in strategies:
    data = agg_df[agg_df['strategy'] == strategy].sort_values('sample_size')
    
    # Compute % improvement from baseline
    baseline = data['rms_mean'].iloc[0]
    improvement_pct = ((baseline - data['rms_mean']) / baseline) * 100
    
    axes.plot(data['sample_size'], improvement_pct, 
             'D-', label=strategy.upper(), color=colors[strategy],
             linewidth=2.5, markersize=8)
    
    # Mark 50% improvement point
    try:
        idx_50 = np.where(improvement_pct >= 50)[0][0]
        sample_50 = data['sample_size'].iloc[idx_50]
        axes.scatter([sample_50], [50], s=200, color=colors[strategy], 
                    marker='*', zorder=5, edgecolors='black', linewidth=2)
    except:
        pass

axes.axhline(50, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='50% Improvement')
axes.set_xscale('log')
axes.set_xlabel('Training Sample Size', fontsize=12, fontweight='bold')
axes.set_ylabel('RMS Improvement from Baseline (%)', fontsize=12, fontweight='bold')
axes.set_title('Convergence Analysis: Improvement Rate', fontsize=14, fontweight='bold')
axes.legend(fontsize=11, loc='lower right')
axes.grid(True, alpha=0.3)
axes.set_ylim([0, 100])

plt.tight_layout()
plt.savefig('07_convergence_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 07_convergence_analysis.png")
plt.close()

# ==============================================================================
# FIGURE 8: HEATMAP OF PERFORMANCE METRICS
# ==============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, metric in enumerate(['rms_mean', 'mae_mean', 'r2_mean']):
    
    # Pivot table for heatmap
    pivot_data = agg_df.pivot_table(
        values=metric, 
        index='strategy', 
        columns='sample_size'
    )
    
    sns.heatmap(pivot_data, ax=axes[idx], cmap='RdYlGn_r', annot=True, 
               fmt='.0f' if 'r2' not in metric else '.4f', cbar_kws={'label': metric})
    
    metric_name = {'rms_mean': 'RMS Error (nT)', 
                  'mae_mean': 'MAE (nT)',
                  'r2_mean': 'R² Score'}[metric]
    axes[idx].set_title(f'{metric_name}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Sample Size', fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Strategy', fontsize=11, fontweight='bold')

plt.suptitle('Performance Heatmap Across All Configurations', 
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('08_heatmap_performance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 08_heatmap_performance.png")
plt.close()

# ==============================================================================
# SUMMARY STATISTICS TABLE
# ==============================================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS BY STRATEGY")
print("="*80)

for strategy in strategies:
    data = agg_df[agg_df['strategy'] == strategy]
    print(f"\n{strategy.upper()}:")
    print(f"  RMS  - Mean: {data['rms_mean'].mean():.2f} nT, Std: {data['rms_std'].mean():.2f} nT")
    print(f"  MAE  - Mean: {data['mae_mean'].mean():.2f} nT, Std: {data['mae_std'].mean():.2f} nT")
    print(f"  R²   - Mean: {data['r2_mean'].mean():.6f}, Std: {data['r2_std'].mean():.6f}")
    
    # Best performance
    best_rms_idx = data['rms_mean'].idxmin()
    best_rms_size = data.loc[best_rms_idx, 'sample_size']
    best_rms_val = data.loc[best_rms_idx, 'rms_mean']
    print(f"  Best RMS: {best_rms_val:.2f} nT at n={int(best_rms_size)}")

print("\n" + "="*80)
print("All plots saved successfully!")
print("="*80)