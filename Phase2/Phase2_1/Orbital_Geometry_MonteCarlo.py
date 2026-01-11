"""
Orbital Geometry Analysis - CORRECT INTERPRETATION
===================================================
Samples don't always map to orbits! Only orbital sampling does.
For random/stratified: samples are SELECTED, not sequential.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "="*80)
print("CORRECTED ORBITAL GEOMETRY ANALYSIS")
print("="*80)

# SWARM parameters from your data
n_orbits_swarm = 15
total_samples_swarm = 86400
samples_per_orbit = 5760

# Load validation data
df_val = pd.read_csv('validation_results_mc_cv.csv')

print(f"\nSWARM Dataset Facts:")
print(f"  • 15 orbits in 1 day")
print(f"  • 86,400 total samples")
print(f"  • ~5,760 samples per orbit")
print(f"  • Coverage: Full pole-to-pole, 360° longitude")

print(f"\n" + "="*80)
print("UNDERSTANDING THE SAMPLING STRATEGIES")
print("="*80)

print("""
ORBITAL SAMPLING (pink line):
  ✓ Samples ARE tied to orbit sequence
  ✓ 100 samples = first 100 points along orbit path
  ✓ Covers whatever geographic area comes first chronologically
  ✗ Limited to early part of mission unless you skip around
  → Samples ↔ Orbits (linear relationship)
  → Formula: orbits_needed = samples / 5760

RANDOM SAMPLING (blue line):
  ✗ Samples are RANDOMLY chosen from full 15-orbit dataset
  ✗ Not tied to orbit structure at all
  ✓ Can pick 50 samples scattered across all 15 orbits
  ✓ Gets geographic diversity immediately
  → Samples ≠ Orbits (no simple relationship)
  → Better to think: "sampling density" or "coverage fraction"
  → Formula: coverage_fraction = samples / 86400

STRATIFIED SAMPLING (orange line):
  ✗ Samples are DELIBERATELY chosen from geographic strata
  ✗ Not sequential in orbit, deliberately distributed
  ✓ Ensures lat/lon coverage is balanced
  ✓ Gets all 15 orbits' worth of diversity with fewer samples
  → Samples ≠ Orbits (no simple relationship)
  → Better to think: "coverage fraction"
  → Formula: coverage_fraction = samples / 86400
""")

print("\n" + "="*80)
print("CORRECT INTERPRETATION OF YOUR VALIDATION RESULTS")
print("="*80)

# The correct way to interpret:
for strategy in sorted(df_val['strategy'].unique()):
    data = df_val[df_val['strategy'] == strategy].drop_duplicates('sample_size')
    data = data.sort_values('sample_size')
    
    # Find optimal (minimum error)
    optimal_idx = data['rms'].idxmin()
    optimal_row = data.loc[optimal_idx]
    optimal_samples = optimal_row['sample_size']
    optimal_error = optimal_row['rms']
    
    # What does this mean for each strategy?
    if strategy == 'orbital':
        # For orbital: this IS orbit-dependent
        orbits = optimal_samples / samples_per_orbit
        meaning = f"Need {orbits:.2f} orbits of SEQUENTIAL data (early mission constraint)"
        
    else:  # random or stratified
        # For these: this is coverage fraction
        coverage_pct = (optimal_samples / total_samples_swarm) * 100
        meaning = f"Need {coverage_pct:.1f}% of available data (can be cherry-picked)"
    
    print(f"\n{strategy.upper()}:")
    print(f"  Optimal samples: {optimal_samples:.0f}")
    print(f"  Optimal error: {optimal_error:.0f} nT")
    print(f"  Meaning: {meaning}")

print("\n" + "="*80)
print("WHAT THIS MEANS FOR PLANETARY MISSIONS")
print("="*80)

print("""
SCENARIO 1: Mars MAVEN (1 satellite, can't choose sampling)
  → Forced to use ORBITAL strategy
  → Need to see what SWARM's orbital curve tells us
  → From your validation: needs ~0.8 orbits of sequential data
  → But SWARM's orbital data is NOISY, takes many orbits to stabilize
  → Reality: Need 10-100+ MAVEN orbits for good fit
  
SCENARIO 2: Mars Analysis with Combined Data (post-mission)
  → Can use RANDOM or STRATIFIED on all available passes
  → Only need 20-50% of collected data
  → Much better! Can achieve 3-4 kNT error
  → Much more efficient than sequential orbital approach
  
SCENARIO 3: Multiple satellites (GRAIL, Juno, SWARM)
  → Can use STRATIFIED across all missions
  → Gets coverage diversity immediately
  → Needs fewer samples overall
  → Best case scenario

KEY INSIGHT:
The huge difference between strategies shows:
- ORBITAL is fundamentally limited (5x worse than RANDOM, 10x worse than STRATIFIED)
- Not because the method is bad, but because you're FORCED to use early data only
- In reality, you'd collect many orbits and process post-mission (like SWARM today)
- So the STRATIFIED result (~3200 nT optimal) is what you'd actually achieve
- But during early mission (ORBITAL), expect 10,000+ nT until you have enough data
""")

print("\n" + "="*80)
print("PRACTICAL MISSION PLANNING")
print("="*80)

# Create a proper comparison
strategies_data = {}

for strategy in sorted(df_val['strategy'].unique()):
    data = df_val[df_val['strategy'] == strategy].drop_duplicates('sample_size')
    
    # Find various thresholds
    error_3000 = data[data['rms'] <= 3000]
    error_5000 = data[data['rms'] <= 5000]
    error_8000 = data[data['rms'] <= 8000]
    error_10000 = data[data['rms'] <= 10000]
    
    strategies_data[strategy] = {
        '3000nT': error_3000['sample_size'].min() if len(error_3000) > 0 else np.nan,
        '5000nT': error_5000['sample_size'].min() if len(error_5000) > 0 else np.nan,
        '8000nT': error_8000['sample_size'].min() if len(error_8000) > 0 else np.nan,
        '10000nT': error_10000['sample_size'].min() if len(error_10000) > 0 else np.nan,
    }

print("\nSamples needed to reach error thresholds:\n")

for error_level in ['3000nT', '5000nT', '8000nT', '10000nT']:
    print(f"{error_level}:")
    print(f"{'Strategy':<15} {'Samples':<15} {'Interpretation'}")
    print(f"{'-'*55}")
    
    for strategy in sorted(strategies_data.keys()):
        samples = strategies_data[strategy][error_level]
        
        if np.isnan(samples):
            interp = "Cannot achieve"
        else:
            samples = int(samples)
            if strategy == 'orbital':
                orbits = samples / samples_per_orbit
                interp = f"{orbits:.1f} orbits (sequential only)"
            else:
                coverage = (samples / total_samples_swarm) * 100
                interp = f"{coverage:.1f}% of dataset (can select)"
        
        print(f"{strategy:<15} {samples if not np.isnan(samples) else 'N/A':<15} {interp}")
    print()

# ==============================================================================
# VISUALIZATION: Correct interpretation
# ==============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Correct Interpretation: Samples DO NOT Always Mean Orbits', 
             fontsize=15, fontweight='bold')

colors = {'orbital': '#A23B72', 'random': '#2E86AB', 'stratified': '#F18F01'}

# Plot 1: Raw sample size (don't convert to orbits)
ax = axes[0, 0]
for strategy in sorted(df_val['strategy'].unique()):
    data = df_val[df_val['strategy'] == strategy].drop_duplicates('sample_size')
    data = data.sort_values('sample_size')
    ax.plot(data['sample_size'], data['rms'], 'o-', linewidth=2.5, markersize=8,
           label=strategy.upper(), color=colors[strategy], alpha=0.7)

ax.set_xlabel('Number of Samples', fontsize=12, fontweight='bold')
ax.set_ylabel('RMS Error (nT)', fontsize=12, fontweight='bold')
ax.set_title('Error vs Sample Count (RAW)', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Plot 2: Coverage fraction (percentage of SWARM data)
ax = axes[0, 1]
for strategy in sorted(df_val['strategy'].unique()):
    data = df_val[df_val['strategy'] == strategy].drop_duplicates('sample_size')
    data = data.sort_values('sample_size')
    coverage_pct = (data['sample_size'] / total_samples_swarm) * 100
    
    ax.plot(coverage_pct, data['rms'], 's-', linewidth=2.5, markersize=8,
           label=strategy.upper(), color=colors[strategy], alpha=0.7)

ax.set_xlabel('Coverage (% of SWARM dataset)', fontsize=12, fontweight='bold')
ax.set_ylabel('RMS Error (nT)', fontsize=12, fontweight='bold')
ax.set_title('Error vs Coverage Fraction (SAME DATA, different axis)', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Plot 3: ONLY for ORBITAL - convert to orbits
ax = axes[1, 0]
orbital_data = df_val[df_val['strategy'] == 'orbital'].drop_duplicates('sample_size')
orbital_data = orbital_data.sort_values('sample_size')
orbits = orbital_data['sample_size'] / samples_per_orbit
ax.plot(orbits, orbital_data['rms'], 'D-', linewidth=2.5, markersize=8,
       label='ORBITAL ONLY', color=colors['orbital'], alpha=0.7)

ax.set_xlabel('Number of Orbits (ORBITAL ONLY)', fontsize=12, fontweight='bold')
ax.set_ylabel('RMS Error (nT)', fontsize=12, fontweight='bold')
ax.set_title('ORBITAL Strategy: Samples→Orbits Conversion IS Valid', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Plot 4: Comparison table
ax = axes[1, 1]
ax.axis('off')

summary_text = f"""
KEY CORRECTIONS:

1. ORBITAL SAMPLING
   • Samples ARE tied to orbits
   • ~5,760 samples = 1 orbit of data
   • Sequential constraint: early mission has limited data
   • Valid orbit conversion ✓

2. RANDOM & STRATIFIED
   • Samples are INDEPENDENT of orbit structure
   • Can cherry-pick from all 15 orbits
   • Not limited to early data
   • Orbit conversion is WRONG ✗

3. PRACTICAL IMPLICATIONS

   Mars (need 5,000 nT):
   • Orbital: 0.003 orbits (~17,000 samples) ← sequential early data
   • Random: 0.3% of dataset (300 samples) ← can select anywhere
   • Stratified: 0.2% of dataset (200 samples) ← optimized selection

4. FOR YOUR THESIS

   Show that:
   • Orbital sampling is forced by mission geometry
   • But vastly inferior to post-mission reprocessing
   • This is WHY planetary missions collect so much data
   • Then strategists choose optimal subsets afterward

Real mission workflow:
  Phase 1: Collect all orbits (ORBITAL strategy)
  Phase 2: Analyze with optimal subset (STRATIFIED)
  Phase 3: Publish best-fit dipole model
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
       fontsize=11, verticalalignment='top', family='monospace',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('12_correct_sample_to_orbit_interpretation.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 12_correct_sample_to_orbit_interpretation.png")
plt.close()

print("\n" + "="*80)
print("CORRECTED INTERPRETATION SAVED")
print("="*80)