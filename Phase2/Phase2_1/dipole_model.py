# dipole_model.py
"""
Main dipole modeling script using dipole_physics module.
Handles data loading, fitting, evaluation, and plotting.
"""

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cdflib

# Core physics import
from dipole_physics import (
    sph_to_cart, nec_to_cart, cart_to_nec, cart_to_nec_grid,
    dipole_field, residuals, cart_to_sph_direction, MU0_OVER_4PI
)

def setup_global_map(title, figsize=(14, 8)):
    """Setup a clean global map with coastlines."""
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines(linewidth=0.5, color='black', alpha=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray',
                alpha=0.3, linestyle='--')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    return fig, ax

def plot_continuous_field_on_earth(lon_grid, lat_grid, data_grid, title, component_name='Component'):
    """Plot continuous field on global grid (pcolormesh)."""
    fig, ax = setup_global_map(title)
    if 'Magnitude' in title or '|B|' in title:
        cmap = 'viridis'
        vmin = np.percentile(data_grid, 2)
        vmax = np.percentile(data_grid, 98)
        norm = None
    else:
        cmap = 'RdBu_r'
        abs_max = np.percentile(np.abs(data_grid), 98)
        vmin, vmax = -abs_max, abs_max
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    mesh = ax.pcolormesh(
        lon_grid, lat_grid, data_grid,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        vmin=vmin if norm is None else None,
        vmax=vmax if norm is None else None,
        shading='auto',
        rasterized=True
    )
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal',
                       pad=0.05, aspect=60, shrink=0.9, extend='both')
    cbar.set_label(f'{component_name} (nT)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig, ax

def plot_scatter_field_on_earth(lon_data, lat_data, data, title, component_name='Component'):
    """Plot scattered field data as points."""
    fig, ax = setup_global_map(title)
    if 'Magnitude' in title or '|B|' in title:
        cmap = 'viridis'
        vmin = np.percentile(data, 2)
        vmax = np.percentile(data, 98)
        norm = None
    else:
        cmap = 'RdBu_r'
        abs_max = np.percentile(np.abs(data), 98)
        vmin, vmax = -abs_max, abs_max
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    sc = ax.scatter(
        lon_data, lat_data, c=data,
        s=20,
        cmap=cmap,
        norm=norm,
        vmin=vmin if norm is None else None,
        vmax=vmax if norm is None else None,
        edgecolor='none',
        alpha=0.8,
        transform=ccrs.PlateCarree(),
        rasterized=True
    )
    cbar = plt.colorbar(sc, ax=ax, orientation='horizontal',
                       pad=0.05, aspect=60, shrink=0.9, extend='both')
    cbar.set_label(f'{component_name} (nT)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig, ax

def create_global_grid(radius_mean, resolution=2.0):
    """Create a global lat-lon grid at constant radius (CONTINUOUS)."""
    lats = np.arange(-89.5, 90.0, resolution)
    lons = np.arange(-180.0, 180.0, resolution)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()
    radius_flat = radius_mean * np.ones_like(lat_flat)
    r_cart_flat = sph_to_cart(radius_flat, lat_flat, lon_flat)
    return lon_grid, lat_grid, r_cart_flat

# Main execution
if __name__ == "__main__":
    NUM_SAMPLES_TRAIN = 100
    USE_DUMMY_DATA = False
    
    if USE_DUMMY_DATA:
        np.random.seed(42)
        NUM_SAMPLES_TOTAL = 86400
        radius_all = np.random.uniform(6371e3, 6471e3, NUM_SAMPLES_TOTAL)
        lat_all = np.random.uniform(-90, 90, NUM_SAMPLES_TOTAL)
        lon_all = np.random.uniform(-180, 180, NUM_SAMPLES_TOTAL)
        B_NEC_all = np.random.randn(NUM_SAMPLES_TOTAL, 3) * 1e-5
    else:
        cdf_file = cdflib.CDF(
            "D:/Study/Masters/Sem-3/EPM/Project/"
            "SW_PREL_MAGA_LR_1B_20181212T000000_20181212T235959_0603_MDR_MAG_LR.cdf"
        )
        info = cdf_file.cdf_info()
        variables = info.zVariables if info.zVariables else info.rVariables
        target_vars = [v for v in variables
                      if any(x in v for x in ['Radius', 'Lat', 'Long', 'B_NEC'])]
        data = {var: cdf_file.varget(var) for var in target_vars}
        min_len = min(len(arr) for arr in data.values())
        NUM_SAMPLES_TOTAL = min_len
        all_indices = np.arange(NUM_SAMPLES_TOTAL)
        radius_all = np.array(data['Radius'][all_indices])
        lat_all = np.array(data['Latitude'][all_indices])
        lon_all = np.array(data['Longitude'][all_indices])
        B_NEC_all = np.array(data['B_NEC'][all_indices])
        
        print("\n=== DATA DIAGNOSTICS ===")
        print(f"Loaded {NUM_SAMPLES_TOTAL} samples from CDF file")
        print(f"Radius: {radius_all.min():.3e} - {radius_all.max():.3e} m "
              f"(alt ~{(radius_all.mean()-6371e3)/1e3:.0f} km)")
        print(f"Lat: [{lat_all.min():.1f}, {lat_all.max():.1f}]°")
        print(f"Lon: [{lon_all.min():.1f}, {lon_all.max():.1f}]°")
        B_mag_all = np.linalg.norm(B_NEC_all, axis=1)
        print(f"|B|: [{B_mag_all.min():.0f}, {B_mag_all.max():.0f}] nT")
        print("Converting B_NEC from nT to Tesla")
        B_NEC_all = B_NEC_all * 1e-9
        print("=" * 50)
    
    # Train/Test split
    print("\n=== TRAIN/TEST SPLIT ===")
    print(f"Total points available: {NUM_SAMPLES_TOTAL}")
    print(f"Training set size: {NUM_SAMPLES_TRAIN}")
    print(f"Testing set size: {NUM_SAMPLES_TOTAL - NUM_SAMPLES_TRAIN}")
    np.random.seed(42)
    train_indices = np.random.choice(NUM_SAMPLES_TOTAL, NUM_SAMPLES_TRAIN, replace=False)
    test_indices = np.setdiff1d(np.arange(NUM_SAMPLES_TOTAL), train_indices)
    
    radius_train = radius_all[train_indices]
    lat_train = lat_all[train_indices]
    lon_train = lon_all[train_indices]
    B_NEC_train = B_NEC_all[train_indices]
    B_cart_train = nec_to_cart(B_NEC_train, lat_train, lon_train)
    
    radius_test = radius_all[test_indices]
    lat_test = lat_all[test_indices]
    lon_test = lon_all[test_indices]
    B_NEC_test = B_NEC_all[test_indices]
    B_cart_test = nec_to_cart(B_NEC_test, lat_test, lon_test)
    
    print("✓ Split complete")
    print("=" * 50)
    
    # Fit dipole
    print("\n=== FITTING ON TRAINING DATA ===")
    r_train = sph_to_cart(radius_train, lat_train, lon_train)
    params0 = [8e22, 8e22, 8e22, 1000.0, 1000.0, 1000.0]
    result = least_squares(
        residuals, params0,
        args=(radius_train, lat_train, lon_train, B_cart_train),
        method='lm', ftol=1e-15, xtol=1e-15, gtol=1e-15,
        verbose=1, max_nfev=1000
    )
    
    mx, my, mz, x0, y0, z0 = result.x
    m_vec = np.array([mx, my, mz])
    r0_vec = np.array([x0, y0, z0])
    
    print("\n=== FITTED PARAMETERS ===")
    print(f"Success: {result.success}")
    print(f"m: [{mx:.3e}, {my:.3e}, {mz:.3e}] A·m²")
    print(f"|m|: {np.linalg.norm(m_vec):.3e} A·m²")
    print(f"r0: [{x0:.3e}, {y0:.3e}, {z0:.3e}] m")
    print("=" * 50)
    
    # Dipole poles
    m_hat = m_vec / np.linalg.norm(m_vec)
    radius_mean = radius_all.mean()
    R_shell = radius_mean
    end1_cart = m_hat * R_shell
    end2_cart = -m_hat * R_shell
    dip_lat1, dip_lon1 = cart_to_sph_direction(end1_cart)
    dip_lat2, dip_lon2 = cart_to_sph_direction(end2_cart)
    
    print("\n=== DIPOLE AXIS ORIENTATION ===")
    print(f"North pole: lat={dip_lat1:.1f}°, lon={dip_lon1:.1f}°")
    print(f"South pole: lat={dip_lat2:.1f}°, lon={dip_lon2:.1f}°")
    
    # Training evaluation
    print("\n=== TRAINING SET EVALUATION ===")
    B_fit_train = dipole_field(r_train, m_vec, r0_vec)
    residuals_train = B_cart_train - B_fit_train
    rms_error_train = np.sqrt(np.mean(residuals_train**2))
    r2_train = 1 - (np.sum(residuals_train**2) /
                    np.sum((B_cart_train - np.mean(B_cart_train))**2))
    print(f"Training RMS Error: {rms_error_train * 1e9:.1f} nT")
    print(f"Training R²: {r2_train:.4f}")
    print("=" * 50)
    
    # Test evaluation
    print("\n=== TEST SET EVALUATION (GENERALIZATION) ===")
    print(f"Evaluating on {len(test_indices)} points NOT used for fitting...")
    r_test = sph_to_cart(radius_test, lat_test, lon_test)
    B_fit_test = dipole_field(r_test, m_vec, r0_vec)
    residuals_test = B_cart_test - B_fit_test
    rms_error_test = np.sqrt(np.mean(residuals_test**2))
    r2_test = 1 - (np.sum(residuals_test**2) /
                   np.sum((B_cart_test - np.mean(B_cart_test))**2))
    print(f"Test RMS Error: {rms_error_test * 1e9:.1f} nT")
    print(f"Test R²: {r2_test:.4f}")
    
    print("\n=== GENERALIZATION ANALYSIS ===")
    print(f"Training RMS: {rms_error_train * 1e9:>8.1f} nT")
    print(f"Test RMS: {rms_error_test * 1e9:>8.1f} nT")
    print(f"Difference: {(rms_error_test - rms_error_train) * 1e9:>8.1f} nT")
    print(f"Ratio (Test/Train): {rms_error_test / rms_error_train:.2f}x\n")
    
    # Global model grid
    print("\n=== CREATING CONTINUOUS GLOBAL MODEL GRID ===")
    lon_grid, lat_grid, r_cart_flat = create_global_grid(radius_mean, resolution=2.0)
    print(f"Global grid: {lon_grid.shape[0]} x {lon_grid.shape[1]} = {lon_grid.size} points")
    print("Computing model field on entire globe...")
    B_model_global = dipole_field(r_cart_flat, m_vec, r0_vec)  # Tesla
    B_model_global_nT = B_model_global * 1e9  # nT
    Bx_grid = B_model_global_nT[:, 0].reshape(lon_grid.shape)
    By_grid = B_model_global_nT[:, 1].reshape(lon_grid.shape)
    Bz_grid = B_model_global_nT[:, 2].reshape(lon_grid.shape)
    Bmag_grid = np.linalg.norm(B_model_global_nT, axis=1).reshape(lon_grid.shape)
    print("✓ Global model field computed")
    
    # Convert to NEC
    N_grid, E_grid, C_grid = cart_to_nec_grid(B_model_global_nT, lat_grid, lon_grid)
    
    # Plotting (all the original visualizations)
    print("\n=== GENERATING VISUALIZATIONS ===")
    
    # Observed field
    B_obs_test_nT = B_cart_test * 1e9
    fig, ax = plot_scatter_field_on_earth(
        lon_test, lat_test, np.linalg.norm(B_obs_test_nT, axis=1),
        'Observed: |B| Magnitude (Test Set, Cartesian)', '|B|'
    )
    ax.scatter(dip_lon1, dip_lat1, color='red', s=60, marker='o', 
               edgecolor='black', linewidth=1, transform=ccrs.PlateCarree(), zorder=5)
    ax.scatter(dip_lon2, dip_lat2, color='blue', s=60, marker='o', 
               edgecolor='black', linewidth=1, transform=ccrs.PlateCarree(), zorder=5)
    plt.show()
    
    # NEC conversions and residuals
    B_obs_test_nec = cart_to_nec(B_cart_test, lat_test, lon_test)
    B_fit_test_nec = cart_to_nec(B_fit_test, lat_test, lon_test)
    residuals_test_nec = B_obs_test_nec - B_fit_test_nec
    B_obs_test_nec_nT = B_obs_test_nec * 1e9
    B_fit_test_nec_nT = B_fit_test_nec * 1e9
    residuals_test_nec_nT = residuals_test_nec * 1e9
    
    print(f"Max residual magnitude (NEC): "
          f"{np.max(np.linalg.norm(residuals_test_nec_nT, axis=1)):.1f} nT")
    
    # Observed NEC components
    print("\n=== PLOTTING OBSERVED NEC COMPONENTS (Test Set) ===")
    for comp, name in zip(range(3), ['N', 'E', 'C']):
        fig, ax = plot_scatter_field_on_earth(
            lon_test, lat_test, B_obs_test_nec_nT[:, comp],
            f'Observed: {name} Component (Test Set, NEC)', name
        )
        ax.scatter(dip_lon1, dip_lat1, color='red', s=60, marker='o', 
                   edgecolor='black', linewidth=1, transform=ccrs.PlateCarree(), zorder=5)
        ax.scatter(dip_lon2, dip_lat2, color='blue', s=60, marker='o', 
                   edgecolor='black', linewidth=1, transform=ccrs.PlateCarree(), zorder=5)
        plt.show()
    
    # Residual NEC components
    print("\n=== PLOTTING RESIDUAL NEC COMPONENTS (Test Set) ===")
    for comp, name in zip(range(3), ['ΔN', 'ΔE', 'ΔC']):
        fig, ax = plot_scatter_field_on_earth(
            lon_test, lat_test, residuals_test_nec_nT[:, comp],
            f'Residual: {name} (Obs - Model, NEC)', name
        )
        ax.scatter(dip_lon1, dip_lat1, color='red', s=60, marker='o', 
                   edgecolor='black', linewidth=1, transform=ccrs.PlateCarree(), zorder=5)
        ax.scatter(dip_lon2, dip_lat2, color='blue', s=60, marker='o', 
                   edgecolor='black', linewidth=1, transform=ccrs.PlateCarree(), zorder=5)
        plt.show()
    
    # Model field maps
    print("\n=== GENERATING MODEL FIELD MAPS (Global, NEC) ===")
    fig, ax = plot_continuous_field_on_earth(
        lon_grid, lat_grid, Bmag_grid,
        'Model: |B| Magnitude (Continuous Global, Cartesian)', '|B|'
    )
    ax.scatter(dip_lon1, dip_lat1, color='red', s=60, marker='o', 
               edgecolor='black', linewidth=1, transform=ccrs.PlateCarree(), zorder=5)
    ax.scatter(dip_lon2, dip_lat2, color='blue', s=60, marker='o', 
               edgecolor='black', linewidth=1, transform=ccrs.PlateCarree(), zorder=5)
    plt.show()
    
    for grid, name in zip([N_grid, E_grid, C_grid], ['N', 'E', 'C']):
        fig, ax = plot_continuous_field_on_earth(
            lon_grid, lat_grid, grid,
            f'Model: {name} Component (Continuous Global, NEC)', name
        )
        ax.scatter(dip_lon1, dip_lat1, color='red', s=60, marker='o', 
                   edgecolor='black', linewidth=1, transform=ccrs.PlateCarree(), zorder=5)
        ax.scatter(dip_lon2, dip_lat2, color='blue', s=60, marker='o', 
                   edgecolor='black', linewidth=1, transform=ccrs.PlateCarree(), zorder=5)
        plt.show()
    
    print("\n✓ All global maps complete!")
    print("\n" + "=" * 70)
    print("KEY TAKEAWAY:")
    print("=" * 70)
    print(f"Trained dipole on {NUM_SAMPLES_TRAIN} points out of {NUM_SAMPLES_TOTAL} total")
    print(f"Tested on {len(test_indices)} independent (unseen) points")
    print(f"Training RMS: {rms_error_train * 1e9:.1f} nT")
    print(f"Test RMS: {rms_error_test * 1e9:.1f} nT")
    print(f"Test/Train Ratio: {rms_error_test / rms_error_train:.2f}x")
    print("\nObserved (NEC): scattered at Swarm locations (test set)")
    print("Model (NEC): continuous 2° global grid at mean Swarm altitude")
    print("Residuals (NEC): scattered error at test locations")
    print("Dipole poles: RED = North, BLUE = South on every map.")
    print("\nThis is genuine validation: testing on data not used for fitting.")
    print("=" * 70)
