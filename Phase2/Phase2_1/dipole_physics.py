# dipole_physics.py
"""
Core physics module for dipole magnetic field modeling.
Contains coordinate transforms and dipole field calculations.
Pure physics - no plotting, data loading, or optimization.
"""

import numpy as np

# Constants
MU0_OVER_4PI = 1e-7  # magnetic permeability / 4pi in T·m/A (SI units)

def sph_to_cart(radius, lat, lon):
    """Convert spherical coordinates to Cartesian (geocentric)."""
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

def nec_to_cart(B_nec, lat, lon):
    """Transform magnetic field from NEC to geocentric Cartesian coordinates."""
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    B_cart = np.zeros_like(B_nec)
    for i in range(len(lat)):
        cos_lat = np.cos(lat_rad[i])
        sin_lat = np.sin(lat_rad[i])
        cos_lon = np.cos(lon_rad[i])
        sin_lon = np.sin(lon_rad[i])
        north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
        east = np.array([-sin_lon, cos_lon, 0.0])
        center = np.array([-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat])
        Bn, Be, Bc = B_nec[i]
        B_cart[i] = Bn * north + Be * east + Bc * center
    return B_cart

def cart_to_nec(B_cart, lat, lon):
    """Transform magnetic field from geocentric Cartesian to NEC coordinates."""
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    B_nec = np.zeros_like(B_cart)
    for i in range(len(lat)):
        cos_lat = np.cos(lat_rad[i])
        sin_lat = np.sin(lat_rad[i])
        cos_lon = np.cos(lon_rad[i])
        sin_lon = np.sin(lon_rad[i])
        north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
        east = np.array([-sin_lon, cos_lon, 0.0])
        center = np.array([-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat])
        Bx, By, Bz = B_cart[i]
        B_vec = np.array([Bx, By, Bz])
        B_nec[i, 0] = np.dot(B_vec, north)  # N
        B_nec[i, 1] = np.dot(B_vec, east)   # E
        B_nec[i, 2] = np.dot(B_vec, center) # C (radially inward)
    return B_nec

def cart_to_nec_grid(B_cart_flat, lat_grid, lon_grid):
    """
    Convert model field on a global grid from geocentric Cartesian to NEC.
    Inputs:
        B_cart_flat : (N, 3) array (same units everywhere, e.g., nT)
        lat_grid : 2D lat grid (deg)
        lon_grid : 2D lon grid (deg)
    Returns:
        N_grid, E_grid, C_grid with same shape as lat_grid/lon_grid.
    """
    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()
    lat_rad = np.deg2rad(lat_flat)
    lon_rad = np.deg2rad(lon_flat)
    N = np.zeros_like(lat_flat, dtype=float)
    E = np.zeros_like(lat_flat, dtype=float)
    C = np.zeros_like(lat_flat, dtype=float)
    for i in range(lat_flat.size):
        cos_lat = np.cos(lat_rad[i])
        sin_lat = np.sin(lat_rad[i])
        cos_lon = np.cos(lon_rad[i])
        sin_lon = np.sin(lon_rad[i])
        north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
        east = np.array([-sin_lon, cos_lon, 0.0])
        center = np.array([-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat])
        Bx, By, Bz = B_cart_flat[i]
        B_vec = np.array([Bx, By, Bz])
        N[i] = np.dot(B_vec, north)
        E[i] = np.dot(B_vec, east)
        C[i] = np.dot(B_vec, center)
    N_grid = N.reshape(lat_grid.shape)
    E_grid = E.reshape(lat_grid.shape)
    C_grid = C.reshape(lat_grid.shape)
    return N_grid, E_grid, C_grid

def dipole_field(r, m, r0):
    """Calculate dipole magnetic field at positions r (Cartesian)."""
    m = np.atleast_2d(m)
    r0 = np.atleast_2d(r0)
    R = r - r0
    R_mag = np.linalg.norm(R, axis=1, keepdims=True)
    R_hat = R / R_mag
    dot_product = np.sum(m * R_hat, axis=1, keepdims=True)
    B = MU0_OVER_4PI * (3 * dot_product * R_hat - m) / (R_mag ** 3)
    return B

def residuals(params, radius, lat, lon, B_obs_cart):
    """Residual function for least squares (Cartesian, FIT SET)."""
    m = params[:3]
    r0 = params[3:]
    r = sph_to_cart(radius, lat, lon)
    B_model = dipole_field(r, m, r0)
    res = B_obs_cart - B_model
    return res.flatten()

def cart_to_sph_direction(v):
    """Convert a direction vector (x,y,z) to (lat, lon) in degrees."""
    x, y, z = v
    r = np.sqrt(x**2 + y**2 + z**2)
    if r == 0:
        return 0.0, 0.0
    lat = np.arcsin(z / r)
    lon = np.arctan2(y, x)
    return np.rad2deg(lat), np.rad2deg(lon)
