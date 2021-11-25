""" Unchanneled Path Length (structured).

Compute the unchanneled path length (UPL), i.e., the shortest distance to a
    channel node (rectangular "raster" grids).

Author: Olivier Gourgue (University of Antwerp & Boston University).

"""

import numpy as np
from scipy import spatial


################################################################################
# Unchanneled path length. #####################################################
################################################################################

def upl(x, y, chn, mask = None):
    """Compute the unchanneled path length (UPL).

    Args:
        x, y (NumPy arrays): Grid cell coordinates (1D).
        chn (NumPy array, boolean): True for channel nodes, False otherwise
            (first dimension for x, second dimension for y), for one (2D array)
            or several time steps (3D array, second dimension for time).
        mask (NumPy array, boolean): True at grid cells where UPL is not
            computed (same shape as chn for one time step; default to None, that
            is, no mask).

    Returns:
        NumPy array: Unchanneled path length (same shape as chn).
    """

    # Number of grid cells.
    nx = chn.shape[0]
    ny = chn.shape[1]

    # Reshape chn as a 3D array, if needed.
    if chn.ndim == 2:
        chn = chn.reshape((nx, ny, 1))

    # Initialize.
    upl = np.zeros(chn.shape)

    # Number of time steps.
    nt = chn.shape[2]

    # Default mask.
    if mask is None:
        mask = np.zeros((nx, ny), dtype = bool)

    # Area of interest.
    not_mask = np.logical_not(mask)

    # Mesh grid.
    xx, yy = np.meshgrid(x, y, indexing = 'ij')

    # Loop over time steps.
    for i in range(nt):

        # Reshape into 1D arrays.
        chn_flat = chn[:, :, i].reshape(-1)
        not_mask_flat = not_mask.reshape(-1)
        xx_flat = xx.reshape(-1)
        yy_flat = yy.reshape(-1)
        upl_flat = upl[:, :, i].reshape(-1)

        # Channel node indices and coordinates.
        ind_chn = np.flatnonzero(chn_flat * not_mask_flat)
        xy_chn = np.array([xx_flat[ind_chn], yy_flat[ind_chn]]).T

        # Platform node indices and coordinates.
        ind_plt = np.flatnonzero(np.logical_not(chn_flat) * not_mask_flat)
        xy_plt = np.array([xx_flat[ind_plt], yy_flat[ind_plt]]).T

        # UPL.
        if len(ind_chn) > 0:
            tree = spatial.KDTree(xy_chn)
            upl_plt, ind = tree.query(xy_plt)
            upl_flat[ind_plt] = upl_plt

        upl[:, :, i] = upl_flat.reshape((nx, ny))

    # Reshape as a 2D array, if needed.
    if nt == 1:
        upl = upl.reshape((nx, ny))

    return upl