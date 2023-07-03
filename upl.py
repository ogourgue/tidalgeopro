""" Unchanneled Path Length.

Compute the unchanneled path length (UPL), i.e., the shortest distance to a
    channel node (triangular grids).

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
        x, y (NumPy arrays): Node coordinates.
        chn (NumPy array, boolean): True for channel nodes, False otherwise, for
            one (1D array) or several time steps (2D array, second dimension for
            time).
        mask (NumPy array, boolean): True at grid nodes where UPL is not
            computed (same shape as chn for one time step; default to None, that
            is, no mask).

    Returns:
        NumPy array: Unchanneled path length (same shape as chn).
    """

    # Reshape chn as a 2D array, if needed.
    if chn.ndim == 1:
        chn = chn.reshape((-1, 1))

    # Initialize.
    upl = np.zeros(chn.shape)

    # Number of nodes.
    n = chn.shape[0]

    # Number of time steps.
    nt = chn.shape[1]

    # Default mask.
    if mask is None:
        mask = np.zeros((n, 1), dtype = bool)

    # Reshape mask as a 2D array, if needed.
    if mask.ndim == 1:
        mask = mask.reshape((-1, 1))

    # Area of interest.
    not_mask = np.logical_not(mask)

    # Loop over time steps.
    for i in range(nt):

        # Channel node indices and coordinates.
        ind_chn = np.flatnonzero(chn[:, i] * not_mask[:, i])
        xy_chn = np.array([x[ind_chn], y[ind_chn]]).T

        # Platform node indices and coordinates.
        ind_plt = np.flatnonzero(np.logical_not(chn[:, i]) * not_mask[:, i])
        xy_plt = np.array([x[ind_plt], y[ind_plt]]).T

        # UPL.
        if len(ind_chn) > 0:
            tree = spatial.KDTree(xy_chn)
            upl_plt, ind = tree.query(xy_plt)
            upl[ind_plt, i] = upl_plt

    # Reshape as a 1D array, if needed.
    if nt == 1:
        upl = upl.reshape(-1)

    return upl