""" Channels (structured).

Extract tidal channels from digital elevation maps (rectangular "raster" grids).

Author: Olivier Gourgue (University of Antwerp)

"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.path as pth
import matplotlib.pyplot as plt
import numpy as np

from shapely import geometry


################################################################################
# Median neighborhood analysis. ################################################
################################################################################

def mna(ws, z):
    """ Compute window median residuals (m).

    The residual is defined as the difference between the window median
    elevation and the elevation.

    Args:
        ws (int): Window size (number of grid cells, must be an odd number).
        z (NumPy array): Elevation on each node, for one (2D array) or several
            time steps (3D array, third dimension for time).

    Returns:
        NumPy Array: Median residual within each window (same shape as z).
    """

    # Number of grid cells.
    nx = z.shape[0]
    ny = z.shape[1]

    # Reshape z as a 3D array, if needed.
    if z.ndim == 2:
        z = z.reshape((nx, ny, 1))

    # Number of time steps.
    nt = z.shape[2]

    # Window size from center cell to edge ("radius").
    r = int(.5 * (ws - 1))

    # Window medians.
    z_med = np.zeros(z.shape)
    for i in range(nx):
        i0 = np.maximum(0, i - r)
        i1 = np.minimum(i + r, nx) + 1
        for j in range(ny):
            j0 = np.maximum(0, j - r)
            j1 = np.minimum(j + r, ny) + 1
            z_med[i, j, :] = np.nanmedian(z[i0:i1, j0:j1, :], axis = [0, 1])

    # Mini-cloud residuals.
    z_res = z_med - z

    # Reshape z_res as a 2D array, if needed.
    if nt == 1:
        z_res = z_res.reshape((nx, ny))

    return z_res


################################################################################
# Extract channels. ############################################################
################################################################################

def channels(z_res, z_res_c):
    """ Extract channel grid cells.

    Args:
        z_res (list of NumPy arrays): Median residuals for different
            neighborhood window sizes, for one (2D arrays) or several time steps
            (3D arrays, third dimension for time).
        z_res_c (float or list of floats): Threshold median residual above
            which a grid cell is considered as part of a channel (one threshold
            value for each neighborhood radius; the same threshold value is
            applied for all neighborhood radius if only one is provided).

    Returns:
        NumPy array (boolean): True for channel grid cells, False otherwise
           (same shape as z_res arrays).
    """

    # Reshape z_res_c, if needed.
    if type(z_res_c) == float:
        z_res_c = [z_res_c] * len(z_res)

    # Initialize channel array.
    chn = np.zeros(z_res[0].shape, dtype = bool)

    # Extract channels.
    for i in range(len(z_res)):
        chn = np.logical_or(chn, z_res[i] > z_res_c[i])

    return chn


################################################################################
# Compute channel polygons. ####################################################
################################################################################

def channel_polygons(x, y, chn, sc = 0):
    """ Compute channel network polygons.

    Args:
        x, y (NumPy arrays): Grid cell coordinates (1D).
        chn (NumPy array, boolean): True for channel grid cells, False
            otherwise (first dimension for x, second dimension for y), for one
            (2D array) or several time steps (3D array, third dimension for
            time).
        sc (float): Threshold polygon surface area, below which polygons are
            disregarded (default to 0).

    Returns:
        MultiPolygon (one time step) or list of MultiPolygons (several time
            steps): Channel network polygons.
    """

    # Number of grid cells.
    nx = chn.shape[0]
    ny = chn.shape[1]

    # Reshape z as a 3D array, if needed.
    if chn.ndim == 2:
        chn = chn.reshape((nx, ny, 1))

    # Number of time steps.
    nt = chn.shape[2]

    # Initialize list of channel network MultiPolygons.
    mpol = []

    # Loop over time step.
    for i in range(nt):

        # Initialize list of polygons.
        pols = []

        # Only compute channel contours if there are channel nodes.
        if np.any(chn[:, :, i]):

            # Channel contours.
            chn_int = chn[:, :, i].astype(int)
            QuadContourSet = plt.contour(x, y, chn_int.T, levels = [.5])

            # Convert to polygons.
            for contour_path in QuadContourSet.collections[0].get_paths():
                xy = contour_path.vertices
                coords = []
                for j in range(xy.shape[0]):
                    coords.append((xy[j, 0], xy[j, 1]))
                pols.append(geometry.Polygon(coords))

            # Sort polygons by surface areas.
            s = [pol.area for pol in pols]
            inds = np.flip(np.argsort(s))
            pols = [pols[ind] for ind in inds]

            # Remove polygons with surface area smaller than smin.
            s = [pol.area for pol in pols]
            for j in range(len(s)):
                if s[j] <= sc:
                    pols = pols[:j]
                    break

            # Insert interiors one by one to avoid inserting interior interiors.
            if len(pols) == 0:
                stop_while = True
            else:
                stop_while = False
            while not stop_while:
                stop_for = False
                for j in range(1, len(pols)):
                    for k in range(j):
                        if pols[j].within(pols[k]):
                            # Update polygon k with interior j.
                            shell = pols[k].exterior.coords
                            holes = []
                            for kk in range(len(pols[k].interiors)):
                                holes.append(pols[k].interiors[kk].coords)
                            holes.append(pols[j].exterior.coords)
                            pols[k] = geometry.Polygon(shell = shell,
                                                       holes = holes)
                            # Delete polygon j.
                            del pols[j]
                            # Stop double for-loop.
                            stop_for = True
                            break
                    if stop_for:
                        break
                # Stop while-loop.
                if j == len(pols) - 1 and k == j - 1:
                    stop_while = True

        # Update list of channel network MultiPolygons.
        mpol.append(geometry.MultiPolygon(pols))

    # Convert mpol into MultiPolygon, if needed.
    if nt == 1:
        mpol = mpol[0]

    return mpol