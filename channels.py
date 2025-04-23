""" Channels.

Extract tidal channels from digital elevation maps (triangular grids).

Author: Olivier Gourgue (University of Antwerp & Boston University)

"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.path as pth
import matplotlib.pyplot as plt
import numpy as np

from shapely import geometry


################################################################################
# Compute mini-clouds. #########################################################
################################################################################

def mini_clouds(x, y, r):
    """ Compute mini-clouds.

    A mini-cloud is the list of neighboring nodes within a certain radius of
    each node.

    Args:
        x, y (NumPy arrays): Node coordinates.
        r (float): Neighborhood radius.

    Returns:
        List of NumPy arrays: Mini-cloud node indices for each node.

    """

    # Number of nodes.
    n = len(x)

    # Integer type.
    if n <= 127:
        dtype = np.int8
    elif n <= 32767:
        dtype = np.int16
    elif n <= 2147483647:
        dtype = np.int32
    else:
        dtype = np.int64

    # Initialize list of mini-clouds.
    clouds = [None] * n

    # Loop on nodes.
    for i in range(n):

        # Node coordinates.
        x0 = x[i]
        y0 = y[i]

        # Nodes within a square bounding box of length 2r.
        tmp = (x >= x0 - r) * (x <= x0 + r) * (y >= y0 - r) * (y <= y0 + r)
        tmp_x = x[tmp]
        tmp_y = y[tmp]

        # Nodes within a circle of radius r.
        d2 = np.zeros(n)
        d2[tmp] = (tmp_x - x0) * (tmp_x - x0) + (tmp_y - y0) * (tmp_y - y0)
        tmp *= (d2 <= r * r)
        ind = np.array(np.argwhere(tmp).reshape(-1), dtype = dtype)

        # Add node indices to mini-cloud.
        clouds[i] = ind

    return clouds


################################################################################
# Export mini-clouds. ##########################################################
################################################################################

def export_mini_clouds(clouds, filename):
    """ Export the list of mini-clouds (binary file).

    Args:
        clouds (list of NumPy arrays): mini-cloud node indices for each node.
        filename (str): File name.
    """

    # Number of mini-clouds.
    n = len(clouds)

    # Integer type.
    if n <= 127:
        dtype = np.int8
    elif n <= 32767:
        dtype = np.int16
    elif n <= 2147483647:
        dtype = np.int32
    else:
        dtype = np.int64

    # Open file.
    file = open(filename, 'w')

    # Write number of mini-clouds.
    np.array(n, dtype = int).tofile(file)

    # Loop over mini-clouds.
    for i in range(n):

        # Write number of nodes.
        np.array(len(clouds[i]), dtype = dtype).tofile(file)

        # Write node indices.
        np.array(clouds[i], dtype = dtype).tofile(file)

    # Close file
    file.close()


################################################################################
# Import mini-clouds. ##########################################################
################################################################################

def import_mini_clouds(filename):
    """ Import the list of mini-clouds (binary file).

    Args:
        filename (str): File name.

    Returns:
        List of NumPy arrays: Mini-cloud node indices for each node.
    """

    # Open file.
    file = open(filename, 'r')

    # Number of mini-clouds.
    n = np.fromfile(file, dtype = int, count = 1)[0]

    # Integer type.
    if n <= 127:
        dtype = np.int8
    elif n <= 32767:
        dtype = np.int16
    elif n <= 2147483647:
        dtype = np.int32
    else:
        dtype = np.int64

    # Initialize list of mini-clouds.
    clouds = [None] * n
    for i in range(n):
        clouds[i] = []

    # Loop over mini-clouds.
    for i in range(n):

        # Read number of nodes.
        nn = np.fromfile(file, dtype = dtype, count = 1)[0]

        # Read node indices.
        clouds[i] = np.fromfile(file, dtype = dtype, count = nn)

    # Close file.
    file.close()

    return clouds


################################################################################
# Median neighborhood analysis. ################################################
################################################################################

def mna(clouds, z):
    """ Compute mini-cloud median residuals (m).

    The residual is defined as the difference between the mini-cloud median
    elevation and the elevation.

    Args:
        clouds (list of NumPy arrays): Mini-cloud node indices for each node.
        z (NumPy array): Elevation on each node, for one (1D array) or several
            time steps (2D array, second dimension for time).

    Returns:
        NumPy Array: Median residual within each mini-cloud (same shape as z).
    """

    # Reshape z as a 2D array, if needed.
    if z.ndim == 1:
        z = z.reshape((-1, 1))

    # Number of nodes.
    n = z.shape[0]

    # Number of time steps.
    nt = z.shape[1]

    # Mini-cloud medians.
    z_med = np.zeros(z.shape)
    for i in range(n):
        z_med[i, :] = np.median(z[clouds[i]], axis = 0)

    # Mini-cloud residuals.
    z_res = z_med - z

    # Reshape z_res as a 1D array, if needed.
    if nt == 1:
        z_res = z_res.reshape(-1)

    return z_res


################################################################################
# Extract channels. ############################################################
################################################################################

def channels(z_res, z_res_c):
    """ Extract channel nodes.

    Args:
        z_res (list of NumPy arrays): Median residuals for different
            neighborhood radius, for one (1D arrays) or several time steps (2D
            arrays, second dimension for time).
        z_res_c (float or list of floats): Threshold median residual above
            which a node is considered as part of a channel (one threshold value
            for each neighborhood radius; the same threshold value is applied
            for all neighborhood radius if only one is provided).

    Returns:
        NumPy array (boolean): True for channel nodes, False otherwise (same
            shape as z_res arrays).
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
# Separate channel segments ####################################################
################################################################################

def split_contours(paths, threshold=100):
    """ Splits contour paths into separate polygon segments based on a distance threshold.

    This function is necessary for Matplotlib v3.8 or greater, where contour paths of the 
    same level are automatically connected into a single path. It detects the undesired 
    connections between distant channel contours, based on a given threshold distance 
    (default = 100), and separates the segments.

    Args:
        paths (List): A list of Matplotlib contour paths, each containing a 
            sequence of vertices.
        threshold (float, optional): The distance threshold to determine disconnections 
            between contour segments. If the distance between consecutive points exceeds 
            this value, a new segment is started. Default is 100.

    Returns:
        polygons (List): A list of raw channel network polygons.
    """
    polygons = []

    for contour_path in paths:
        contour = contour_path.vertices
        segments = []  # Store individual segments
        current_segment = [contour[0]]
        
        # Loop through the contour path and detect incorrectly connected segments
        for i in range(1, len(contour)):
            # Check distance between two consecutive points against the threshold
            distance = np.linalg.norm(contour[i] - contour[i-1])
            if distance > threshold:
                if len(current_segment) > 1:
                    segments.append(current_segment)
                current_segment = [contour[i]]
            else:
                current_segment.append(contour[i])
        
        # Append the last segment
        if len(current_segment) > 1:
            segments.append(current_segment)
        
        # Convert each segment to a separate Polygon
        for segment in segments:
            polygons.append(geometry.Polygon(segment))

    return polygons

################################################################################
# Compute channel polygons. ####################################################
################################################################################

def channel_polygons(x, y, tri, chn, sc = 0):
    """ Compute channel network polygons.

    Args:
        x, y (NumPy arrays): Node coordinates.
        tri (NumPy array): For each triangle, the indices of the three points
            that make up the triangle, ordered in an anticlockwise manner.
        chn (NumPy array, boolean): True for channel nodes, False otherwise, for
            one (1D array) or several time steps (2D array, second dimension for
            time).
        sc (float): Threshold polygon surface area, below which polygons are
            disregarded (default to 0).

    Returns:
        MultiPolygon (one time step) or list of MultiPolygons (several time
            steps): Channel network polygons.
    """

    # Reshape chn as a 2D array, if needed.
    if chn.ndim == 1:
        chn = chn.reshape((-1, 1))

    # Number of time steps.
    nt = chn.shape[1]

    # Initialize list of channel network MultiPolygons.
    mpol = []

    # Loop over time step.
    for i in range(nt):

        # Initialize list of polygons.
        pols = []

        # Only compute channel contours if there are channel nodes.
        if np.any(chn[:, i]):

            # Channel contours.
            chn_int = chn[:, i].astype(int)
            TriContourSet = plt.tricontour(x, y, tri, chn_int, levels = [.5])

            # Check matplotlib version for tricontour (changes in versions > 3.8)
            version_mpl = tuple(map(int,(matplotlib.__version__).split('.'))) 

            if version_mpl > (3, 8, 0):
                # Split continuous contours and convert to polygons (matplotlib >3.8)
                pols = split_contours(TriContourSet.get_paths())
            else:
                # Convert to polygons (original version, matplotlib <3.8)
                for contour_path in TriContourSet.collections[0].get_paths():
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


################################################################################
# Nodes-in-channel-polygons check.#############################################
################################################################################

def nodes_in_channel_polygons(x, y, mpol):
    """ Check if nodes are within channel polygons.

    Args:
        x, y (NumPy arrays): Node coordinates.
        mpol (MultiPolygon or list of MultiPolygons): Channel network polygons. Multipolygon if one time step, list of MultiPolygons if several time
            steps.

    Returns:
        NumPy array (boolean): True for channel nodes, False otherwise. 1D array if one time step, 2D array (second dimension for time) if several
            time steps.
    """

    # Convert mpol into list, if needed.
    if type(mpol) != list:
        mpol = [mpol]

    # Number of nodes.
    n = x.shape[0]

    # Number of time steps.
    nt = len(mpol)

    # Node array.
    nodes = np.zeros((n, 2))
    nodes[:, 0] = x
    nodes[:, 1] = y

    # Initialize channel array.
    chn = np.zeros((n, nt), dtype = bool)

    # Loop over time steps.
    for i in range(nt):

        # Convert MultiPolygon exteriors into list of Matplotlib paths.
        paths_ext = []
        for pol in mpol[i].geoms:
            vertices = np.zeros((len(pol.exterior.coords) + 1, 2))
            vertices[:-1, :] = np.array(pol.exterior.coords)
            vertices[-1, :] = vertices[0, :]
            paths_ext.append(pth.Path(vertices, closed = True))

        # Convert MultiPolygon interiors into list of Matplotlib paths.
        paths_int = []
        for pol in mpol[i].geoms:
            for interior in pol.interiors:
                vertices = np.zeros((len(interior.coords) + 1, 2))
                vertices[:-1, :] = np.array(interior.coords)
                vertices[-1, :] = vertices[0, :]
                paths_int.append(pth.Path(vertices, closed = True))

        # Nodes inside Polygon exteriors.
        for path in paths_ext:
            chn[:, i] = np.logical_or(chn[:, i], path.contains_points(nodes))

        # Remove nodes inside Polygons interiors.
        for path in paths_int:
            ind = path.contains_points(nodes)
            chn[ind, i] = False

    # Reshape chn as a 1D array, if needed.
    if nt == 1:
        chn = chn.reshape(-1)

    return chn