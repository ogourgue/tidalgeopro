""" Watersheds.

Compute watershed area, maximum upstream length, and total upstream length along
the skeleton of a tidal channel network.

Author: Olivier Gourgue (University of Antwerp & Boston University)

"""


import numpy as np

################################################################################
# Watersheds. ##################################################################
################################################################################

def watersheds(x, y, node_sections, node_dl, skl_xy, skl_sections, skl_dl,
               mask = None, v = None):
    """Compute watershed area, maximum and total upstream length along skeleton.

    Args:
        x (NumPy array, shape (M)): Structured grid x-coordinates.
        y (NumPy array, shape (N)): Structured grid y-coordinates.
        node_sections (Numpy array): Skeleton section indices at skeleton nodes.
        node_dl (Numpy array): Downstream length at skeleton nodes.
        skl_xy (NumPy array): Skeleton point coordinates.
        skl_sections (NumPy array): Skeleton section indices at skeleton points.
        skl_dl (Numpy array): Downstream length at skeleton points.
        mask (NumPy array, bool, shape (M, N)): True outside area of interest
            (default to None, that is, False everywhere).
        v (list of Numpy arrays, shape (M, N)): Variables to integrate along the
            skeleton over the local watershed surface (default to None, that is,
            no variable to integrate).

    Returns:
        NumPy array: Watershed surface area at skeleton points.
        NumPy array: Maximum upstream length at skeleton points.
        NumPy array: Total upstream length at skeleton points.
        NumPy array: Optional variables integrated at skeleton points over the
            local watershed surface (first 2 dimensions as v, third dimension for number of variables).
    """

    # Number of grid cells.
    nx = x.shape[0]
    ny = y.shape[0]

    # Initialize mask, if needed.
    if mask is None:
        mask = np.zeros((nx, ny), dtype = bool)

    # Initialize v, if needed.
    if v is None:
        v = np.zeros((nx, ny, 0))

    # Reshape v, if needed.
    if v.ndim == 2:
        v = v.reshape((nx, ny, 1))

    # Number of variables to integrate.
    nv = v.shape[2]

    # Number of skeleton nodes.
    nn = node_dl.shape[0]

    # Number of skeleton points.
    ns = skl_dl.shape[0]

    # Grid cell surface area.
    ds = (x[1] - x[0]) * (y[1] - y[0])

    # Skeleton point coordinates.
    skl_x = skl_xy[:, 0]
    skl_y = skl_xy[:, 1]

    # Calculate watershed strip area (i.e., surface area corresponding to grid
    # cells closer to one give skeleton point) and watershed strip integrals.
    skl_wsa = np.zeros(ns)
    skl_wsi = np.zeros((ns, nv))
    for i in range(nx):
        for j in range(ny):
            if not mask[i, j]:
                # Square distance to skeleton points.
                d2 = (x[i] - skl_x) ** 2 + (y[j] - skl_y) ** 2
                # Index of minimum distance.
                skl_wsa[np.argmin(d2)] += ds
                for k in range(nv):
                    skl_wsi[np.argmin(d2), k] += v[i, j, k] * ds

    # Calculate local watershed area (i.e., watershed area per skeleton section)
    # and local watershed integrals on skeleton nodes.
    node_lwa = np.zeros(nn)
    node_lwi = np.zeros((nn, nv))
    for i in range(len(node_sections)):
        # Only non-empty sections.
        if np.sum(skl_sections == i) > 0:
            # Skeleton section point indices.
            ind = (skl_sections == i)
            # Downstream node: local watershed area is the sum of watershed
            # strip area of all skeleton section points. The result is cumulated
            # at confluence points.
            node_lwa[node_sections[i, 0]] += np.sum(skl_wsa[ind])
            for j in range(nv):
                node_lwi[node_sections[i, 0], j] += np.sum(skl_wsi[ind, j])
            # Upstream node (if channel head): local watershed area is watershed
            # strip area of most upstream skeleton section point.
            if np.sum(node_sections == node_sections[i, 1]) == 1:
                node_lwa[node_sections[i, 1]] = skl_wsa[ind][-1]
                for j in range(nv):
                    node_lwi[node_sections[i, 1], j] = skl_wsi[ind, j][-1]
            # Upstream node (if split point): local watershed area is watershed
            # strip area of most upstream skeleton section point. The split
            # point is the upstream node in two channel sections. Watershed
            # strip area is shared by most upstream skeleton point of both
            # channel sections and must be cumulated to obtain local watershed
            # area.
            if np.sum(node_sections == node_sections[i, 1]) == 2:
                node_lwa[node_sections[i, 1]] += skl_wsa[ind][-1]
                for j in range(nv):
                    node_lwi[node_sections[i, 1], j] += skl_wsi[ind, j][-1]

    # Calculate metrics on skeleton nodes.
    # Initialize watershed area (integrals) to local watershed area (integrals).
    node_wa = node_lwa
    node_wi = node_lwi
    # Initialize maximum and total upstream lengths to zero.
    node_mul = np.zeros(nn)
    node_tul = np.zeros(nn)
    # Loop over skeleton sections in decreasing order of their upstream node
    # downstream length.
    for i in np.argsort(-node_dl[node_sections[:, 1]]):
        # Channel section nodes.
        n0 = node_sections[i, 0]
        n1 = node_sections[i, 1]
        # If not a channel head, propagate upstream watershed area (integrals)
        # downstream.
        if np.sum(node_sections == n1) > 1:
            node_wa[n0] += node_wa[n1]
            for j in range(nv):
                node_wi[n0, j] += node_wi[n1, j]
        # Propagate maximum upstream length downstream.
        mul = node_mul[n1] + node_dl[n1] - node_dl[n0]
        if mul > node_mul[n0]:
            node_mul[n0] = mul
        # Propagate total upstream length downstream.
        tul = node_tul[n1] + node_dl[n1] - node_dl[n0]
        node_tul[n0] += tul

    # Divide integrals by watershed area.
    node_v = np.zeros((nn, nv))
    for i in range(nv):
        ind = node_wa > 0
        node_v[ind, i] = node_wi[ind, i] / node_wa[ind]

    # Calculate metrics on skeleton points.
    skl_wa = np.zeros(ns)
    skl_wi = np.zeros((ns, nv))
    skl_mul = np.zeros(ns)
    skl_tul = np.zeros(ns)
    for i in np.unique(skl_sections):
        # Skeleton point indices (upstream to downstream).
        points = np.flip(np.argwhere(skl_sections == i)[:, 0])
        for j in range(len(points)):
            # Skeleton point index.
            n = points[j]
            # Upstream channel section node index.
            n1 = node_sections[i, 1]
            skl_wa[n] = node_wa[n1] + np.sum(skl_wsa[points[:j]])
            for k in range(nv):
                skl_wi[n, k] = node_wi[n1, k] + np.sum(skl_wsi[points[:j], k])
            skl_mul[n] = node_mul[n1] + node_dl[n1] - skl_dl[n]
            skl_tul[n] = node_tul[n1] + node_dl[n1] - skl_dl[n]

    # Divide integrals by watershed area.
    skl_v = np.zeros((ns, nv))
    for i in range(nv):
        ind = skl_wa > 0
        skl_v[ind, i] = skl_wi[ind, i] / skl_wa[ind]

    # Reshape node_v and skl_v, if needed.
    if nv == 1:
        node_v = node_v.reshape(-1)
        skl_v = skl_v.reshape(-1)

    if nv == 0:
        return node_wa, node_mul, node_tul, skl_wa, skl_mul, skl_tul
    else:
        return (node_wa, node_mul, node_tul, node_v, skl_wa, skl_mul, skl_tul,
                skl_v)