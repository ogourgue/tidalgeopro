""" Cross-sections.

Compute tidal channel cross-sections and related geometric characteristics.

Author: Olivier Gourgue (University of Antwerp & Boston University).

"""

import numpy as np
from scipy import interpolate

from shapely import geometry


################################################################################
# Cross-sections. ##############################################################
################################################################################

def cross_sections(skl_xy, skl_sections, mpol, n0 = 1, n1 = np.inf, ratio = 1):
    """ Compute cross-sections along the channel network skeleton.

    The normal directions are determined based on the skeleton direction
    averaged over (2 * n + 1) points. The algorithm starts with n = n0. The
    procedure is repeated for increasing values of n <= n1 for intersecting
    cross-sections. Remaining intersecting cross-sections are disregarded.

    Args:
        skl_xy (NumPy array): Skeleton point coordinates.
        skl_sections (NumPy array): Skeleton section indices at skeleton points.
        mpol (MultiPolygon): Tidal channel network.
        n0 (int): Minimum number of neighboring points (forward and backward) to
            determine the skeleton normal direction (default to 1).
        n1 (int): Maximum number of neighboring points (forward and backward) to
            determine the skeleton normal direction (default to infinity).
        ratio (float): Threshold ratio between cross-section width and channel
            width (computed as twice the distance between skeleton and channel
            banks) over which cross-sections are disregarded (default to 1).

    Returns:
        NumPy array: Cross-section coordinates (x0, y0, x1, y1) along the
            skeleton (NaN values for disregarded cross-sections).
    """

    # Convert channel MultiPolygon into MultiLineString allowing calculating
    # intersections with potential cross-sections.
    lss = []
    for pol in mpol.geoms:
        lss.append(geometry.LineString(pol.exterior.coords))
        for interior in pol.interiors:
            lss.append(interior.coords)
    mls = geometry.MultiLineString(lss)

    # Channel width.
    width = channel_width(skl_xy, mpol)

    # Initialize list of cross-sections.
    css = []


    ###################################################################
    # Determine non-intersecting cross-sections per skeleton section. #
    ###################################################################

    # Loop over skeleton sections.
    for i in np.unique(skl_sections):

        # Number of points.
        nbp = np.sum(skl_sections == i)

        # A skeleton section with only 1 point is disregarded.
        if nbp == 1:
            css_loc = [None]

        else:

            # Local arrays.
            xy_loc = skl_xy[skl_sections == i]
            width_loc = width[skl_sections == i]

            # Initialize number of points to calculate normal directions.
            n = n0

            # Initialize list of cross-section indices to recalculate normal
            # direction.
            ind = list(range(nbp))

            # Calculate cross-sections and increase number of points to
            # calculate normal directions as long as cross-sections intersect
            # each other or if maximum number of points is reached.
            while n <= n1 or len(ind) > 0:

                # Cross sections with n points to calculate normal directions.
                css_loc_n = cross_sections_loc(xy_loc, mls, width_loc, n, ratio)

                # Update list of cross-sections.
                if n == n0:
                    css_loc = css_loc_n
                else:
                    for j in ind:
                        css_loc[j] = css_loc_n[j]

                # Indices of intersecting cross-sections.
                ind = []
                for j in range(nbp):
                    cs = css_loc[j]
                    if cs is not None:
                        others = css_loc[:j] + css_loc[j + 1:]
                        others = list(filter(None, others))
                        if len(others) > 0:
                            if cs.intersects(geometry.MultiLineString(others)):
                                ind.append(j)

                # Update number of points to calculate normal directions.
                n += 1

            # Remove remaining cross-sections intersecting each other.
            for j in ind:
                css_loc[j] = None

        # Update list of cross-sections.
        css += css_loc


    ##########################################################################
    # Filter cross-sections intersecting those from other skeleton sections. #
    ##########################################################################

    # Initialize intersection table (ith entry is the list of cross-section
    # indices intersecting ith cross-section).
    tab = []
    for i in range(len(css)):
        tab.append([])

    # Compute intersection table.
    for i in range(len(css)):
        for j in range(len(css)):
            if css[i] is not None and css[j] is not None and i != j:
                if css[i].intersects(css[j]):
                    tab[i].append(j)

    # Number of intersections for each cross-section.
    ni = np.zeros(len(css), dtype = int)
    for i in range(len(css)):
        ni[i] = len(tab[i])

    # As long as the total number of intersections is higher than zero
    while np.sum(ni) > 0:

        # For each loop, we deal with cross-sections with highest number of
        # intersections and we get rid of them if:
        # 1) They don't intersect other cross-sections with highest number of
        #    intersections (because that means that they only intersect
        #    cross-sections with lower number of intersections, which we decide
        #    to favor).
        # 2) Their ratio between cross-section length and channel width is
        #    higher than other cross-sections with highest number of
        #    intersections that they intersect.

        # Indices of cross-sections with highest number of intersections
        ind = np.argwhere(ni == np.max(ni)).flatten()

        # Loop over cross-sections with highest number of intersections.
        for i in ind:

            # Indices of cross-section with highest number of intersections
            # that it intersects.
            inter = np.intersect1d(tab[i], ind)

            # Rule 1: Disregard cross-section if it does not intersect other
            # cross-sections with same highest number of intersections.
            if len(inter) == 0:

                # Disregard cross-section.
                css[i] = None

                # Update intersection table.
                for j in tab[i]:
                    tab[j].remove(i)
                tab[i] = []

            # Rule 2: Disregard cross-section if ratio between cross-section
            # length and channel width is higher than other cross-sections with
            # same highest number of intersections.
            else:

                # Check if that is the cross-section with highest ratio.
                disregard_bool = True
                for j in inter:
                    if css[i].length / width[i] < css[j].length / width[j]:
                        disregard_bool = False

                # If yes.
                if disregard_bool:

                    # Disregard cross-section.
                    css[i] = None

                    # Update intersection table.
                    for j in tab[i]:
                        tab[j].remove(i)
                    tab[i] = []

        # Update number of intersections for each cross-section.
        for i in range(len(css)):
            ni[i] = len(tab[i])


    #######################################################
    # Convert cross-section LineStrings into NumPy array. #
    #######################################################

    cross_sections = np.zeros((len(css), 4)) + np.nan
    for i in range(len(css)):
        if css[i] is not None:
            cross_sections[i, 0] = css[i].xy[0][0]
            cross_sections[i, 1] = css[i].xy[1][0]
            cross_sections[i, 2] = css[i].xy[0][1]
            cross_sections[i, 3] = css[i].xy[1][1]

    return cross_sections


################################################################################
# Channel width. ###############################################################
################################################################################

def channel_width(skl_xy, mpol):
    """ Compute the channel width along the skeleton.

    The channel width is here defined as twice the distance between the skeleton
    and the channel banks.

    Args:
        skl_xy (NumPy array): Skeleton point coordinates.
        mpol (MultiPolygon): Tidal channel network.

    Returns:
        NumPy array: Channel width at skeleton points.
    """

    # Initialize.
    width = np.zeros(skl_xy.shape[0])

    # Compute channel width.
    for i in range(len(width)):
        point = geometry.Point(skl_xy[i, :])
        width[i] = point.distance(mpol.boundary) * 2

    return width


################################################################################
# Local cross-sections. ########################################################
################################################################################

def cross_sections_loc(skl_xy, mls, width, n, ratio):
    """ Compute cross-sections on a single skeleton section.

    The normal directions are determined based on the skeleton direction
    averaged over (2 * n + 1) points.

    Args:
        skl_xy (NumPy array): Skeleton point coordinates.
        mls (MultiLineString): Tidal channel network.
        width (NumPy array): Channel width at skeleton points.
        n (int): Number of neighboring points (forward and backward) to
            determine the skeleton normal direction.
        ratio (float): Threshold ratio between cross-section width and channel
            width (computed as twice the distance between skeleton and channel
            banks) over which cross-sections are disregarded.

    Returns:
        List of 2-point LineStrings (cross-sections) or None (disregarded cross-sections).
    """

    # Initialize list of cross-sections.
    css = []

    # Number of skeleton points.
    nbp = skl_xy.shape[0]

    # Loop over skeleton points.
    for i in range(nbp):

        # Skeleton point coordinates and corresponding Point.
        x = skl_xy[i, 0]
        y = skl_xy[i, 1]
        p = geometry.Point(x, y)

        # Direction perpendicular to skeleton.
        if nbp < 2 * n + 1:
            x0 = skl_xy[0, 0]
            y0 = skl_xy[0, 1]
            x1 = skl_xy[-1, 0]
            y1 = skl_xy[-1, 1]
        elif i < n:
            x0 = skl_xy[0, 0]
            y0 = skl_xy[0, 1]
            x1 = skl_xy[2 * n, 0]
            y1 = skl_xy[2 * n, 1]
        elif i > nbp - n - 1:
            x0 = skl_xy[-2 * n - 1, 0]
            y0 = skl_xy[-2 * n - 1, 1]
            x1 = skl_xy[-1, 0]
            y1 = skl_xy[-1, 1]
        else:
            x0 = skl_xy[i - n, 0]
            y0 = skl_xy[i - n, 1]
            x1 = skl_xy[i + n, 0]
            y1 = skl_xy[i + n, 1]
        nx = y0 - y1
        ny = x1 - x0
        norm = ((nx ** 2) + (ny ** 2)) ** .5
        nx /= norm
        ny /= norm

        # Potential cross-section.
        ls = geometry.LineString([(x - nx * width[i] * ratio,
                                   y - ny * width[i] * ratio),
                                  (x + nx * width[i] * ratio,
                                   y + ny * width[i] * ratio)])

        # Intersections between potential cross-section and channel banks.
        mp = ls.intersection(mls)

        # Disregard cross-section if only 0 or 1 intersections.
        if mp.geom_type in ['Point', 'LineString']:
            css.append(None)

        elif mp.geom_type == 'MultiPoint':

            # If more than 2 intersections, keep 2 closest to skeleton point.
            if len(mp.geoms) > 2:
                dist = np.zeros(len(mp.geoms))
                for j in range(len(mp.geoms)):
                    dist[j] = p.distance(mp.geoms[j])
                ind = np.argsort(dist)
                mp = geometry.MultiPoint([mp.geoms[ind[0]], mp.geoms[ind[1]]])

            # Intersection points.
            p0 = mp.geoms[0]
            p1 = mp.geoms[1]

            # Update potential cross-section.
            ls = geometry.LineString([p0, p1])

            # Disregard cross-section if 2 intersections are on the same side of
            # the skeleton.
            if p.distance(p0) + p.distance(p1) - ls.length > 1e-6:
                css.append(None)

            # Disregard cross-section if ratio between cross-section length and
            # channel width is larger than threshold ratio.
            elif ls.length / width[i] > ratio:
                css.append(None)

            # Otherwise, the potential cross-section is added to the list.
            else:
                css.append(ls)

    return css


################################################################################
# cross-section metrics ########################################################
################################################################################

def metrics(cross_sections, x, y, z, dx, buffer = None):
    """ Compute channel cross-section width, depth and surface area.

    Args:
        cross_sections (NumPy array): Cross-section coordinates (x0, y0, x1, y1)
            along the skeleton (NaN values for disregarded cross-sections).
        x (NumPy array): X-coordinates.
        y (NumPy array): Y-coordinates.
        z (NumPy array): Bottom elevation.
        dx (float): Grid resolution along cross-sections.
        buffer (float): Buffer length around cross-sections to determine a
            bounding box to limit the number of DEM points for interpolation
            (Default to None, no buffer). Optimal value should about 2 to 3
            times the DEM grid resolution (low values might lead to disregard
            useful data points, high values will increase the computational time
            for no gain in accuracy).

    Returns:
        NumPy array: Channel width at skeleton points.
        NumPy array: Channel depth at skeleton points.
        NumPy array: Channel cross-section surface area at skeleton points.
    """

    # Initialize arrays.
    width = np.zeros(cross_sections.shape[0]) + np.nan
    depth = np.zeros(cross_sections.shape[0]) + np.nan
    area = np.zeros(cross_sections.shape[0]) + np.nan

    # Loop over (non-disregarded) cross-sections.
    for i in range(cross_sections.shape[0]):
        if np.isfinite(cross_sections[i, 0]):

            # Cross-section edge coordinates.
            x0 = cross_sections[i, 0]
            y0 = cross_sections[i, 1]
            x1 = cross_sections[i, 2]
            y1 = cross_sections[i, 3]

            # Cross-section LineString.
            cs = geometry.LineString([(x0, y0), (x1, y1)])

            # Curvilinear coordinates along cross-section.
            s0 = np.remainder(cs.length, dx) / 2
            s = np.arange(s0, cs.length, dx)

            # Cross-section interpolation points.
            cs_x = np.zeros(s.shape)
            cs_y = np.zeros(s.shape)
            for j in range(len(s)):
                point = cs.interpolate(s[j])
                cs_x[j] = point.x
                cs_y[j] = point.y

            # Bounding box.
            left = np.minimum(x0, x1) - buffer
            bottom = np.minimum(y0, y1) - buffer
            right = np.maximum(x0, x1) + buffer
            top = np.maximum(y0, y1) + buffer

            # Elevation in the bounding box.
            ind = (x > left) * (x < right) * (y > bottom) * (y < top)
            bb_x = x[ind]
            bb_y = y[ind]
            bb_z = z[ind]

            # Interpolate elevation at cross-section edges.
            z0 = interpolate.griddata((bb_x, bb_y), bb_z, (x0, y0))
            z1 = interpolate.griddata((bb_x, bb_y), bb_z, (x1, y1))

            # Interpolate elevation along cross-section.
            cs_z = interpolate.griddata((bb_x, bb_y), bb_z, (cs_x, cs_y))

            # Compute metrics.
            width[i] = cs.length
            depth[i] = np.clip(.5 * (z0 + z1) - np.min(cs_z), 0, None)
            area[i] = np.sum(np.clip(.5 * (z0 + z1) - cs_z, 0, None)) * dx

    return width, depth, area