import numpy as np
from scipy.interpolate import griddata
from shapely.geometry import Point, LineString, MultiPoint, MultiLineString



################################################################################
# channel width ################################################################
################################################################################

def channel_width(coords, mpol):

  """ Compute channel width along skeleton, as twice the distance between the skeleton and the channel edges.

  !!! Attention !!!
  This is the distance to the closest point defining the channel edges. A more accurate calculation can be found in the channelwidth module. The present function gives satisfactory results for high resolution channel edges.

  Required parameters:
  coords (Numpy array of shape (n, 2)): coordinates of the skeleton points
  mpol (MultiPolygon): structure of multiple Polygons describing channel edges

  Returns:
  Numpy array of shape (n): channel width at each skeleton point

  """

  # initialize
  width = np.zeros(coords.shape[0])

  # compute channel width
  for i in range(len(width)):
    point = Point(coords[i, :])
    width[i] = point.distance(mpol.boundary) * 2

  return width


################################################################################
# local cross-sections #########################################################
################################################################################

def cross_sections_loc(coords, mls, width, ratio = 1, n = 1):

  """ Compute cross-sections on a single skeleton section (no confluence branches). The normal directions are determined based on the skeleton direction averaged over (2 * n + 1) points.

  Required parameters:
  coords (Numpy array of shape (m, 2)): coordinates of the skeleton points
  mls (MultiLineString): structure of multiple LineStrings describing channel edges
  width (Numpy array of shape (m)): channel width at each skeleton point, defined as twice the shortest distance between skeleton and channel edges

  Optional parameters:
  ratio (float, default = 1): threshold ratio (cross-section length / channel width): cross-sections with large ratio are disregarded
  n (int, default = 1): number of neighboring points (forward and backward) to determine normal direction of skeleton

  Returns:
  List of 2-point LineStrings (cross-sections) or None (disregarded cross-sections)

  """

  # initialize list of cross-sections
  css = []

  # number of points
  nbp = coords.shape[0]

  # for each point
  for i in range(nbp):

    # point coordinates and corresponding Point
    x = coords[i, 0]
    y = coords[i, 1]
    p = Point(x, y)

    # direction perpendicular to skeleton
    if nbp < 2 * n + 1:
      x0 = coords[0, 0]
      y0 = coords[0, 1]
      x1 = coords[-1, 0]
      y1 = coords[-1, 1]
    elif i < n:
      x0 = coords[0, 0]
      y0 = coords[0, 1]
      x1 = coords[2 * n, 0]
      y1 = coords[2 * n, 1]
    elif i > nbp - n - 1:
      x0 = coords[-2 * n - 1, 0]
      y0 = coords[-2 * n - 1, 1]
      x1 = coords[-1, 0]
      y1 = coords[-1, 1]
    else:
      x0 = coords[i - n, 0]
      y0 = coords[i - n, 1]
      x1 = coords[i + n, 0]
      y1 = coords[i + n, 1]
    nx = y0 - y1
    ny = x1 - x0
    norm = ((nx ** 2) + (ny ** 2)) ** .5
    nx /= norm
    ny /= norm

    # potential cross-section
    ls = LineString([(x - nx * width[i] * ratio, y - ny * width[i] * ratio),
                     (x + nx * width[i] * ratio, y + ny * width[i] * ratio)])

    # intersections between potential cross-section and channel edges
    mp = ls.intersection(mls)

    # disregard cross-section if only 0 or 1 intersections
    if mp.geom_type in ['Point', 'LineString']:
      css.append(None)

    elif mp.geom_type == 'MultiPoint':

      # if more than 2 intersections, keep the 2 closest to skeleton point
      if len(mp.geoms) > 2:
        dist = np.zeros(len(mp.geoms))
        for j in range(len(mp.geoms)):
          dist[j] = p.distance(mp.geoms[j])
        ind = np.argsort(dist)
        mp = MultiPoint([mp.geoms[ind[0]], mp.geoms[ind[1]]])

      # intersection points
      p0 = mp.geoms[0]
      p1 = mp.geoms[1]

      # update potential cross-section
      ls = LineString(mp)

      # disregard cross-section if 2 intersections are on the same side of the
      # skeleton
      if p.distance(p0) + p.distance(p1) - ls.length > 1e-6:
        css.append(None)

      # disregard cross-section if ratio between cross-section length and
      # channel width (as twice the distance to channel edges) is larger than
      # the threshold ratio
      elif ls.length / width[i] > ratio:
        css.append(None)

      # otherwise, potential cross-section is added to list of cross-sections
      else:
        css.append(ls)

  return css


################################################################################
# cross-sections ###############################################################
################################################################################

def cross_sections(coords, sections, mpol, n_min = 1, n_max = np.inf,
                   ratio = 1):

  """ Compute cross-sections along the channel network skeleton. The normal directions are determined based on the skeleton direction averaged over (2 * n + 1) points. The algorithm starts with n = n_min. The procedure is repeated for increasing values of n <= n_max for intersecting cross-sections. Remaining intersecting cross-sections are disregarded.

  Required parameters:
  coords (Numpy array of shape (m, 2)): coordinates of the skeleton points
  sections (Numpy array of shape (m)): section index of each skeleton point
  mpol (MultiPolygon): structure of multiple Polygons describing channel edges

  Optional parameters:
  n_min (int, default = 1): minimum number of neighboring points (forward and backward) to determine normal direction of skeleton
  n_max (int, default = inf): maximum number of neighboring points (forward and backward) to determine normal direction of skeleton
  ratio (float, default = 1): threshold ratio (cross-section length / channel width): cross-sections with large ratio are disregarded

  Returns:
  NumPy array of shape (m, 4): cross-section coordinates (x0, y0, x1, y1) along the skeleton (np.nan values for disregarded cross-sections)

  """

  # channel edges: convert MultiPolygon into MultiLineString
  # necessary to calculate intersections with potential cross-sections
  channel_edges = []
  for pol in mpol.geoms:
    channel_edges.append(LineString(pol.exterior.coords))
    for interior in pol.interiors:
      channel_edges.append(interior.coords)
  channel_edges = MultiLineString(channel_edges)

  # channel width (as twice the distance between skeleton and channel edges)
  width = channel_width(coords, mpol)

  # initialize list of cross-sections
  css = []


  ##################################################################
  # determine non-intersecting cross-sections per skeleton section #
  ##################################################################

  # for each skeleton section
  for i in np.unique(sections):

    # number of points
    nbp = np.sum(sections == i)

    # a skeleton section with only 1 point is disregarded
    if nbp == 1:
      css_loc = [None]

    else:

      # local arrays
      coords_loc = coords[sections == i]
      width_loc = width[sections == i]

      # initialize number of points to calculate normal directions
      n = n_min

      # initialize list of cross-section indices to recalculate normal direction
      ind = list(range(nbp))

      # calculate cross-sections and increase number of points to calculate
      # normal directions as long as cross-sections intersect each other or
      # if maximum number of points is reached
      while n <= n_max or len(ind) > 0:

        # cross sections with n points to calculate normal directions
        css_loc_n = cross_sections_loc(coords_loc, channel_edges, width_loc,
                                       ratio, n)

        # update list of cross-sections
        if n == n_min:
          css_loc = css_loc_n
        else:
          for j in ind:
            css_loc[j] = css_loc_n[j]

        # indices of intersecting cross-sections
        ind = []
        for j in range(nbp):
          cs = css_loc[j]
          if cs is not None:
            others = css_loc[:j] + css_loc[j + 1:]
            others = list(filter(None, others))
            if len(others) > 0:
              if cs.intersects(MultiLineString(others)):
                ind.append(j)

        # update number of points to calculate normal directions
        n += 1

      # remove remaining cross-sections intersecting each other
      for j in ind:
        css_loc[j] = None

    # update list of cross-sections
    css += css_loc


  #########################################################################
  # filter cross-sections intersecting those from other skeleton sections #
  #########################################################################

  # initialize intersection table (tab)
  # tab[i] is the list of cross-section indices intersecting ith cross-section
  tab = []
  for i in range(len(css)):
    tab.append([])

  # compute intersection table
  for i in range(len(css)):
    for j in range(len(css)):
      if css[i] is not None and css[j] is not None and i != j:
        if css[i].intersects(css[j]):
          tab[i].append(j)

  # number of intersection for each cross-section
  ni = np.zeros(len(css), dtype = int)
  for i in range(len(css)):
    ni[i] = len(tab[i])

  # as long as the total number of intersections is higher than zero
  while np.sum(ni) > 0:

    # for each loop, we deal with cross-sections with highest number of
    # intersections and we get rid of them if
    # 1. they don't intersect other cross-sections with highest number of
    # intersections (because that means that they only intersect cross-sections
    # with lower number of intersections, which we decide to favor)
    # 2. their ratio cross-section length / channel width is higher than other
    # cross-sections with highest number of intersections they intersect

    # indices of cross-sections with highest number of intersections
    ind = np.argwhere(ni == np.max(ni)).flatten()

    # for each cross-section with highest number of intersections
    for i in ind:

      # indices of cross-sections with highest number of intersections that it
      # intersects
      inter = np.intersect1d(tab[i], ind)

      # rule 1: disregard cross-section if it does not intersect other
      # cross-sections with same highest number of intersections
      if len(inter) == 0:

        # disregard cross-section
        css[i] = None

        # update intersection table
        for j in tab[i]:
          tab[j].remove(i)
        tab[i] = []

      # rule 2: disregard cross-section if ratio cross-section length / channel
      # width is higher than other cross-sections with same highest number of
      # intersections
      else:

        # check if that is the cross-section with highest ratio
        disregard_bool = True
        for j in inter:
          if css[i].length / width[i] < css[j].length / width[j]:
            disregard_bool = False

        # if yes
        if disregard_bool:

          # disregard cross-section
          css[i] = None

          # update intersection table
          for j in tab[i]:
            tab[j].remove(i)
          tab[i] = []

    # update number of intersections for each cross-section
    for i in range(len(css)):
      ni[i] = len(tab[i])


  ######################################################
  # convert cross-section LineStrings into NumPy array #
  ######################################################

  cross_sections = np.zeros((len(css), 4)) + np.nan
  for i in range(len(css)):
    if css[i] is not None:
      cross_sections[i, 0] = css[i].xy[0][0]
      cross_sections[i, 1] = css[i].xy[1][0]
      cross_sections[i, 2] = css[i].xy[0][1]
      cross_sections[i, 3] = css[i].xy[1][1]

  return cross_sections


################################################################################
# cross-section metrics ########################################################
################################################################################

def metrics(cross_sections, x, y, z, dx, buffer = None):

  """ Compute cross-section metrics (width, depth and surface area) along the skeleton, based on cross-sections coordinates and an unstructured-grid DEM.

  Required parameters:
  cross_sections (Numpy array of shape (n, 4)): cross-section coordinates (x0, y0, x1, y1) along the skeleton
  x, y (NumPy arrays of shape (m)): coordinates of the DEM
  z (NumPy array of shape (m)): bottom elevation
  dx (float): cross-section spatial interpolation step

  Optional parameters:
  buffer (float, default = None): buffer length around cross-sections determining a bounding box to limit the number of DEM points for interpolation (optimal value should about 2-3 times the DEM grid resolution - lower value might lead to disregard useful data points, higher value will increase the computational time for no gain in accuracy)

  Returns
  Numpy array of shape (n): cross-section width along the skeleton
  Numpy array of shape (n): cross-section depth along the skeleton
  Numpy array of shape (n): cross-section surface area along the skeleton

  """

  # initialize arrays
  width = np.zeros(cross_sections.shape[0]) + np.nan
  depth = np.zeros(cross_sections.shape[0]) + np.nan
  area = np.zeros(cross_sections.shape[0]) + np.nan

  # for each (non-disregarded) cross-section
  for i in range(cross_sections.shape[0]):
    if np.isfinite(cross_sections[i, 0]):

      # cross-section edge coordinates
      x0 = cross_sections[i, 0]
      y0 = cross_sections[i, 1]
      x1 = cross_sections[i, 2]
      y1 = cross_sections[i, 3]

      # cross-section LineString
      cs = LineString([(x0, y0), (x1, y1)])

      # curvilinear coordinates along cross-section
      s0 = np.remainder(cs.length, dx) / 2
      s = np.arange(s0, cs.length, dx)

      # cross-section interpolation points
      cs_x = np.zeros(s.shape)
      cs_y = np.zeros(s.shape)
      for j in range(len(s)):
        point = cs.interpolate(s[j])
        cs_x[j] = point.x
        cs_y[j] = point.y

      # bounding box
      left = np.minimum(x0, x1) - buffer
      bottom = np.minimum(y0, y1) - buffer
      right = np.maximum(x0, x1) + buffer
      top = np.maximum(y0, y1) + buffer

      # DEM in the bounding box
      ind = (x > left) * (x < right) * (y > bottom) * (y < top)
      bb_x = x[ind]
      bb_y = y[ind]
      bb_z = z[ind]

      # interpolate elevation at cross-section edges
      z0 = griddata((bb_x, bb_y), bb_z, (x0, y0))
      z1 = griddata((bb_x, bb_y), bb_z, (x1, y1))

      # interpolate elevation along cross-section
      cs_z = griddata((bb_x, bb_y), bb_z, (cs_x, cs_y))

      # compute metrics
      width[i] = cs.length
      depth[i] = np.clip(.5 * (z0 + z1) - np.min(cs_z), 0, None)
      area[i] = np.sum(np.clip(.5 * (z0 + z1) - cs_z, 0, None)) * dx


  return width, depth, area




