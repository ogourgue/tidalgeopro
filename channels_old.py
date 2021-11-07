""" Channels

This module allows to extract tidal channels from elevation maps and polygons describing the channel edges (limited to triangular grids for now)

Author: Olivier Gourgue
       (University of Antwerp, Belgium & Boston University, MA, United States)

"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy import linalg
from shapely import geometry


################################################################################
# boolean channel ##############################################################
################################################################################

def boolean_channels(x, y, z, hc, radius = None, cloud_fn = None, \
                     combine_radius_logical = True, \
                     combine_time_logical = False, \
                     linear_detrend_logical = False):

  """ Extract tidal channels from elevation maps, based on median neighborhood analysis, inspired by the first step of the multi-step approach by Liu et al. (2015, dx.doi.org/10.1016/j.jhydrol.2015.05.058)

  Required parameters:
  x, y (NumPy arrays of shape (n)): grid node coordinates
  z (NumPy array of shape (n) or (n, m): bottom elevation at grid nodes (axis 0) and different time steps (axis 1)
  hc (float): threshold residual (median(z) - z) above which the node is considered as within a channel

  Optional parameters:
  radius (float or list of p floats): mini-cloud radius(-ii)
  cloud_fn (file name or list of file names): mini-cloud binary file name(s)
  combine_radius_logical (boolean, default = True): if True, combine the results for each mini-cloud radius
  combine_time_logical (boolean, default = False): if True, combine the results for each time step (combine_radius_logical must be True)
  linear_detrend_logical (boolean, default = False): combine median analysis results with a median analysis applied to a linearly detrend elevation within each mini-cloud (experimental, not tested in depth and very slow for large datasets)

  Returns:
  NumPy array of shape (n) or (n, p) and type logical: True if channel, False otherwise

  """


  ##################
  # initialization #
  ##################

  # radius as a list
  if radius is not None and type(radius) is not list:
    radius = [radius]

  # cloud_fn as a list
  if cloud_fn is not None and type(cloud_fn) is not list:
    cloud_fn = [cloud_fn]

  # number of radius and/or mini-cloud file names
  if radius is not None:
    nr = len(radius)
  elif cloud_fn is not None:
    nr = len(cloud_fn)

  # check optional input
  if radius is not None and cloud_fn is not None:
    if len(radius) is not len(cloud_fn):
      print('number of radius must the same as number cloud_fn')
      sys.exit()
  if radius is None and cloud_fn is None:
    print('radius or cloud_fn must be defined (both also accepted)')
    sys.exit()

  # check combine options
  if combine_time_logical and not combine_radius_logical:
    print('combined in radius is mandatory if combined in time')
    sys.exit()

  # number of nodes
  nnode = len(x)

  # number of time steps
  if z.ndim == 1:
    nt = 1
  else:
    nt = z.shape[1]

  # initialize boolean channel
  if combine_radius_logical:
    if combine_time_logical:
      channel = np.zeros(nnode, dtype = bool)
    else:
      channel = np.zeros(z.shape, dtype = bool)
  else:
    channel = []


  # for each radius
  for i in range(nr):


    #####################################
    # compute/import/export mini-clouds #
    #####################################

    # if list of radius and mini-cloud file names are given
    if radius is not None and cloud_fn is not None:

      # import mini-cloud
      if os.path.isfile(cloud_fn[i]):
        cloud = import_mini_cloud(cloud_fn[i])

      # or compute & export mini-cloud
      else:
        cloud = mini_cloud_radius(x, y, radius[i])
        export_mini_cloud(cloud, cloud_fn[i])

    # if list of mini-cloud file names is not given
    elif cloud_fn is None:

      # compute mini-cloud
      cloud = mini_cloud_radius(x, y, radius[i])

    # if list of radius is not given
    elif radius is None:

      # import mini-cloud
      cloud = import_mini_cloud(cloud_fn[i])


    ###########################
    # compute boolean channel #
    ###########################

    # compute median
    median = np.zeros(z.shape)
    for j in range(len(x)):
      median[j, :] = np.median(z[cloud[j]], axis = 0)

    # compute residual
    res = median - z

    # compute boolean channel
    tmp_1 = (res > hc)


    ###################################
    # compute detrent boolean channel #
    ###################################

    if linear_detrend_logical:

      # for each mini-cloud
      for j in range(nnode):

        # number of nodes in the mini-cloud
        ncloud = len(cloud[j])

        # coordinates of nodes in the mini-cloud
        xcloud = x[cloud[j]]
        ycloud = y[cloud[j]]

        # initialization
        zl = np.zeros(z.shape)

        # bottom elevation in the mini-cloud
        if nt == 1:
          zcloud = z[cloud[j]]
        else:
          zcloud = z[cloud[j], :]

        # initialize linear system
        a = np.zeros((ncloud, 3))
        if nt == 1:
          b = np.zeros(ncloud)
        else:
          b = np.zeros((ncloud, nt))

        # build linear system
        a[:, 0] = xcloud
        a[:, 1] = ycloud
        a[:, 2] = 1
        b[:] = zcloud

        # solve linear system
        coef, _, _, _ = linalg.lstsq(a, b, lapack_driver = 'gelsy')

        # linear detrent elevation
        if nt == 1:
          zl[j] = coef[0] * x[j] + coef[1] * y[j] + coef[2]
        else:
          zl[j, :] = coef[0, :] * x[j] + coef[1, :] * y[j] + coef[2, :]

      # compute median
      median = np.zeros(z.shape)
      for j in range(nnode):
        median[j, :] = np.median(z[cloud[j]] - zl[j], axis = 0)

      # compute residual
      res = median - z

      # compute boolean channel
      tmp_2 = (res > hc)

    else:
      tmp_2 = np.zeros(z.shape, dtype = bool)


    ############################
    # combine boolean channels #
    ############################

    # compute boolean channel
    if combine_radius_logical:
      if combine_time_logical:
        channel += np.sum(tmp_1 + tmp_2, axis = 1)
      else:
        channel += tmp_1 + tmp_2
    else:
      channel.append(tmp_1 + tmp_2)

  # I think this is not correct, so I comment it. Further check is needed.
  # no list if only one variable
  # if nr == 1:
  #   channel = channel[0]

  # return boolean channel
  return channel


################################################################################
# compute mini cloud ###########################################################
################################################################################

def mini_cloud(x, y, tri):

  """ Compute mini-cloud of each node of a trinagular grid, as the list of all neighboring nodes sharing at least one triangle with it

  Required paramaters:
  x, y (NumPy arrays of shape (n)): grid node coordinates
  tri (NumPy array of shape (m, 3)): triangle connectivity table

  Returns:
  list of n arrays of different shapes: each array is a mini-cloud of node ids

  """

  # number of triangles
  ntri = len(tri)

  # number of nodes
  nnode = len(x)

  # integer type
  if nnode <= 127:
    dtype = np.int8
  elif nnode <= 32767:
    dtype = np.int16
  elif nnode <= 2147483647:
    dtype = np.int32
  else:
    dtype = np.int64

  # initialize list of mini-clouds
  cloud = [None] * nnode
  for i in range(nnode):
    cloud[i] = []

  # for each triangle
  for i in range(ntri):

    # for each triangle vertex
    for j in tri[i, :]:

      # for each triangle vertex
      for k in tri[i, :]:

        # add node j in mini-cloud k if not already in
        if j not in cloud[k]:
          cloud[k].append(j)

  # for each node
  for i in range(nnode):
    cloud[i] = np.array(cloud[i], dtype = dtype)

  return cloud



################################################################################
# compute mini cloud radius ####################################################
################################################################################


def mini_cloud_radius(x, y, r, nmax = np.inf):

  """ Compute mini-cloud of each point of a point cloud, as the list of all neighboring points in a certain radius

  Required paramaters:
  x, y (NumPy arrays of shape (n)): grid node coordinates
  r (float): radius defining neighborhood

  Optional parameter:
  nmax (integer): maximum number of nodes per mini-cloud (random selection if necessary)

  Returns:
  list of n arrays of different shapes: each array is a mini-cloud of node ids

  """

  # number of nodes
  nnode = len(x)

  # integer type
  if nnode <= 127:
    dtype = np.int8
  elif nnode <= 32767:
    dtype = np.int16
  elif nnode <= 2147483647:
    dtype = np.int32
  else:
    dtype = np.int64

  # initialize list of mini-clouds
  cloud = [None] * nnode

  # for each node
  for i in range(nnode):

    # node coordinates
    x0 = x[i]
    y0 = y[i]

    # data within square bounding box of length (2r)
    tmp_j = (x >= x0 - r) * (x <= x0 + r) * (y >= y0 - r) * (y <= y0 + r)
    tmp_x = x[tmp_j]
    tmp_y = y[tmp_j]

    # data within circle of radius r
    d2 = np.zeros(nnode)
    d2[tmp_j] = (tmp_x - x0) * (tmp_x - x0) + (tmp_y - y0) * (tmp_y - y0)
    tmp_j *= (d2 <= r * r)
    j = np.array(np.argwhere(tmp_j).reshape(-1), dtype = dtype)

    # random selection if too many nodes in the mini-cloud
    if len(j) > nmax:
      j = j[np.random.randint(0, len(j), nmax)]

    # add node indices to mini-cloud
    cloud[i] = j

  return cloud



################################################################################
# export mini cloud ############################################################
################################################################################

def export_mini_cloud(cloud, filename):

  """ Export list of mini-clouds in a binary file

  Required parameters:
  cloud (list of n arrays of different shapes): each array is a mini-cloud of node ids
  filename (file name): binary file name

  """

  # number of mini-clouds
  ncloud = len(cloud)

  # integer type
  if ncloud <= 127:
    dtype = np.int8
  elif ncloud <= 32767:
    dtype = np.int16
  elif ncloud <= 2147483647:
    dtype = np.int32
  else:
    dtype = np.int64

  # open file
  file = open(filename, 'w')

  # number of mini-clouds
  np.array(ncloud, dtype = int).tofile(file)

  # for each mini-cloud
  for i in range(ncloud):

    # number of nodes
    np.array(len(cloud[i]), dtype = dtype).tofile(file)

    # node indices
    np.array(cloud[i], dtype = dtype).tofile(file)

  # close file
  file.close()



################################################################################
# import mini cloud ############################################################
################################################################################

def import_mini_cloud(filename):

  """ Import list of mini-clouds from a binary file

  Required parameter:
  filename (file name): binary file name

  Returns:
  list of n arrays of different shapes: each array is a mini-cloud of node ids

  """

  # open file
  file = open(filename, 'r')

  # number of mini-clouds
  ncloud = np.fromfile(file, dtype = int, count = 1)[0]

  # integer type
  if ncloud <= 127:
    dtype = np.int8
  elif ncloud <= 32767:
    dtype = np.int16
  elif ncloud <= 2147483647:
    dtype = np.int32
  else:
    dtype = np.int64

  # initialize list of mini-clouds
  cloud = [None] * ncloud
  for i in range(ncloud):
    cloud[i] = []

  # for each mini-cloud
  for i in range(ncloud):

    # number of nodes
    n = np.fromfile(file, dtype = dtype, count = 1)[0]

    # node indices
    cloud[i] = np.fromfile(file, dtype = dtype, count = n)

  # close file
  file.close()

  return cloud



################################################################################
# channel edges ################################################################
################################################################################

def channel_edges(x, y, tri, channel, smin = 1e-6):

  """ Compute the channel edges of a tidal channel network based on a boolean field determining which nodes of a triangular mesh are within a channel or not

  Required parameters:
  x, y (NumPy arrays of shape (n)) grid node coordinates
  tri (NumPy array of shape (m, 3): triangle connectivity table
  channel (NumPy array of shape (n) and type logical): True if within channel, False otherwise

  Optional parameters:
  smin (float, default 1e-6): minimum polygon surface area (m^2); polygons with smaller surface area will be disregarded

  Returns:
  MultiPolygon: structure of multiple polygons describing channel edges

  """

  #########################
  # compute channel edges #
  #########################

  TriContourSet = plt.tricontour(x, y, tri, channel.astype(int), levels = [.5])


  ###############################
  # convert to shapely polygons #
  ###############################

  pols = []
  for contour_path in TriContourSet.collections[0].get_paths():

    xy = contour_path.vertices
    coords = []
    for i in range(xy.shape[0]):
      coords.append((xy[i, 0], xy[i, 1]))
    pols.append(geometry.Polygon(coords))


  #################################
  # include interiors in polygons #
  #################################

  # sort polygons by surface areas
  s = [pol.area for pol in pols]
  inds = np.flip(np.argsort(s))
  pols = [pols[ind] for ind in inds]

  # insert interiors one by one (to avoid inserting interiors of interiors)
  stop_while = False
  while not stop_while:
    stop_for = False
    for i in range(1, len(pols)):
      for j in range(i):
        if pols[i].within(pols[j]):
          # update polygon j with interior i
          shell = pols[j].exterior.coords
          holes = []
          for k in range(len(pols[j].interiors)):
            holes.append(pols[j].interiors[k].coords)
          holes.append(pols[i].exterior.coords)
          pols[j] = geometry.Polygon(shell = shell, holes = holes)
          # delete polygon i
          del pols[i]
          # stop double for-loop
          stop_for = True
          break
      if stop_for:
        break
    # stop while-loop
    if i == len(pols) - 1 and j == i - 1:
      stop_while = True

  return geometry.MultiPolygon(pols)