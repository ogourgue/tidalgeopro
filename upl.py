""" UPL: Unchanneled Path Length

This module allows to calculate the unchanneled path length in a watershed with a channel network

Author: Olivier Gourgue
       (University of Antwerp, Belgium & Boston University, MA, United States)

"""


import numpy as np
from scipy import spatial



################################################################################
# compute upl ##################################################################
################################################################################

def compute_upl(x, y, channel):

  """ Compute the unchanneled path length from a boolean channel field

  Required parameters:
  x, y (NumPy arrays of shape (n)) grid node coordinates
  channel (NumPy array of shape (n, m) and type logical): True if channel, False otherwise (m is number of time steps)

  Returns:
  NumPy array of shape (n, m): unchanneled path length

  """

  # initialize
  upl = np.zeros(channel.shape)

  # case of one time step
  if channel.ndim == 1:
    channel = channel.reshape((channel.shape[0], 1))
    upl = upl.reshape((upl.shape[0], 1))

  # for each time step
  for i in range(channel.shape[1]):

    # channel nodes
    channel_ind = np.flatnonzero(channel[:, i])
    channel_xy = np.array([x[channel_ind], y[channel_ind]]).T

    # non-channel nodes
    non_channel_ind = np.flatnonzero(channel[:, i] == 0)
    non_channel_xy = np.array([x[non_channel_ind], y[non_channel_ind]]).T

    # unchanneled path length
    if len(channel_ind) > 0:
        tree = spatial.KDTree(channel_xy)
        non_channel_upl, ind = tree.query(non_channel_xy)
        upl[non_channel_ind, i] = non_channel_upl

  if upl.shape[1] == 1:
    upl = upl.reshape((upl.shape[0]))

  # return
  return upl