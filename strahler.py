""" Strahler.

Compute the Strahler stream order along the skeleton of a tidal channel network.

Author: Olivier Gourgue (University of Antwerp)

"""


import numpy as np

################################################################################
# Stream orders. ###############################################################
################################################################################

def stream_orders(node_sections, dns):
    """Compute Strahler stream orders along skeleton.

    Only tested for non-braided rivers.

    Args:
        node_sections (Numpy array): Skeleton node indices at skeleton sections.
        dns (list of int): Downstream node indices.

    Returns:
        NumPy array: Strahler stream order at skeleton sections.
    """

    # Number of sections.
    ns = node_sections.shape[0]

    # Number of nodes.
    nn = np.max(node_sections) + 1

    # Initialize stream order array.
    so = np.zeros(ns, dtype = int)

    ################
    # First order. #
    ################

    # Loop over nodes.
    for i in range(nn):

        # If a node is only connected to one section and is not a downstream
        # node, its order is 1.
        if (np.sum(node_sections == i) == 1) and i not in (dns):

            # Section index.
            s = np.where(node_sections == i)[0][0]

            # Stream order.
            so[s] = 1

    #################
    # Higher order. #
    #################

    # As long as the stream order array contains zeros.
    while np.sum(so == 0) > 0:

        # Initialize number of changes.
        nc = 0

        # Loop over nodes.
        for i in range(nn):

            # Connected sections.
            con = np.where(node_sections == i)[0]

            # If a node has only one connected section with no stream order.
            if np.sum(so[con] == 0) == 1:

                # Maximum stream order along connected sections.
                so_max = np.max(so[con])

                # If only one connected section with maximum stream order, then
                # the same stream order is propagated.
                if np.sum(so[con] == so_max) == 1:
                    so[con[so[con] == 0][0]] = so_max
                    nc += 1

                # If several connected sections with maximum stream order, then
                # a higher stream order is propagated.
                elif np.sum(so[con] == so_max) > 1:
                    so[con[so[con] == 0][0]] = so_max + 1
                    nc += 1

        # Leave the loop if nothing has changed.
        if nc == 0:
            break

    return so

