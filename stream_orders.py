""" Stream orders.

Compute stream orders along the skeleton of a tidal channel network.

Author: Olivier Gourgue (University of Antwerp)

"""


import numpy as np

################################################################################
# Hack. ########################################################################
################################################################################

def hack(node_sections, node_mul, node_dl, dns):
    """Compute Strahler stream orders along skeleton.

    Args:
        node_sections (NumPy array): Skeleton node indices at skeleton sections.
        node_mul (NumPy array): Maximum upstream length at skeleton nodes.
        node_dl (NumPy array): Downstream length at skeleton nodes.
        dns (list of int): Downstream node indices.

    Returns:
        NumPy array: Hack stream order at skeleton sections.
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

    # Initialize list of new sections (sections with updated stream order).
    new = []

    # Loop over downstream nodes.
    for n in dns:

        # If a section is connected to a downstream node, its order is 1.
        s = np.where(node_sections == n)[0][0]

        # Stream order.
        so[s] = 1

        # Update list of new sections.
        new.append(s)

    #################
    # Higher order. #
    #################

    # As long as the stream order array contains zeros.
    while np.sum(so == 0) > 0:

        # Initialize list of old sections (new sections in the previous loop).
        old = new.copy()

        # Initialize list of new sections.
        new = []

        # Loop over old sections.
        for s in old:

            # List connected upstream sections.
            con = list(np.where(node_sections == node_sections[s, 1])[0])
            con.remove(s)

            # If only one connected section, its order is the same as
            # downstream.
            if len(con) == 1:
                so[con[0]] = so[s]

            else:

                # Upstream, downstream node indices of the connected sections.
                up = node_sections[con, 1]
                down = node_sections[con, 0]

                # Maximum upstream length at the upstream node of the connected
                # sections.
                mul_up = node_mul[up]

                # Length of the connected sections.
                length = node_dl[up] - node_mul[down]

                # Maximum upstream length at the downstream node of the
                # connected sections.
                mul_down = mul_up + length

                # Loop over the connected sections.
                for i in range(len(con)):

                    # If it is the connected section with the highest maximum
                    # upstream length, its order is the same as downstream.
                    if i == np.argmax(mul_down):
                        so[con[i]] = so[s]

                    # Otherwise, the order increases by 1.
                    else:
                        so[con[i]] = so[s] + 1

            # Update list of new sections.
            for ss in con:
                new.append(ss)

        # End loop if there is no new sections.
        if len(new) == 0:
            break

    return so

################################################################################
# Strahler. ####################################################################
################################################################################

def strahler(node_sections, node_dl, dns):
    """Compute Strahler stream orders along skeleton.

    Args:
        node_sections (NumPy array): Skeleton node indices at skeleton sections.
        node_dl (NumPy array): Downstream length at skeleton nodes.
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
        if (np.sum(node_sections == i) == 1) and i not in dns:

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

            # If a node has only one connected section with no stream order and
            # is not a downstream node.
            if np.sum(so[con] == 0) == 1 and i not in dns:

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

        # If nothing has changed, deal with loops (braided channels).
        if nc == 0:

            # Initialize list of loop nodes.
            loopn = []

            # Loop over nodes.
            for i in range(nn):

                # Connected sections.
                con = np.where(node_sections == i)[0]

                # If a node has one connected section with a stream order and
                # several connected sections with no stream order, update list
                # of loop nodes.
                if np.sum(so[con] > 0) > 0 and np.sum(so[con] == 0) > 1:
                    loopn.append(i)

            # Downstream length at loop nodes.
            loopn_dl = node_dl[loopn]

            # Loop node with maximum downstream length.
            ind = np.argmax(loopn_dl)

            # Connected sections.
            con = np.where(node_sections == loopn[ind])[0]

            # Maximum stream order along connected sections.
            so_max = np.max(so[con])

            # Propagate stream order.
            so[con[so[con] == 0][0]] = so_max

    return so

