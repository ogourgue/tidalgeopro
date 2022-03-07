""" Stream orders.

Compute stream orders along the skeleton of a tidal channel network.

Author: Olivier Gourgue (University of Antwerp)

"""


import numpy as np

################################################################################
# Hack. ########################################################################
################################################################################

def hack(node_sections, node_mul, dns):
    """Compute Strahler stream orders along skeleton.

    Args:
        node_sections (NumPy array): Skeleton node indices at skeleton sections.
        node_mul (NumPy array): Maximum upstream length at skeleton nodes.
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

                # Max. upstream length of the connected section upstream nodes.
                mul = node_mul[node_sections[con, 1]]

                # Loop over the connected sections.
                for i in range(len(con)):

                    # If it the connected section with the highest max. upstream
                    # length, its order is the same as downstream.
                    if i == np.argmax(mul):
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

def strahler(node_sections, dns):
    """Compute Strahler stream orders along skeleton.

    Only tested for non-braided channel networks.

    Args:
        node_sections (NumPy array): Skeleton node indices at skeleton sections.
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

        # Leave the loop if nothing has changed.
        if nc == 0:
            break

    return so

