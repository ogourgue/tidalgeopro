""" Skeleton.

Compute the skeleton of a tidal channel network.

Author: Olivier Gourgue (University of Antwerp & Boston University).

"""

import numpy as np
import shapely as shp
import time

import centerline.geometry


################################################################################
# Raw skeleton. ################################################################
################################################################################

def raw_skeleton(mpol):
    """Compute the center line of a tidal channel network.

    Args:
        mpol (MultiPolygon): Tidal channel network.

    Returns:
        List of MultiLineStrings: Raw skeleton. There is one raw skeleton
            MultilineString per tidal channel network Polygon.
    """

    print('')
    print('Computing the raw skeleton can be a long process.')
    print('')

    # Start timer.
    start = time.time()

    # Compute raw skeleton, as center line of the tidal channel network. All
    # segments (LineStrings of two Points) generated by Centerline are merged
    # into longer LineStrings using linemerge from Shapely operations.
    skls = []
    for pol in mpol.geoms:
        skls.append(shp.ops.linemerge(centerline.geometry.Centerline(pol)))

    # Convert raw skeletons into MultiLineStrings if necessary.
    for i in range(len(skls)):
        if skls[i].type == 'LineString':
            skls[i] = shp.geometry.MultiLineString([skls[i]])

    # Print time.
    print('Raw skeleton computed in %.2f seconds.' % (time.time() - start))
    print('')

    return skls


################################################################################
# Clean skeleton. ##############################################################
################################################################################

def clean_skeleton(skls, mpol, ratio = 1):
    """Clean the raw skeleton by removing small channel head sections.

    Args:
        skls (list of MultiLineStrings): Raw skeleton.
        mpol (MultiPolygon): Tidal channel network.
        ratio (float): Threshold ratio (section length / distance between
            downstream node and channel banks). Channel head sections with lower
            ratio are removed. Default to 1.

    Returns:
        List of MultiLineStrings: Clean skeleton. There is one clean skeleton
            MultilineString per tidal channel network Polygon.
    """

    # Remove small channel head sections.
    # Loop over tidal channel networks.
    for i in range(len(mpol.geoms)):
        # Tidal channel network.
        pol = mpol.geoms[i]
        # Raw skeleton.
        skl = skls[i]
        # Loop as long as there are channel head sections to remove.
        while True:
            # Convert MultiLineString into list of LineStrings.
            lss = []
            for ls in skl.geoms:
                lss.append(ls)
            # Indices of channel sections to remove.
            ind = []
            # Loop over channel sections.
            for j in range(len(skl.geoms)):
                # Channel section.
                ls = skl.geoms[j]
                # Remaining skeleton.
                skl0 = shp.geometry.MultiLineString(lss[:j] + lss[j + 1:])
                # Channel section nodes.
                p0 = shp.geometry.Point(ls.coords[0])
                p1 = shp.geometry.Point(ls.coords[-1])
                # Distance between channel section nodes and remaining skeleton.
                d0 = p0.distance(skl0)
                d1 = p1.distance(skl0)
                # Test if this is a channel head section.
                if d0 > 0 or d1 > 0:
                    # Downstream channel section node.
                    if d0 == 0:
                        pd = p0
                    else:
                        pd = p1
                    # Distance between downstream channel section node and
                    # channel banks.
                    d = pd.distance(pol.exterior)
                    for interior in pol.interiors:
                        if pd.distance(interior) < d:
                            d = pd.distance(interior)
                    # Test if the channel head section must be removed.
                    if ls.length / d < ratio:
                        ind.append(j)
            # Test if there are channel head sections to remove.
            if len(ind) > 0:
                # Remove them and merge remaining sections.
                lss = []
                for j in range(len(skl.geoms)):
                    if j not in ind:
                        lss.append(skl.geoms[j])
                skl = shp.ops.linemerge(shp.geometry.MultiLineString(lss))
                # Convert skeleton into MultiLineString if necessary.
                if skl.type == 'LineString':
                    skl = shp.geometry.MultiLineString([skl])
            else:
                # Update raw skeleton.
                skls[i] = skl
                # End loop.
                break

    return skls


################################################################################
# Final skeleton. ##############################################################
################################################################################

def final_skeleton(skls, mpol, ls, dx, buf = 1e-6):
    """Process the clean skeleton and compute downstream length.

    Skeleton nodes are network branching locations. Skeleton sections are
        channel reaches between two node. Skeleton points are equal-distance
        locations along the skeleton. The downstream length is the
        along-skeleton distance between a certain location and the downstream
        boundary.

    Args:
        skls (list of MultiLineStrings): Raw skeleton.
        mpol (MultiPolygon): Tidal channel network.
        ls (LineString): Downstream boundary.
        dx (float): Distance between two points of the final skeleton.
        buf (float): Buffer tolerance when evaluating zero distance (default to
            1e-6).

    Returns:
        NumPy array: Skeleton node coordinates.
        Numpy array: Skeleton section indices at skeleton nodes.
        Numpy array: Downstream length at skeleton nodes.
        NumPy array: Skeleton point coordinates.
        NumPy array: Skeleton section indices at skeleton points.
        Numpy array: Downstream length at skeleton points.
    """

    # Intersection between downstream LineString and channel network
    # MultiPolygon. Buffering is needed due to rounding errors if downstream
    # LineString is on the channel network MultiPolygon boundary.
    dmls = ls.intersection(mpol.buffer(buf))
    if dmls.type == 'LineString':
        dmls = shp.geometry.MultiLineString([dmls])

    # Merge skeletons into one list of LineStrings.
    lss = []
    for skl in skls:
        for ls in skl.geoms:
            lss.append(ls)

    # Connect skeleton with downstream LineStrings and determine downstream node
    # indices.
    for dls in dmls.geoms:
        # Distance between skeleton sections and downstream LineStrings.
        d = np.zeros(len(lss))
        for i, ls in enumerate(lss):
            d[i] = ls.distance(dls)
        # Indexes sorted by increasing distance.
        ind = np.argsort(d)
        # Loop over sorted skeleton sections.
        for i in ind:
            # Test if one of the section nodes is as close to the downstream
            # LineStrings than the skeleton section itself. If yes, the
            # candidate downstream section is directly connected to that node.
            # If no, the candidate downstream section is connected to a middle
            # point. Test if the candidate downstream section is entirely within
            # the channel network MultiPolygon (buffering is needed).

            # Skeleton section nodes and their distance to the downstream
            # LineString.
            p0 = shp.geometry.Point(lss[i].coords[0])
            p1 = shp.geometry.Point(lss[i].coords[-1])
            d0 = p0.distance(dls)
            d1 = p1.distance(dls)
            # Connection to first node.
            if d0 == d[i]:
                p2 = dls.interpolate(dls.project(p0))
                ls = shp.geometry.LineString([p0, p2])
                if ls.within(mpol.buffer(buf)):
                    lss.append(ls)
                    break
            # Connection to second node.
            elif d1 == d[i]:
                p2 = dls.interpolate(dls.project(p1))
                ls = shp.geometry.LineString([p1, p2])
                if ls.within(mpol.buffer(buf)):
                    lss.append(ls)
                    break
            # Connection to a middle point.
            else:
                p2, p3 = shp.ops.nearest_points(dls, lss[i])
                #p2 = dls.interpolate(.5, normalized = True)
                #p3 = lss[i].interpolate(lss[i].project(p2))
                ls = shp.geometry.LineString([p3, p2])
                if ls.within(mpol.buffer(buf)):
                    lss.append(ls)
                    # Split skeleton section in two.
                    # Loop over channel section points.
                    for j in range(len(lss[i].coords)):
                        # Channel section point.
                        pj = shp.geometry.Point(lss[i].coords[j])
                        # Case of split point exactly on a section point.
                        if lss[i].project(pj) == lss[i].project(p3):
                            lss.append(
                                shp.geometry.LineString(lss[i].coords[:j + 1]))
                            lss.append(
                                shp.geometry.LineString(lss[i].coords[j:]))
                            break
                        # Case of split point between two section points.
                        if lss[i].project(pj) > lss[i].project(p3):
                            lss.append(
                                shp.geometry.LineString(lss[i].coords[:j] +
                                                        [(p3.x, p3.y)]))
                            lss.append(
                                shp.geometry.LineString([(p3.x, p3.y)] +
                                                        lss[i].coords[j:]))
                            break
                    del lss[i]
                    break

    # Skeleton node coordinates.
    node_xy = []
    for ls in lss:
        for coords in [ls.coords[0], ls.coords[-1]]:
            if coords not in node_xy:
                node_xy.append(coords)
    node_xy = np.array(node_xy)

    # Skeleton section connectivity table.
    node_sections = np.zeros((len(lss), 2), dtype = int)
    for i, ls in enumerate(lss):
        n0 = int(np.argwhere(np.all(node_xy == ls.coords[0], axis = 1)))
        n1 = int(np.argwhere(np.all(node_xy == ls.coords[-1], axis = 1)))
        node_sections[i, 0] = n0
        node_sections[i, 1] = n1

    # Downstream node indices.
    dns = []
    for i in range(node_xy.shape[0]):
        # Test if distance between channel section node and downstream
        # LineString is zero.
        x, y = node_xy[i, :]
        if shp.geometry.Point((x, y)).distance(dmls) < buf:
            dns.append(i)

    # Compute downstream length at skeleton nodes and reorganize skeleton.
    (node_xy,
     node_sections,
     node_dl,
     lss) = donwstream_length(node_xy, node_sections, lss, dns)

    # Compute final skeleton by (i) removing channel sections and skeleton nodes
    # that have not been connected to the main skeleton, and (ii) redefining
    # skeleton points at regular distance from each other.

    # Initialize lists.
    node_xy_new = []
    node_sections_new = []
    node_dl_new = []
    skl_xy = []
    skl_sections = []
    skl_dl = []
    lss_new = []

    # Loop over old channel sections.
    for s in range(len(lss)):
        # Old node indices and corresponding downstream lengths.
        n0 = node_sections[s, 0]
        n1 = node_sections[s, 1]
        node_dl0 = node_dl[n0]
        node_dl1 = node_dl[n1]
        # Only keep channel sections connected to the main skeleton.
        if np.isfinite(node_dl0) and np.isfinite(node_dl1):

            # Update node lists by removing channel sections and skeleton nodes
            # that have not been connected to the main skeleton.

            # Update node lists for first node.
            node_xy0 = list(node_xy[n0, :])
            if node_xy0 in node_xy_new:
                node_sections_new.append([node_xy_new.index(node_xy0)])
            else:
                node_xy_new.append(node_xy0)
                node_sections_new.append([len(node_xy_new) - 1])
                node_dl_new.append(node_dl0)
            # Update node lists for last node.
            node_xy1 = list(node_xy[n1, :])
            if node_xy1 in node_xy_new:
                node_sections_new[-1].append(node_xy_new.index(node_xy1))
            else:
                node_xy_new.append(node_xy1)
                node_sections_new[-1].append(len(node_xy_new) - 1)
                node_dl_new.append(node_dl1)

            # Redefine skeleton points at regular distance from each other.

            # Downstream length at extreme regular points of the channel
            # section.
            skl_dl_loc_0 = np.ceil(node_dl0 / dx) * dx
            skl_dl_loc_1 = np.floor(node_dl1 / dx) * dx
            # Update regular skeleton point coordinates, sections and downstream
            # lengths.
            for skl_dl_loc in np.arange(skl_dl_loc_0, skl_dl_loc_1 + dx, dx):
                point = lss[s].interpolate(skl_dl_loc - node_dl0)
                skl_xy.append([point.x, point.y])
                skl_dl.append(skl_dl_loc)
                skl_sections.append(len(node_sections_new) - 1)

    # Convert lists into arrays.
    node_xy = np.array(node_xy_new)
    node_sections = np.array(node_sections_new)
    node_dl = np.array(node_dl_new)
    skl_xy = np.array(skl_xy)
    skl_sections = np.array(skl_sections)
    skl_dl = np.array(skl_dl)

    return node_xy, node_sections, node_dl, skl_xy, skl_sections, skl_dl

################################################################################
# Downstream length. ###########################################################
################################################################################

def donwstream_length(node_xy, node_sections, lss, dns):
    """Compute the downstream length and orient channel sections downstream.

    Args:
        node_xy (NumPy array): Skeleton node coordinates.
        node_sections (Numpy array): Skeleton section indices at skeleton nodes.
        lss (List of LineStrings): Skeleton sections.
        dns (List of int): List of downstream skeleton section indices.

    Returns:
        NumPy array: Skeleton node coordinates.
        Numpy array: Skeleton section indices at skeleton nodes.
        Numpy array: Downstream length at skeleton nodes.
        List of LineStrings: Skeleton sections.
    """

    # Initialize buffer list with downstream section indices.
    buf = []
    for dn in dns:
        buf.append(np.argwhere(node_sections == dn).reshape(-1)[0])

    # Channel section lengths.
    lens = np.zeros(node_sections.shape[0])
    for i in range(lens.shape[0]):
        lens[i] = lss[i].length

    # Initialize downstream length at skeleton nodes.
    node_dl = np.zeros(node_xy.shape[0]) + np.inf
    # Loop over channel sections in the buffer list.
    for i in range(len(buf)):
        # Section index.
        s = buf[i]
        # Downstream node index.
        nd = dns[i]
        # Upstream node index.
        if nd == node_sections[s, 0]:
            nu = node_sections[s, 1]
        else:
            nu = node_sections[s, 0]
        # Update downstream length at channel section nodes.
        node_dl[nd] = 0
        node_dl[nu] = lens[s]
        # Update channel section and LineString orientation.
        if nd != node_sections[s, 0]:
            node_sections[s, 0] = nd
            node_sections[s, 1] = nu
            ls_coords = list(lss[s].coords)
            ls_coords.reverse()
            lss[s] = shp.geometry.LineString(ls_coords)

    # Loop as long as the buffer list is not empty.
    while len(buf) > 0:
        # Initialize list of connected channel sections.
        con = []
        # Loop over channel sections in the buffer list.
        for s in buf:
            # Channel section nodes.
            n0 = node_sections[s, 0]
            n1 = node_sections[s, 1]
            # Look for connected channel sections.
            for n in [n0, n1]:
                ind = list(np.argwhere(node_sections == n)[:, 0])
                for nc in ind:
                    if nc != s:
                        con.append(nc)
        # Reset buffer list.
        buf = []
        # Loop over connected channel sections.
        for s in con:
            # Channel section nodes.
            n0 = node_sections[s, 0]
            n1 = node_sections[s, 1]
            # If downstream length at channel section nodes is lower than
            # previous value, update value and add channel section to buffer
            # list.
            node_dl0 = node_dl[n1] + lens[s]
            node_dl1 = node_dl[n0] + lens[s]
            if node_dl0 < node_dl[n0]:
                node_dl[n0] = node_dl0
                buf.append(s)
            if node_dl1 < node_dl[n1]:
                node_dl[n1] = node_dl1
                buf.append(s)
            # If downstream length if higher at downstream nodes vs. upstream
            # nodes, update channel section and LineString orientation.
            for s in range(node_sections.shape[0]):
                n0 = node_sections[s, 0]
                n1 = node_sections[s, 1]
                node_dl0 = node_dl[n0]
                node_dl1 = node_dl[n1]
                if node_dl1 < node_dl0:
                    node_sections[s, 0] = n1
                    node_sections[s, 1] = n0
                    ls_coords = list(lss[s].coords)
                    ls_coords.reverse()
                    lss[s] = shp.geometry.LineString(ls_coords)

    # Split channel sections at maximum downstream length.
    # Initialize list of channel sections to delete.
    trash = []
    # Loop over channel sections.
    for s in range(node_sections.shape[0]):
        # Channel section nodes.
        n0 = node_sections[s][0]
        n1 = node_sections[s][1]
        # Test if channel section length is higher than the difference between
        # upstream length at channel section nodes.
        if lens[s] - np.abs(node_dl[n1] - node_dl[n0]) > 1e-6:
            # Maximum downstream length on that channel section and
            # corresponding skeleton point.
            dl_max = .5 * (lens[s] + node_dl[n0] + node_dl[n1])
            p_max = lss[s].interpolate(dl_max - node_dl[n0])
            n_max = node_xy.shape[0]
            # Split LineString at point of maximum downstream length.
            # Loop over channel section points.
            for i in range(len(lss[s].coords)):
                # Channel section point.
                pi = shp.geometry.Point(lss[s].coords[i])
                # Case of split point exactly on a channel section point.
                if lss[s].project(pi) == dl_max - node_dl[n0]:
                    node_xy = np.append(node_xy, [[p_max.x, p_max.y]], axis = 0)
                    node_dl = np.append(node_dl, dl_max)
                    node_sections = np.append(node_sections, [[n0, n_max]],
                                              axis = 0)
                    node_sections = np.append(node_sections, [[n1, n_max]],
                                              axis = 0)
                    lss.append(shp.geometry.LineString(lss[s].coords[:i + 1]))
                    lss.append(shp.geometry.LineString(lss[s].coords[i:]))
                    trash.append(s)
                    break
                # Case of split point between two channel section points.
                if lss[s].project(pi) > dl_max - node_dl[n0]:
                    node_xy = np.append(node_xy, [[p_max.x, p_max.y]], axis = 0)
                    node_dl = np.append(node_dl, dl_max)
                    node_sections = np.append(node_sections, [[n0, n_max]],
                                              axis = 0)
                    node_sections = np.append(node_sections, [[n1, n_max]],
                                              axis = 0)
                    lss.append(shp.geometry.LineString(lss[s].coords[:i] +
                                                       [(p_max.x, p_max.y)]))
                    lss.append(shp.geometry.LineString([(p_max.x, p_max.y)] +
                                                       lss[s].coords[i:]))
                    trash.append(s)
                    break
    # Delete original split channel sections.
    trash.reverse()
    for s in trash:
        node_sections = np.delete(node_sections, s, axis = 0)
        del lss[s]

    return node_xy, node_sections, node_dl, lss
