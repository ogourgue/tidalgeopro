""" Watersheds

This module allows to compute watershed areas and upstream lengths along a tidal channel network skeleton

Author: Olivier Gourgue
       (University of Antwerp, Belgium & Boston University, MA, United States)

"""


import numpy as np
import osgeo.gdal as gdal
import osgeo.osr as osr

# pysheds-0.2.7
from pysheds.grid import Grid



################################################################################
# virtual digital elevation model ##############################################
################################################################################

def virtual_dem(x, y, skl_coords, skl_dist):

  """ Calculate a virtual digital elevation model, as the distance to the skeleton + the downstream length along the skeleton.

  Required parameters:
  x (Numpy array of shape (nx)): x-coordinates
  y (Numpy array of shape (ny)): y-coordinates
  skl_coords (Numpy array of shape (m, 2)): coordinates of the skeleton points
  skl_dist (Numpy array of shape (m)): downstream length at each skeleton point

  Returns:
  Numpy array of shape (nx, ny): virtual DEM
  Numpy array of shape (p, 2): unique skeleton points anchored on the structured grid
  Numpy array of shape (m): connectivity table to retrieve the original skeleton point; i-th entry is the index of the corresponding unique anchored point
  Numpy array of shape (p): downstream length at each unique skeleton points anchored on the structured grid

  """

  # skeleton point coordinates anchored on the grid
  grid_skl_coords = np.zeros(skl_coords.shape)
  for i in range(skl_coords.shape[0]):
    grid_skl_coords[i, 0] = x[np.argmin(np.abs(skl_coords[i, 0] - x))]
    grid_skl_coords[i, 1] = y[np.argmin(np.abs(skl_coords[i, 1] - y))]

  # remove duplicate points (grid_skl_coords_unique) and create connectivity
  # table to retrieve original skeleton points (grid_skl_inverse)
  (grid_skl_coords_unique,
   grid_skl_inverse) = np.unique(grid_skl_coords, axis = 0,
                                 return_inverse = True)
  grid_skl_dist = np.zeros(grid_skl_coords_unique.shape[0])
  for i in range(len(grid_skl_dist)):
    grid_skl_dist[i] = np.max(skl_dist[grid_skl_inverse == i])

  # virtual dem
  vdem = np.zeros((len(x), len(y)))
  for i in range(len(x)):
    for j in range(len(y)):
      dist = ((x[i] - grid_skl_coords_unique[:, 0]) ** 2 +
              (y[j] - grid_skl_coords_unique[:, 1]) ** 2) ** .5
      vdem[i, j] = np.min(dist) + grid_skl_dist[np.argmin(dist)]

  return vdem, grid_skl_coords_unique, grid_skl_inverse, grid_skl_dist



################################################################################
# watershed metrics ############################################################
################################################################################

def metrics(x, y, skl_coords, skl_dist, mask = None, z = None, platforms = None,
            cov = None, tiff = 'vdem.tiff', remove_tiff = True,
            resolve_flats = True):

  """ Calculate watershed areas, upstream mainstream lengths and mean watershed platform elevations along the skeleton of a tidal channel network.

  Required parameters:
  x (Numpy array of shape (nx)): x-coordinates
  y (Numpy array of shape (ny)): y-coordinates
  skl_coords (Numpy array of shape (m, 2)): coordinates of the skeleton points
  skl_dist (Numpy array of shape (m)): downstream length at each skeleton point

  Optional parameters:
  mask (Numpy array of shape(nx, ny), default = None): defines which grid cells are outside the domain of interest; if None, the domain of interest is the entire grid
  z (Numpy array of shape(nx, ny), default = None): bottom elevation; if None, mean watershed platform elevation is not computed
  platforms (Numpy array of shape(nx, ny), default = None): defines which grid cells are platforms; if None, mean watershed platform elevation is not computed
  tiff (string, default = 'vdem.tiff'): geo tiff file name where the virtual DEM is stored before being imported by pysheds functions
  remove_tiff (boolean, default = True): if True the geo tiff file is removed at the end of the calculation

  Returns:
  Numpy array of shape (m): watershed area along the skeleton
  Numpy array of shape (m): upstream mainstream length along the skeleton
  Numpy array of shape (m): mean watershed platform elevation along the skeleton (only if z and platforms are not None)

  """

  ###############
  # virtual dem #
  ###############

  # compute virtual dem
  (vdem,
   grid_skl_coords,
   grid_skl_inverse,
   grid_skl_dist) = virtual_dem(x, y, skl_coords, skl_dist)


  #############################################################
  # save virtual dem into geo tiff file (required by pysheds) #
  #############################################################

  # from Python GDAL/OGR Cookbook 1.0 documentation: https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html#create-raster-from-array

  # grid data
  x0 = x[0]
  dx = x[1] - x[0]
  y0 = y[0]
  ny = len(y)

  # coordinate system does not matter as long a x, y are projected coordinates
  # (here: Belge 1972 / Belgian Lambert 72)
  epsg = 31370

  # save geo tiff
  driver = gdal.GetDriverByName('GTiff')
  outRaster = driver.Create(tiff, vdem.shape[0], vdem.shape[1], 1,
                            gdal.GDT_Byte)
  outRaster.SetGeoTransform((x0, dx, 0, y0 + (ny - 1) * dx, 0, -dx))
  outband = outRaster.GetRasterBand(1)
  outband.WriteArray(np.flipud(vdem.T))
  outRasterSRS = osr.SpatialReference()
  outRasterSRS.ImportFromEPSG(epsg)
  outRaster.SetProjection(outRasterSRS.ExportToWkt())
  outband.FlushCache()


  ########################################
  # watersheds and corresponding metrics #
  ########################################

  # read geo tiff
  grid = Grid.from_raster(tiff, data_name = 'zero')

  # add virtual dem manually (import geo tiff does not seem to work)
  grid.add_gridded_data(data = np.flipud(vdem.T), data_name = 'vdem',
                        affine = grid.affine, crs = grid.crs)

  # initialize variables
  grid_skl_watershed_area = np.zeros(grid_skl_coords.shape[0])
  grid_skl_upstream_length = np.zeros(grid_skl_coords.shape[0])
  if z is not None and platforms is not None:
    grid_skl_platform_elevation = np.zeros(grid_skl_coords.shape[0]) + np.nan
  if cov is not None and platforms is not None:
    grid_skl_platform_cover = np.zeros(grid_skl_coords.shape[0]) + np.nan

  # downstream length on the structured grid (nan values outside skeleton cells)
  grid_dist = np.zeros(vdem.shape) + np.nan
  for i in range(grid_skl_coords.shape[0]):
    indx = np.argwhere(grid_skl_coords[i, 0] == x)[0]
    indy = np.argwhere(grid_skl_coords[i, 1] == y)[0]
    grid_dist[indx, indy] = grid_skl_dist[i]

  # flow direction
  if resolve_flats:
    grid.resolve_flats(data = 'vdem', out_name = 'inflated_vdem')
    grid.flowdir(data = 'inflated_vdem', out_name = 'dir', routing = 'dinf')
  else:
    grid.flowdir(data = 'vdem', out_name = 'dir', routing = 'dinf')

  for i in range(grid_skl_coords.shape[0]):

    # determine catchment
    grid.catchment(data = 'dir', x = grid_skl_coords[i, 0],
                   y = grid_skl_coords[i, 1], out_name = 'catch',
                   recursionlimit = 15000, xytype = 'label', routing = 'dinf')

    # convert to boolean array with index 'ij'
    catch = np.array(grid.catch)
    catch = (catch != 0)
    catch = np.flipud(catch).T

    # apply mask
    catch[mask] = False

    # force the pour point to be in the catchment
    indx = np.argwhere(grid_skl_coords[i, 0] == x)[0]
    indy = np.argwhere(grid_skl_coords[i, 1] == y)[0]
    catch[indx, indy] = True

    # watershed area
    grid_skl_watershed_area[i] = np.sum(catch) * (dx ** 2)

    # upstream mainstream length
    grid_skl_upstream_length[i] = (np.nanmax(grid_dist[catch]) -
                                   grid_skl_dist[i])

    # mean watershed platform elevation
    if z is not None and platforms is not None:
      if (np.any(platforms * catch) and
          np.any(np.isfinite(z[platforms * catch]))):
        grid_skl_platform_elevation[i] = np.nanmean(z[platforms * catch])

    # mean watershed vegetation cover
    if cov is not None and platforms is not None:
      if (np.any(platforms * catch) and
          np.any(np.isfinite(cov[platforms * catch]))):
        grid_skl_platform_cover[i] = np.nanmean(cov[platforms * catch])


  # project variables on final skeleton
  skl_watershed_area = grid_skl_watershed_area[grid_skl_inverse]
  skl_upstream_length = grid_skl_upstream_length[grid_skl_inverse]
  if z is not None and platforms is not None:
    skl_platform_elevation = grid_skl_platform_elevation[grid_skl_inverse]
  if cov is not None and platforms is not None:
    skl_platform_cover = grid_skl_platform_cover[grid_skl_inverse]

  # list of outputs (depends on inputs)
  output = [skl_watershed_area, skl_upstream_length]
  if z is not None and platforms is not None:
    output.append(skl_platform_elevation)
  if cov is not None and platforms is not None:
    output.append(skl_platform_cover)

  return output


