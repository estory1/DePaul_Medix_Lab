import numpy as np
import math


def isPixelInROI(x, y, roi):
  return false



# def computeMergedROI(edge_map_numeric):


# Converted from C++: http://stackoverflow.com/a/23869476 (gorgeous viz: https://silentmatt.com/rectangle-intersection/)
def isRectOverlap(l1x, l1y, r1x, r1y, l2x, l2y, r2x, r2y):
  # print "isRectOverlap: " + str([l1x, l1y, r1x, r1y, l2x, l2y, r2x, r2y])
  return not (l1x > r2x or r1x < l2x or l1y > r2y or r1y < l2y)



# def computeBinary(xMaxVal, yMaxVal, edge_map_numeric):
# def computeBinary(xMaxVal, yMaxVal, edge_map_numeric_sortXAsc, edge_map_numeric_sortYAsc):
def computeBinary(pxs, pys, pxe, pye, wUnit, hUnit, edge_map_numeric_sortXAsc, edge_map_numeric_sortYAsc):
  ret = np.zeros((pxe - pxs, pye - pys))

  # edge_map_numeric_sortXAsc = np.sort(edge_map_numeric.view('i8,i8'), order=['f0'], axis=0).view(np.int)
  minX = (edge_map_numeric_sortXAsc[0])[0]
  maxX = (edge_map_numeric_sortXAsc[-1])[0]
  # edge_map_numeric_sortYAsc = np.sort(edge_map_numeric.view('i8,i8'), order=['f1'], axis=0).view(np.int)
  minY = (edge_map_numeric_sortYAsc[0])[1]
  maxY = (edge_map_numeric_sortYAsc[-1])[1]

  # if minX < 0:
  #   raise ValueError("min x in edge pixels < 0. Huh??")
  # if maxX > xMaxVal:
  #   raise ValueError("max x in edge pixels > edge of image. Huh??")
  # if minY < 0:
  #   raise ValueError("min y in edge pixels < 0. Huh??")
  # if maxY > yMaxVal:
  #   raise ValueError("max y in edge pixels > edge of image. Huh??")

  # check if any part of nodule is in ROI 
  if isRectOverlap(pxs, pys, pxe, pye, minX, minY, maxX, maxY):

    # scan across the pixel x values
    for roiX in edge_map_numeric_sortXAsc[:, 0]:       # range(minX, maxX + 1):   # +1 since range returns [), not [] (math notation, not Python array).
      # scan across the pixel y values
      px = False
      roiYvals = edge_map_numeric_sortYAsc[np.argwhere(edge_map_numeric_sortYAsc[:,0] == roiX), 1]
      print roiYvals
      roiYIdx = 0
      roiYLen = len(roiYvals)

      yRange = range(roiYvals[roiYIdx], roiYvals[-1] + 1)
      print yRange
      for yInRoiXRange in range(roiYvals[roiYIdx], roiYvals[-1]):
        if yInRoiXRange == roiYvals[roiYIdx+1]:
          px = not px
          roiYIdx += 1
        ret[roiX, yInRoiXRange] = px

  return ret