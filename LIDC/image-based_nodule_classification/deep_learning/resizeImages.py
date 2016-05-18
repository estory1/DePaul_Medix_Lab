#!/opt/local/bin/python2.7
#
# Resizes a subtree of images to the size of the largest image in the set, preserving relative paths.
#
# Author: Evan Story (estory1@gmail.com) 
# Date created: 2015111?
#
# Usage:
#   python resizeImages.py {input folder path root} {input file name glob} {output folder path root} 
#
import sys
import os
import errno
from PIL import Image       # install first: py27-Pillow

import re
import fnmatch
import os
import multiprocessing as mp
import math


# Find files by file name pattern: http://stackoverflow.com/a/2186673
# Since this is a generator function (uses yield), processing can occur on each yield's return, not upon complete execution of this function.
def find_files(dirRoot, patt):
  for root, dirs, files in os.walk(dirRoot):
    for baseName in files:
      if fnmatch.fnmatch(baseName, patt):
        filename = os.path.join(root, baseName)
        yield filename

# Ensure folder exists. Src: http://stackoverflow.com/questions/273192/in-python-check-if-a-directory-exists-and-create-it-if-necessary
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

# Resize all images to max image size and save with new file name.
# If this fn has only a single input, then we Need the "*" s.t. the parameter is passed as a ptr during multiprocessing. Src: http://stackoverflow.com/questions/27388718/passing-a-list-of-items-to-multiprocessing-function
def resizeImages(output_root, allValidFilePaths, maxPxDim):
  for fp in allValidFilePaths:
    im = Image.open(fp)
    imRsz = im.resize((maxPxDim,maxPxDim), Image.NEAREST)   # resize using nearest-neighbor
    # extract relative folder path
    folderArr = fp.split(os.path.sep)[-4:][0:3]
    folderPath = output_root + os.path.sep + os.path.sep.join(folderArr)
    print "* folderPath: " + folderPath
    make_sure_path_exists(folderPath)
    fn = fp.split(os.path.sep)[-1]
    # generate file save path
    fpRsz = os.path.sep.join([folderPath, fn + "." + str(imRsz.width) + "x" + str(imRsz.height) + ".tiff"]) #re.sub("^.*\/", "", fp) + "." + str(imRsz.width) + "x" + str(imRsz.height) + ".tiff")
    # print "Would save: " + fpRsz #+ " -- for: " + fp
    print "* Saving: " + fpRsz #+ " -- for: " + fp
    imRsz.save(fpRsz)



### First, create the output folder if nonexistent.
output_root = sys.argv[3]
# make_sure_path_exists(output_root)


### Get all file paths.
allFilePaths = list(find_files(sys.argv[1], sys.argv[2]))

### Find largest image.
maxPxSz = 0
maxPxDim = -1
maxPxFp = ""
allValidFilePaths = []
for fp in allFilePaths:
  fSz = os.path.getsize(fp)
  if fSz == 0:
    print "* 0 byte file found: " + fp
  else:
    allValidFilePaths.append(fp)
#     im = Image.open(fp)
#     # print fn + ": " + str(os.path.getsize(fn)) + " : " + str(im.format) + " : " + str(im.size) + " : " + str(im.mode)
#     px = im.width * im.height
#     if px > maxPxSz:
#       maxPxDim = max(im.size)
#       maxPxSz = px
#       maxPxFp = fp

# # The largest image found: what is its path, dimensionality, and byte size?
# print "* Largest: " + str(maxPxDim) + " : " + str(maxPxSz) + " : " + maxPxFp

# print str(len(allValidFilePaths)) + " / " + str(sum(1 for i in allFilePaths))   # can't call len() on a generator object, so I guess we'll just loop through again and sum each time. Probably only a few million CPU cycles...


### Do resizing.
# Multiproc src: http://sebastianraschka.com/Articles/2014_multiprocessing_intro.html
numCpusToUse = int(math.floor(mp.cpu_count() * 0.8))

maxPxDim = 32

pool = mp.Pool(processes=numCpusToUse)
results = [ pool.apply_async(resizeImages, args=(output_root, allValidFilePaths, maxPxDim) ) ]
output = [ p.get() for p in results ]
# print output

