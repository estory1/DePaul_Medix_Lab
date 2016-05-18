#!/opt/local/bin/python2.7

# Extract counts of the specified fields.
#
# Author: Evan Story (estory0@gmail.com)
# Date: 20151013
#
# Usage:
#   At a command prompt with Python and Pandas installed,
#   and from within the folder containing the CSV files whose columns you wish to count,
#   run:

import sys
from os import listdir
from os.path import isfile, join
import pandas as pd
import json
from collections import OrderedDict
import argparse
import numpy as np
import re

import manager_ConvolutionKernel as mgr_CK

# Returns the command-line syntax of this script.
def getSyntax():
  return "\tSyntax: python field_count.py -{p,d} {CSV file path} [CSV file(s) path(s)...]"

# Increments by 1 at a location at dictionary[col][val].
def accumulate(d, col, val):
  if d[col].has_key(val):
    d[col][val] = d[col][val] + 1
  else:
    d[col][val] = 1

# # Looks-up and returns the category for the specified row's ConvolutionalKernel column.
# def getConvolutionalKernelCategory(row):
#   oem_ConvolutionalKernel_OEM_map = lookup_categories_ConvolutionKernel[ row["Manufacturer"] ]
#   cat = ""
#   for c in oem_ConvolutionalKernel_OEM_map.keys():
#     if len(oem_ConvolutionalKernel_OEM_map[c]) > 0 and re.search(oem_ConvolutionalKernel_OEM_map[c], row["ConvolutionKernel"], flags=re.IGNORECASE):
#       cat = c
#       break
#   return cat

# # A representation of Daniela's 20151130 "CT Recon kernel Mapping.xls".
# lookup_categories_ConvolutionKernel = {
#   "GE MEDICAL SYSTEMS": {
#     "Smooth": "Smooth",
#     "Standard": "Standard",
#     "Sharp": "Bone",
#     "Overenhancing": "Lung"
#   },
#   "SIEMENS": {
#     "Smooth": "B10",
#     "Standard": "(B30|B31)",
#     "Sharp": "B50",
#     "Overenhancing": "B70"
#   },
#   "TOSHIBA": {
#     "Smooth": "",
#     "Standard": "(FC10|FC01)",
#     "Sharp": "FC50",
#     "Overenhancing": ""
#   },
#   "Philips": {
#     "Smooth": "A",
#     "Standard": "(B|C)",
#     "Sharp": "D",
#     "Overenhancing": ""
#   }
# }


parser = argparse.ArgumentParser()

parser.add_argument("-c", nargs="+", required=True, help="Column(s) on which to query.")
parser.add_argument("-p", action="store", required=True, help="Type of count to perform: p = per patient; d = per DICOM file; d_cf = per DICOM file, categories of ConvolutionKernel (from Daniela's 20151130 \"CT Recon kernel Mapping.xls\"), p_cf = same as d_cf except is per patient.")
parser.add_argument("-t", nargs=2, action="store", help="Crosstab of the two specified columns.")
parser.add_argument("-f", nargs="+", required=True, help="Input file(s).")
args = parser.parse_args()


# Input check.
if len(sys.argv) < 3:
  sys.exit("Must have >= 3 parameters.\n\n" + getSyntax())

# Create single attr dict for frequencies.
colsAcrossCsvFiles = { key : {} for key in args.c }     # for an unknown reason, the following causes all key-value pairs to display for each key: dict.fromkeys(args.c, {})
colsAcrossCsvFiles = OrderedDict(sorted(colsAcrossCsvFiles.items(), key=lambda t: t[0]))

# Assign arg arrays.
countType = args.p
files = args.f
crosstabCols = args.t
doCrosstab = args.t != None and len(args.t) == 2
ctRows = []
ctCols = []
if doCrosstab:
  ctRowName = args.t[0]
  ctColName = args.t[1]
  print ctRowName
  print ctColName


# For each file...
for fn in files:
  # ...open the file...
  with open(fn, 'rb') as f:
    # ...read the file.
    df = pd.read_csv(f)

    # For each column in the set of columns on which we're querying...
    for col in colsAcrossCsvFiles.keys():

      # ...if the column is in the file's set of columns...
      if col in df.columns:

        # FOR PER-PATIENT (a.k.a. PER-SCAN) COUNTS: (count a value for only 1 of its n occurrences per DICOM file)
        if countType == "p":
          colVals = df[col]
          uniqueColVals = set(colVals)

          for ucv in uniqueColVals:
            accumulate(colsAcrossCsvFiles, col, ucv)
            # TODO: could this actually ever work?... generally, no - the # elements in the rows and cols can easily get out of sync.
            # if doCrosstab:
            #   if col == ctRowName:
            #     ctRows += uniqueColVals
            #   if col == ctColName:
            #     ctCols += uniqueColVals


        # FOR PER-DICOM-FILE COUNTS: (count a value for each of its n occurrences per DICOM file)
        if countType == "d":
          for idx, row in df.iterrows():
            rowVal = row[col]
            accumulate(colsAcrossCsvFiles, col, rowVal)
            if doCrosstab:
              if col == ctRowName:
                ctRows.append(str(rowVal))
              if col == ctColName:
                ctCols.append(str(rowVal))


        # FOR PER-DICOM FILE, categories of ConvolutionKernel
        if countType == "d_cf":
          for idx, row in df.iterrows():
            rowVal = row[col]
            
            cat = mgr_CK.getConvolutionalKernelCategory(row)

            val =  row["Manufacturer"] + ": " + row["ConvolutionKernel"] + ": " + cat # cat #
            accumulate(colsAcrossCsvFiles, col, val)
            if doCrosstab:
              if col == ctRowName:
                ctRows.append(str(rowVal))
              if col == ctColName:
                ctCols.append(str(rowVal))


        # FOR PER-PATIENT COUNTS, categories of ConvolutionKernel
        if countType == "p_cf":
          vals_oem_ck = df[["Manufacturer","ConvolutionKernel"]].drop_duplicates()

          for idx, row in vals_oem_ck.iterrows():
            cat = mgr_CK.getConvolutionalKernelCategory(row)
            val =  row["Manufacturer"] + ": " + row["ConvolutionKernel"] + ": " + cat # cat #
            accumulate(colsAcrossCsvFiles, col, val)
            




# Print frequency tables.
for col in colsAcrossCsvFiles:
  print "----- " + col + " -----"
  print json.dumps(colsAcrossCsvFiles[col], sort_keys=True, indent=4)

# Optional: print crosstab.
if doCrosstab:
  a = np.array(ctRows, dtype=float)
  b = np.array(ctCols, dtype=float)
  # print len(a)
  # print ctRows
  # print a
  # print len(b)
  # print ctCols
  # print b
  ct = pd.crosstab(a, b, rownames=['a'], colnames=['b'])
  print ct
  ct.to_csv("field_count_crosstab.csv")
