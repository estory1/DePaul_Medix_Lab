#!/opt/local/bin/python2.7
#
# Merges DICOM CSV files into a single CSV file.
#
# Author: Evan Story (estory0@gmail.com)
# Date: 20151214
#
# Usage:
#   At a command prompt with Python and Pandas installed, run e.g.:
#
#   ./merge_DICOM_files.py -f ../../LIDC_Complete_20141106/Extracts/DICOM_metadata_extracts/*.csv
#   

import sys
import os
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-f", nargs="+", required=True, help="Input file(s).")
args = parser.parse_args()

files = args.f


print "* Processing files..."
all_files = []

# For each file...
for fn in files:
  # ...open the file...
  with open(fn, 'rb') as f:
    # ...read the file.
    df = pd.read_csv(f)

    all_files.append(df)


print "* Merging..."
df_out = pd.concat(all_files)

print "* Writing output file..."
df_out.to_csv(os.path.sep.join(["..", "..", "LIDC_Complete_20141106", "Extracts", "imageSOP_UID-filePath-dicominfo-ALL_PATIENTS.csv"]))