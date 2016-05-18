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
#   ./append_ConvolutionKernel_category.py -f ../../LIDC_Complete_20141106/Extracts/DICOM_metadata_extracts/*.csv
#   

import sys
import os
import pandas as pd
import argparse
import re

import manager_ConvolutionKernel as mck


parser = argparse.ArgumentParser()
parser.add_argument("-f", nargs="+", required=True, help="Input file(s).")
args = parser.parse_args()

files = args.f


print "* Processing file..."
all_files = []

# ...open the file...
with open(files[0], 'rb') as f:     # os.path.sep.join("..", "..", "LIDC_Complete_20141106", "Extracts", "imageSOP_UID-filePath-dicominfo-ALL_PATIENTS.csv")
  # ...read the file.
  df = pd.read_csv(f, low_memory=False)


cols = [col for col in df.columns if not re.search(r'Unnamed: ', col)]
df2 = df[cols]


# cat = pd.Series(mck.getConvolutionalKernelCategory(df.) , index=df.index)
df2["ConvolutionalKernel_Category"] = df2.apply(mck.getConvolutionalKernelCategory, axis=1)
df2.to_csv(os.path.sep.join(["..", "..", "LIDC_Complete_20141106", "Extracts", "imageSOP_UID-filePath-dicominfo-ALL_PATIENTS-with_ConvolutionalKernel_Category.csv"]))