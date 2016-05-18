import pandas as pd
import re
from collections import OrderedDict


# Looks-up and returns the category for the specified row's ConvolutionalKernel column.
def getConvolutionalKernelCategory(row):
  oem_ConvolutionalKernel_OEM_map = lookup_categories_ConvolutionKernel[ row["Manufacturer"] ]
  cat = ""
  for c in oem_ConvolutionalKernel_OEM_map.keys():
    if len(oem_ConvolutionalKernel_OEM_map[c]) > 0 and re.search(oem_ConvolutionalKernel_OEM_map[c], row["ConvolutionKernel"], flags=re.IGNORECASE):
      cat = c
      break
  return cat

# A representation of Daniela's 20151130 "CT Recon kernel Mapping.xls".
lookup_categories_ConvolutionKernel = {
  "GE MEDICAL SYSTEMS": {
    "Smooth": "Smooth",
    "Standard": "Standard",
    "Sharp": "Bone",
    "Overenhancing": "Lung"
  },
  "SIEMENS": {
    "Smooth": "B10",
    "Standard": "(B30|B31)",
    "Sharp": "B50",
    "Overenhancing": "B70"
  },
  "TOSHIBA": {
    "Smooth": "",
    "Standard": "(FC10|FC01)",
    "Sharp": "FC50",
    "Overenhancing": ""
  },
  "Philips": {
    "Smooth": "A",
    "Standard": "(B|C)",
    "Sharp": "D",
    "Overenhancing": ""
  }
}