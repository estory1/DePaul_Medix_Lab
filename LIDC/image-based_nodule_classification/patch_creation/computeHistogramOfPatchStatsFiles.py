#!/opt/local/bin/python2.7

### Purpose: Generate a histogram of the mean values of all patches using the summary stats CSV files. ###
import copy
import sys
import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()
import gc     # for manually-forced GC
import collections

if len(sys.argv) < 2:
  print "Syntax: " + sys.argv[0] + " {path to 'LIDC-IDRI-edge_patches' folder, without a trailing "+os.sep+"}\n\n (without the '{' and '}' chars)"
  sys.exit(1)


num_rounding_decimals = 2
binSize = pow(10, num_rounding_decimals)



## Get all the file paths. ##
def find_patch_summary_files(curr_folder_path, summary_file_glob):
  """Navigates to e.g.:

  '../../../LIDC_Complete_20141106/LIDC-IDRI-edge_patches/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192/patches/'
  
  and returns the CSV files there, without enumerating all the image files in the bin/ and orig/ folders beneath that level."
  """
  return glob.glob (curr_folder_path + os.sep + '*'+os.sep+'*'+os.sep+'*'+os.sep+'patches'+os.sep + summary_file_glob)  # we're globbing on e.g.: ('../../../LIDC_Complete_20141106/LIDC-IDRI-edge_patches/*/*/*/patches/*.csv')


# Navigate into each patient's "patches" folder containing the CSV files, without enumerating and then discarding all the bin and orig image files.
patches_root_dir_path_match_str = ""
patches_root_dir_path = sys.argv[1]
summary_file_glob = "*-bin_patch_stats.csv"
all_summary_files = find_patch_summary_files(patches_root_dir_path, summary_file_glob)
# print all_summary_files


## Process the CSV files. ##
# all_patients_and_means = []
all_patients_and_means_unordered = {}
# For each file...
for sf in all_summary_files:
  gc.collect()
  # Read the file...
  print "* Reading file: " + sf
  with open(sf, 'rb') as f:
    csvreader = csv.reader(f, delimiter=',', quotechar='"')
    # Skip the header row.
    csvreader.next()
    # Get the mean out of each row.
    for row in csvreader:
      # Extract its mean value, and round so we can use it as a sufficiently-small key...
      mean = round(float(row[1]), num_rounding_decimals)
      # Extract the patient ID from the file path...
      dirs_in_path = sf.split(os.sep)
      pat_id = dirs_in_path[-5]
      # Append the value to an array.
      # all_patients_and_means_unordered.append([pat_id, mean])
      mean_str = str(mean)

      if not all_patients_and_means_unordered.has_key(pat_id):
        all_patients_and_means_unordered[pat_id] = {}
      if not all_patients_and_means_unordered[pat_id].has_key(mean_str):
        all_patients_and_means_unordered[pat_id][mean_str] = 0
      all_patients_and_means_unordered[pat_id][mean_str] += 1

      # if all_patients_and_means_unordered[pat_id].has_key(mean_str):
      #   print all_patients_and_means_unordered[pat_id][mean_str]

# Sort the results at each level.
# all_patients_and_means = collections.OrderedDict(all_patients_and_means_unordered)
all_patients_and_means = collections.OrderedDict(sorted(all_patients_and_means_unordered.items(), key=lambda t: t[0]))
for p in all_patients_and_means.keys():
  all_patients_and_means[p] = collections.OrderedDict(sorted(all_patients_and_means_unordered[p].items(), key=lambda t: t[0])) # collections.OrderedDict(all_patients_and_means_unordered[p])

# print all_patients_and_means
# sys.exit()



## Generate the histogram structure. ##
# all_means = [m[1] for m in all_patients_and_means]
# all_means = [m for sublist in all_patients_and_means.values() for m in sublist]
# all_means = []
# for pat_id in all_patients_and_means.keys():
#   n = 0
#   tmp = []
#   for intensity_bin in all_patients_and_means[pat_id].keys():
#     qty = all_patients_and_means[pat_id][intensity_bin]
#     n += qty
#     tmp.append(float(float(intensity_bin) * qty))
#   all_means.extend(np.divide(tmp, n).tolist())
# all_means = [float(intensity_bin) for intensity_bin,count in all_patients_and_means[pat_id].items() for pat_id in all_patients_and_means.keys()]
all_means_unordered = {}
for p in all_patients_and_means.keys():
  for k in all_patients_and_means[p].keys():
    if not all_means_unordered.has_key(k):
      all_means_unordered[k] = 0
    all_means_unordered[k] += all_patients_and_means[p][k]
all_means = collections.OrderedDict(sorted(all_means_unordered.items(), key=lambda t: t[0]))

# all_means_gt_zero = [m for m in all_means if m > 0]
# all_means_gt_zero = []
# for pat_id in all_patients_and_means.keys():
#   for intensity_bin in all_patients_and_means[pat_id].keys():
#     if (intensity_bin > 0):
#       all_means_gt_zero.append(intensity_bin)
# all_means_gt_zero = [intensity_bin for intensity_bin in all_means if intensity_bin > 0]
# print all_means_gt_zero
# print all_means
all_means_gt_zero = copy.copy(all_means)
all_means_gt_zero.pop('0.0', None)
# print all_means

# sys.exit()


## Render the histogram. ##
def render_histogram(X, bin_size, title, file_path):
  if len(X) > 0:
    print "* Rendering: " + title
    divisor = float(bin_size)
    end_range = bin_size + 1
    
    fig = plt.figure()
    plt.hist(X, bins=[x / divisor for x in range(0, end_range)])  # [0,1] by 0.1
    plt.title(title)
    plt.savefig(file_path)


def render_histogram_prebinned(Y, X_lab, title, file_path):
  if len(X_lab) > 0:
    print "* Rendering: " + title
    
    fig = plt.figure()
    # plt.hist(X, bins=[x / divisor for x in range(0, end_range)])  # [0,1] by 0.1
    X_pos = np.arange(len(X_lab))
    
    plt.bar(X_pos, Y, align='center', alpha=0.5)
    plt.xticks(X_pos, X_lab)
    plt.title(title)
    plt.savefig(file_path)



all_pats_dirname = 'all_patients'
if not os.path.exists(all_pats_dirname):
    os.makedirs(all_pats_dirname)

# 10 bins
# render_histogram(all_means, 10, 'Edge patch intensity distribution for all means (' + str(binSize) + ' bins)', all_pats_dirname + os.sep + 'all_means-10bins.png')
# render_histogram(all_means_gt_zero, 10, 'Edge patch intensity distribution for means > 0 (' + str(binSize) + ' bins)', all_pats_dirname + os.sep + 'all_means_gt_zero-10bins.png')

# 100 bins
# render_histogram(all_means, 100, 'Edge patch intensity distribution for all means (' + str(binSize) + ' bins)', all_pats_dirname + os.sep + 'all_means-' + str(binSize) + 'bins.png')
# render_histogram(all_means_gt_zero, 100, 'Edge patch intensity distribution for means > 0 (' + str(binSize) + ' bins)', all_pats_dirname + os.sep + 'all_means_gt_zero-' + str(binSize) + 'bins.png')
render_histogram_prebinned(all_means.values(), all_means.keys(), 'Edge patch intensity distribution for all means (' + str(binSize) + ' bins)', all_pats_dirname + os.sep + 'all_means-' + str(binSize) + 'bins.png')
render_histogram_prebinned(all_means_gt_zero.values(), all_means_gt_zero.keys(), 'Edge patch intensity distribution for means > 0 (' + str(binSize) + ' bins)', all_pats_dirname + os.sep + 'all_means_gt_zero-' + str(binSize) + 'bins.png')


# Per-patient distributions.
per_pat_dirname = 'per-patient'
if not os.path.exists(per_pat_dirname):
    os.makedirs(per_pat_dirname)
# Get the distinct set of patient IDs.
# all_pats = list(set([p[0] for p in all_patients_and_means]))
all_pats = all_patients_and_means.keys()

def thin_Xlab((xlab, mod_val)):
  if mod_val % 2:
    return ""
  return xlab

# Loop over all patients and output their patient-specific intensity histograms.
for pat_id in all_pats:
  # all_means = [pm[1] for pm in all_patients_and_means if pm[0] == pat_id]
  all_means = all_patients_and_means[pat_id]
  # render_histogram(all_means, 10, 'Edge patch intensity distribution for all means for: ' + pat_id + ' (' + str(binSize) + ' bins)', per_pat_dirname + os.sep + 'all_means-10bins-' + pat_id + '.png')
  # if len(all_means.keys()) > 20:
  #   m = map(thin_Xlab, zip(all_means_gt_zero.keys(), range(len(all_means_gt_zero.keys()))))
  #   print m
  #   print len(m)
  #   render_histogram_prebinned(all_means.values(), m, 'Edge patch intensity distribution for all means for: ' + pat_id + ' (' + str(binSize) + ' bins)', per_pat_dirname + os.sep + 'all_means-' + str(binSize) + 'bins-' + pat_id + '.png')
  # else:
  render_histogram_prebinned(all_means.values(), all_means.keys(), 'Edge patch intensity distribution for all means for: ' + pat_id + ' (' + str(binSize) + ' bins)', per_pat_dirname + os.sep + 'all_means-' + str(binSize) + 'bins-' + pat_id + '.png')

  # all_means_gt_zero = [m for m in all_means if m > 0]
  # render_histogram(all_means_gt_zero, 10, 'Edge patch intensity distribution for means > 0 for: ' + pat_id + ' (' + str(binSize) + ' bins)', per_pat_dirname + os.sep + 'all_means_gt_zero-10bins-' + pat_id + '.png')
  all_means_gt_zero = copy.copy(all_means)
  all_means_gt_zero.pop('0.0', None)
  # if len(all_means.keys()) > 20:
  #   m = map(thin_Xlab, zip(all_means_gt_zero.keys(), range(len(all_means_gt_zero.keys()))))
  #   print m
  #   render_histogram_prebinned(all_means_gt_zero.values(), m, 'Edge patch intensity distribution for means > 0 for: ' + pat_id + ' (' + str(binSize) + ' bins)', per_pat_dirname + os.sep + 'all_means_gt_zero-' + str(binSize) + 'bins-' + pat_id + '.png')
  # else:
  render_histogram_prebinned(all_means_gt_zero.values(), all_means_gt_zero.keys(), 'Edge patch intensity distribution for means > 0 for: ' + pat_id + ' (' + str(binSize) + ' bins)', per_pat_dirname + os.sep + 'all_means_gt_zero-' + str(binSize) + 'bins-' + pat_id + '.png')
