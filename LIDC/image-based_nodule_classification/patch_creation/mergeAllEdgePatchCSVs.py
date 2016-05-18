#!/opt/local/bin/python2.7
import sys
import os
import fnmatch
import glob
import csv
from time import strftime

# Merges all edge patch CSV files into a single, monstrous CSV file for import by csvsql.


def esprint(msg):
  print("[%s] %s" % (strftime("%Y%m%d-%H%M%S"), msg))

## Get all the file paths. ##
def find_patch_summary_files(curr_folder_path, summary_file_glob):
  """Navigates to e.g.:

  '../../../LIDC_Complete_20141106/LIDC-IDRI-edge_patches/LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192/patches/'
  
  and returns the CSV files there, without enumerating all the image files in the bin/ and orig/ folders beneath that level."
  """
  return glob.glob (curr_folder_path + os.sep + '*'+os.sep+'*'+os.sep+'*'+os.sep+'patches'+os.sep + summary_file_glob)  # we're globbing on e.g.: ('../../../LIDC_Complete_20141106/LIDC-IDRI-edge_patches/*/*/*/patches/*.csv')


def writeCSVRowsForFile(writer, csv_file_path):
  [xml_study_node, study_instance_uid, series_instance_uid] = csv_file_path.split(os.sep)[-5:-2]

  with open(csv_file_path, 'rb') as input_file_handle:
    reader_handle = csv.reader(input_file_handle, delimiter=',', quotechar='"')

    # Skip header row.    
    reader_handle.next()

    # Loop over each row in the input CSV file.
    for row in reader_handle:
      # Parse each row for its atomic values to import into Postgres cols.
      [filename, mean, stddev] = row
      [dicom_filename, patch_bbox_str, roi_str, rs_str] = filename.split('-')
      [upper_left_str, ignore, lower_right_str] = patch_bbox_str.split('_')
      [upper_left_x, upper_left_y] = upper_left_str.split(',')
      [lower_right_x, lower_right_y] = lower_right_str.split(',')
      roi_id = roi_str.split('_')[1]
      rs_id = rs_str.split('_')[1].split('.')[0]

      # print [filename, roi_id, rs_id]
      arr_to_write = [xml_study_node, study_instance_uid, series_instance_uid, dicom_filename + '.dcm', int(upper_left_x), int(upper_left_y), int(lower_right_x), int(lower_right_y), int(roi_id), int(rs_id), float(mean), float(stddev), csv_file_path]
      # if (float(mean) > 0):
      #   print arr_to_write

      writer.writerow(arr_to_write)



if __name__ == "__main__":
  with open('edge_patch_metadata-ALL_PATS.csv', 'wb') as output_file_handle:
    writer_handle = csv.writer(output_file_handle, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

    all_csv_file_paths = find_patch_summary_files(sys.argv[1], sys.argv[2])
    
    # Write header.
    writer_handle.writerow(['xml_study_node', 'study_instance_uid', 'series_instance_uid', 'dicom_filename', 'upper_left_x', 'upper_left_y', 'lower_right_x', 'lower_right_y', 'roi_id', 'rs_id', 'mean', 'std_dev', 'edge_patch_stats_csv_file_path'])   # see genPatchSet.m

    # Loop over each input CSV file.
    for csv_file_path in all_csv_file_paths:
      esprint('Processing: ' + csv_file_path)
      # Write rows to the output CSV file for each input CSV file.
      writeCSVRowsForFile(writer_handle, csv_file_path)