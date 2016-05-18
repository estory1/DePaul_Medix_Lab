#!/opt/local/bin/python2.7

import sys
import os
import numpy as np
import fnmatch
import math
import traceback
from time import strftime
from multiprocessing import Process, Lock
import multiprocessing as mp
import pandas as pd
from PIL import Image




# def read_images_and_labels(images_root_dir_path, images_file_glob, master_join4_file_path, patient_dicom_root_dir_path, patient_dicom_file_glob="*.csv"):
def read_images_and_labels(images_root_dir_path, images_file_glob):
  """
    Reads a supervised image dataset. More specifically:

    1) Reads the LIDC images from a specified folder with a specified file glob.
    2) Reads the LIDC malignancy ratings for each image, given DICOM metadata CSV files and the master_join4 file, converting missing values to values outside the LIDC's valid interval [1,5] for malignancy.
    3) Ensures, by ordering, the malignancy is associated with the appropriate image.
    4) Returns the image data and malignancy series each   as NumPy arrays.
  """
  # EXAMPLE VALUES (row 40086 in master_join4.csv):
  # StudyInstanceUID: "1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178"
  # SeriesInstanceUID: "1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192"
  # XmlStudyNode: "LIDC-IDRI-0001"
  # imageSOP_UID: "1.3.6.1.4.1.14519.5.2.1.6279.6001.261151233960269013402330853013"
  # DICOM_original_fullPath: "D:\LIDC\LIDC_Complete_20141106\LIDC-IDRI\LIDC-IDRI-0001\1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178\1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192\000003.dcm"
  # roi_id: 5
  # malignancy: 5
  #
  # TO GET LABELS (P-code):
  #   read each patient's DICOM data file into a hashtable, with XmlStudyNode as key and the file's Pandas matrix as the value.
  #   read the master_join4 file
  #   for each image file:
  #     imageSOPUID = imageSOPUID for match in DICOM file on: XmlStudyNode, StudyInstanceUID, SeriesInstanceUID, DICOM_original_fullPath.split('\')[-1]
  #     malignancy = malignancy for match in master_join4 file on: imageSOPUID, roi_Id
  import pandas as pd
  from PIL import Image

  # Get the full image list.
  esprint("Reading images: " + images_root_dir_path + os.path.sep + images_file_glob)
  allFilePaths = list(find_files(images_root_dir_path, images_file_glob))
  esprint("Image count: " + str(len(allFilePaths)))

  if len(allFilePaths) == 0:
    print("read_all_images: No files found.")
  else:
    #   read each patient's DICOM data file into a hashtable, with XmlStudyNode as key and the file's Pandas matrix as the value.
    # dicom_files = {}
    # esprint("Reading DICOM metadata files: " + patient_dicom_root_dir_path + os.path.sep + patient_dicom_file_glob)
    # for dicom_csv_filepath in list(find_files(patient_dicom_root_dir_path, patient_dicom_file_glob)): # "imageSOP_UID-filePath-dicominfo-LIDC-IDRI-0001.csv"
    #   dicom_filename = dicom_csv_filepath.split(os.path.sep)[-1]
    #   with open(dicom_csv_filepath, 'rb') as f:
    #     df = pd.read_csv(f, delimiter=',')
    #     dicom_files[dicom_filename] = pd.DataFrame({  "StudyInstanceUID": df.StudyInstanceUID,
    #                                                   "SeriesInstanceUID": df.SeriesInstanceUID,
    #                                                   "XmlStudyNode": df.XmlStudyNode,
    #                                                   "FileNode": df.FileNode,
    #                                                   "imageSOP_UID": df.imageSOP_UID,
    #                                                   "DICOM_original_fullPath": df.DICOM_original_fullPath })

    # #   read the master_join4 file
    # esprint("Reading semantic ratings from: " + master_join4_file_path)
    # with open(master_join4_file_path, 'rb') as f:
    #   master_join4 = pd.read_csv(f, delimiter=",", header=1, usecols=["imageSOP_UID", "roi_Id", "readingSession_Id", "malignancy"])
    #   master_join4 = master_join4.fillna(-1)

    # Init image array.
    image_data = []
    # malignancy_series = []
    output = mp.Queue()

    # def wrappedMultiProcAllFiles(image_data, malignancy_series, dicom_files, master_join4, allFilePaths, lock, output):
    def wrappedMultiProcAllFiles(image_data, allFilePaths, lock, output):
      """
      Theoretically prevents the stupid hang seen here: http://stackoverflow.com/questions/15314189/python-multiprocessing-pool-hangs-at-join
      """
      try:
        # multiProcAllFiles(image_data, malignancy_series, dicom_files, master_join4, allFilePaths, lock, output)
        multiProcAllFiles(image_data, allFilePaths, lock, output)
      except Exception, e:
        print('%s' % (traceback.print_exc()))
      # esprint("wrappedMultiProcAllFiles: should return")
      return


    # def multiProcAllFiles(image_data, malignancy_series, dicom_files, master_join4, allFilePaths, lock, output):
    def multiProcAllFiles(image_data, allFilePaths, lock, output):
      esprint("Processing images and finding their corresponding malignancy values for " + str(len(allFilePaths)) + " images..." )
      for filePath in allFilePaths:
        # Read the image pixels.
        im = Image.open(filePath)

        # Append results in a consistent order.
        lock.acquire()
        # Convert image to NumPy array for full-image stats.
        npim = np.array(im)

        width, height = im.size
        # Intensity summary
        intensity_mean = npim.mean()
        intensity_med = np.median(npim)
        intensity_std = npim.std()
        intensity_min = npim.min()
        intensity_max = npim.max()

        image_data.append([filePath, width, height, intensity_mean, intensity_med, intensity_std, intensity_min, intensity_max])

        lock.release()

      output.put(image_data)
      esprint("multiProcAllFiles: returning...")
      return


    # Multiprocess the files; takes hours with single processing.
    lock = Lock()
    numCpusToUse = int(math.floor(mp.cpu_count()))
    # processes = [mp.Process(target=multiProcAllFiles, args=(image_data, malignancy_series, dicom_files, master_join4, allFilePaths, lock, output)) for x in range(numCpusToUse)]
    processes = []
    num_files = len(allFilePaths)
    for i in range(numCpusToUse):
      idxMin = int(math.floor( (max(0,i)/float(numCpusToUse)) * num_files ))
      idxMax = int(math.floor( (max(0,i+1)/float(numCpusToUse)) * num_files ))
      print("idxMin: " + str(idxMin) + "; idxMax: " + str(idxMax) + "; num_files: " + str(num_files))
      processes.append(mp.Process(target=wrappedMultiProcAllFiles, args=(image_data, allFilePaths[idxMin:idxMax], lock, output)))
    for p in processes:
      p.start()
    esprint("Processes started")

    for p in processes:
      # Join the processes.
      image_result = output.get()
      # esprint(image_result)
      # image_data = image_data + image_result
      image_data.extend(image_result)

    # All processes completed, results collected, so return.
    return image_data



def esprint(msg):
  print("[%s] %s" % (strftime("%Y%m%d-%H%M%S"), msg))

# Find files by file name pattern: http://stackoverflow.com/a/2186673
# Since this is a generator function (uses yield), processing can occur on each yield's return, not upon complete execution of this function.
def find_files(dirRoot, patt):
  for root, dirs, files in os.walk(dirRoot):
    for baseName in files:
      if fnmatch.fnmatch(baseName, patt):
        filename = os.path.join(root, baseName)
        fSz = os.path.getsize(filename)
        # Ignore 0 byte files, but yield > 0 byte files.
        if fSz == 0:
          print("* 0 byte file found: " + filename)
        else:
          yield filename


def compute5NumSummary(image_data):
  n = 0

  arr_filePath  = [ l[0] for l in [p for p in image_data] ]
  arr_width     = np.array([ l[1] for l in [p for p in image_data] ])
  arr_height    = np.array([ l[2] for l in [p for p in image_data] ])
  arr_int_mean  = np.array([ l[3] for l in [p for p in image_data] ])
  arr_int_med   = np.array([ l[4] for l in [p for p in image_data] ])
  arr_int_std   = np.array([ l[5] for l in [p for p in image_data] ])
  arr_int_min   = np.array([ l[6] for l in [p for p in image_data] ])
  arr_int_max   = np.array([ l[7] for l in [p for p in image_data] ])

  # esprint( [ l[0] for l in [p for p in image_data] ] )    # [j for j in [i for i in [ l for l in 

  # Use Pandas to compute 5num summary: http://stackoverflow.com/a/13635260
  pd_s_width    = pd.Series(arr_width)
  pd_s_height   = pd.Series(arr_height)
  pd_s_int_mean = pd.Series(arr_int_mean)
  pd_s_int_med  = pd.Series(arr_int_med)
  pd_s_int_std  = pd.Series(arr_int_std)
  pd_s_int_min  = pd.Series(arr_int_min)
  pd_s_int_max  = pd.Series(arr_int_max)

  print("----- width (per crop image) -----")
  print(pd_s_width.describe())
  print("----- height (per crop image) -----")
  print(pd_s_height.describe())
  print("----- intensity: mean (per crop image) -----")
  print(pd_s_int_mean.describe())
  print("----- intensity: median (per crop image) -----")
  print(pd_s_int_med.describe())
  print("----- intensity: std dev (per crop image) -----")
  print(pd_s_int_std.describe())
  print("----- intensity: min (per crop image) -----")
  print(pd_s_int_min.describe())
  print("----- intensity: max (per crop image) -----")
  print(pd_s_int_max.describe())


  # Save a list of crop images with < 25 pixels for cleanup.
  arr_err_crop_toosmall = [ l[0:3] for l in [p for p in image_data] if (l[1] * l[2]) < 25 ]
  # esprint(arr_err_crop_toosmall)
  import csv
  filename_toosmall = 'toosmall.tsv'
  esprint("Writing: " + filename_toosmall)
  with open(filename_toosmall, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['File path', 'Width', 'Height'])
    writer.writerows(arr_err_crop_toosmall)


  # Recompute 5# summary, excluding the images which are too small.
  arr_width     = np.array([ l[1] for l in [p for p in image_data] if (l[1] * l[2]) >= 25 ])
  arr_height    = np.array([ l[2] for l in [p for p in image_data] if (l[1] * l[2]) >= 25 ])
  arr_int_mean  = np.array([ l[3] for l in [p for p in image_data] if (l[1] * l[2]) >= 25 ])
  arr_int_med   = np.array([ l[4] for l in [p for p in image_data] if (l[1] * l[2]) >= 25 ])
  arr_int_std   = np.array([ l[5] for l in [p for p in image_data] if (l[1] * l[2]) >= 25 ])
  arr_int_min   = np.array([ l[6] for l in [p for p in image_data] if (l[1] * l[2]) >= 25 ])
  arr_int_max   = np.array([ l[7] for l in [p for p in image_data] if (l[1] * l[2]) >= 25 ])
  pd_s_width    = pd.Series(arr_width)
  pd_s_height   = pd.Series(arr_height)
  pd_s_int_mean = pd.Series(arr_int_mean)
  pd_s_int_med  = pd.Series(arr_int_med)
  pd_s_int_std  = pd.Series(arr_int_std)
  pd_s_int_min  = pd.Series(arr_int_min)
  pd_s_int_max  = pd.Series(arr_int_max)

  print("----- width (per crop image; >= 25pixels) -----")
  print(pd_s_width.describe())
  print("----- height (per crop image); >= 25pixels -----")
  print(pd_s_height.describe())
  print("----- intensity: mean (per crop image); >= 25pixels -----")
  print(pd_s_int_mean.describe())
  print("----- intensity: median (per crop image); >= 25pixels -----")
  print(pd_s_int_med.describe())
  print("----- intensity: std dev (per crop image); >= 25pixels -----")
  print(pd_s_int_std.describe())
  print("----- intensity: min (per crop image); >= 25pixels -----")
  print(pd_s_int_min.describe())
  print("----- intensity: max (per crop image); >= 25pixels -----")
  print(pd_s_int_max.describe())




if __name__ == "__main__":
  esprint([sys.argv[1], sys.argv[2]])
  image_data = read_images_and_labels(sys.argv[1], sys.argv[2])
  compute5NumSummary(image_data)
  esprint("Done.")

