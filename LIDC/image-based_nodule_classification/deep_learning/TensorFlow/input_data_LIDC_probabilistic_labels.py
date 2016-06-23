#!/opt/local/bin/python2.7

# 20151119
# Evan Story (estory1@gmail.com)
#
# Reads LIDC data into data structures that can be trained/validated/tested upon by the TensorFlow MNIST tutorial's deep convolutional neural net code.
# Borrows heavily from the TensorFlow input_data.py script.
# 

"""Functions for downloading and reading LIDC data."""
from __future__ import print_function
import sys
import gzip
import os
import urllib
import numpy
import numpy as np
import traceback

import fnmatch
from time import strftime
import math
from multiprocessing import Process, Lock
import multiprocessing as mp

SOURCE_URL = 'NOT_IMPLEMENTED'


# def maybe_download(filename, work_directory):
#   """Download the data from the LIDC website, unless it's already here."""
#   if not os.path.exists(work_directory):
#     os.mkdir(work_directory)
#   filepath = os.path.join(work_directory, filename)
#   if not os.path.exists(filepath):
#     filepath, _ = urllib.urlretrieve(SOURCE_URL + filename, filepath)
#     statinfo = os.stat(filepath)
#     print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
#   return filepath


def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)


def extract_images(filename):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError(
          'Invalid magic number %d in MNIST image file: %s' %
          (magic, filename))
    num_images = _read32(bytestream)
    print("num_images: " + str(num_images))
    rows = _read32(bytestream)
    print("rows: " + str(rows))
    cols = _read32(bytestream)
    print("cols: " + str(cols))
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    print("1: numpy.shape(data):" + str(numpy.shape(data)))
    data = data.reshape(num_images, rows, cols, 1)
    print("2: numpy.shape(data):" + str(numpy.shape(data)))
    return data


def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def extract_labels(filename, one_hot=False):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError(
          'Invalid magic number %d in MNIST label file: %s' %
          (magic, filename))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels)
    return labels



class DataSets(object):
  pass

class DataSet(object):
  def __init__(self, images, labels, fake_data=False):
    if fake_data:
      self._num_examples = 10000
    else:
      assert images.shape[0] == labels.shape[0], (
          "images.shape: %s labels.shape: %s" % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]
      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      esprint('** DataSet ** images.shape = ' + str(images.shape))
      # assert images.shape[3] == 1
      # images = images.reshape(images.shape[0],
      #                         images.shape[1] * images.shape[2])
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(numpy.float32)
      images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
  @property
  def images(self):
    return self._images
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1.0 for _ in xrange(784)]
      fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


# Reads the LIDC dataset and splits the data into training, validation, and testing sets.
def read_data_sets(images_root_dir_path, images_file_glob, master_join4_file_path, patient_dicom_root_dir_path, patient_dicom_file_glob, fake_data=False, one_hot=False):
  # data_sets = DataSets()
  # if fake_data:
  #   data_sets.train = DataSet([], [], fake_data=True)
  #   data_sets.validation = DataSet([], [], fake_data=True)
  #   data_sets.test = DataSet([], [], fake_data=True)
  #   return data_sets

  # TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  # TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  # TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  # TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
  # VALIDATION_SIZE = 5000
  # local_file = maybe_download(TRAIN_IMAGES, train_dir)
  # train_images = extract_images(local_file)
  # local_file = maybe_download(TRAIN_LABELS, train_dir)
  # train_labels = extract_labels(local_file, one_hot=one_hot)
  # local_file = maybe_download(TEST_IMAGES, train_dir)
  # test_images = extract_images(local_file)
  # local_file = maybe_download(TEST_LABELS, train_dir)
  # test_labels = extract_labels(local_file, one_hot=one_hot)

  # validation_images = train_images[:VALIDATION_SIZE]
  # validation_labels = train_labels[:VALIDATION_SIZE]

  # train_images = train_images[VALIDATION_SIZE:]
  # train_labels = train_labels[VALIDATION_SIZE:]

  images, labels = read_images_and_labels(images_root_dir_path, images_file_glob, master_join4_file_path, patient_dicom_root_dir_path, patient_dicom_file_glob)
  n = numpy.shape(images)[0]
  TRAIN_SIZE = int(math.floor(n * 0.7))
  VALIDATION_SIZE = int(math.floor(TRAIN_SIZE * 0.1))

  train_images = images[0:(TRAIN_SIZE - VALIDATION_SIZE)]
  train_labels = labels[0:(TRAIN_SIZE - VALIDATION_SIZE)]

  validation_images = images[(TRAIN_SIZE - VALIDATION_SIZE):TRAIN_SIZE]
  validation_labels = labels[(TRAIN_SIZE - VALIDATION_SIZE):TRAIN_SIZE]

  test_images = images[TRAIN_SIZE:]
  test_labels = labels[TRAIN_SIZE:]

  # data_sets.train = DataSet(train_images, train_labels)
  # data_sets.validation = DataSet(validation_images, validation_labels)
  # data_sets.test = DataSet(test_images, test_labels)
  # return data_sets
  return [(train_images, train_labels), (validation_images, validation_labels), (test_images, test_labels)]



def read_images_and_labels(images_root_dir_path, images_file_glob, master_join4_file_path, patient_dicom_root_dir_path, patient_dicom_file_glob="*.csv"):
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

  if len(allFilePaths) == 0:
    print("read_all_images: No files found.")
  else:
    #   read each patient's DICOM data file into a hashtable, with XmlStudyNode as key and the file's Pandas matrix as the value.
    dicom_files = {}
    esprint("Reading DICOM metadata files: " + patient_dicom_root_dir_path + os.path.sep + patient_dicom_file_glob)
    for dicom_csv_filepath in list(find_files(patient_dicom_root_dir_path, patient_dicom_file_glob)): # "imageSOP_UID-filePath-dicominfo-LIDC-IDRI-0001.csv"
      dicom_filename = dicom_csv_filepath.split(os.path.sep)[-1]
      with open(dicom_csv_filepath, 'rb') as f:
        df = pd.read_csv(f, delimiter=',')
        dicom_files[dicom_filename] = pd.DataFrame({  "StudyInstanceUID": df.StudyInstanceUID,
                                                      "SeriesInstanceUID": df.SeriesInstanceUID,
                                                      "XmlStudyNode": df.XmlStudyNode,
                                                      "FileNode": df.FileNode,
                                                      "imageSOP_UID": df.imageSOP_UID,
                                                      "DICOM_original_fullPath": df.DICOM_original_fullPath })

    #   read the master_join4 file
    esprint("Reading semantic ratings from: " + master_join4_file_path)
    with open(master_join4_file_path, 'rb') as f:
      # master_join4 = pd.read_csv(f, delimiter=",", header=1, usecols=["imageSOP_UID", "roi_Id", "readingSession_Id", "malignancy"])
      master_join4 = pd.read_csv(f, usecols=["imageSOP_UID", "DL_5_1", "DL_5_2", "DL_5_3", "DL_5_4", "DL_5_5"])   #  delimiter=",", header=1,
      master_join4 = master_join4.fillna(-1)


    # for index, data in (master_join4 [ master_join4["imageSOP_UID"] == "1.3.6.1.4.1.14519.5.2.1.6279.6001.796308805230198385262306509458" ][ ["DL_5_1", "DL_5_2", "DL_5_3", "DL_5_4", "DL_5_5"] ] ).iterrows():
    #   print ( data.tolist() )
    # sys.exit(0)

    # Init image array.
    image_data = []
    malignancy_series = []
    output = mp.Queue()

    def wrappedMultiProcAllFiles(image_data, malignancy_series, dicom_files, master_join4, allFilePaths, lock, output):
      """
      Theoretically prevents the stupid hang seen here: http://stackoverflow.com/questions/15314189/python-multiprocessing-pool-hangs-at-join
      """
      try:
        multiProcAllFiles(image_data, malignancy_series, dicom_files, master_join4, allFilePaths, lock, output)
      except Exception, e:
        print('%s' % (traceback.print_exc()))
      # esprint("wrappedMultiProcAllFiles: should return")
      return


    def multiProcAllFiles(image_data, malignancy_series, dicom_files, master_join4, allFilePaths, lock, output):
      esprint("Processing images and finding their corresponding malignancy values for " + str(len(allFilePaths)) + " images..." )
      for filePath in allFilePaths:
        # Read the image pixels.
        im = Image.open(filePath)

        # Get the label:
        # 1) extract lookup components.
        fpArr = filePath.split(os.path.sep)
        XmlStudyNode = fpArr[-4]
        StudyInstanceUID = fpArr[-3]
        SeriesInstanceUID = fpArr[-2]
        fpFilename = fpArr[-1]
        roi_id = int(fpFilename.split('-')[1].split('_')[1])
        rs_id = int(fpFilename.split('-')[2].split('_')[1])
        orig_dicom_filename = fpFilename.split('-')[0] + ".dcm"

        # esprint("1")

        try:
          # 2) query DICOM extract file for the imageSOP_UID.
          dicom_data = dicom_files["imageSOP_UID-filePath-dicominfo-"+XmlStudyNode+".csv"]
          # finding first match on criteria: http://stackoverflow.com/questions/9868653/find-first-list-item-that-matches-criteria
          idx_row_containing_imagesSOP_UID = next( i for i in range(0,len(dicom_data)) if orig_dicom_filename in dicom_data["DICOM_original_fullPath"][i] and (dicom_data["XmlStudyNode"][i] == XmlStudyNode) and (dicom_data["StudyInstanceUID"][i] == StudyInstanceUID) and (dicom_data["SeriesInstanceUID"][i] == SeriesInstanceUID) )
          imageSOP_UID = dicom_data["imageSOP_UID"][idx_row_containing_imagesSOP_UID]

          # esprint("2")

          # 3) query the semantic ratings file (master_join4.csv) for the malignancy.
          # esprint('filePath: ' + filePath + ', imageSOP_UID: ' + imageSOP_UID)
          # idx_row_containing_malignancy = next( i for i in range(0,len(master_join4)) if (imageSOP_UID == master_join4["imageSOP_UID"][i]) and (roi_id == master_join4["roi_Id"][i]) and (rs_id == master_join4["readingSession_Id"][i]) )
          # idx_row_containing_malignancy = next( i for i in range(0,len(master_join4)) if (imageSOP_UID == master_join4["imageSOP_UID"][i]) )

          # idx_row_containing_malignancy = -1
          # for i in range(0,len(master_join4)):
          #   if (imageSOP_UID == master_join4["imageSOP_UID"][i]):
          #     idx_row_containing_malignancy = i

          # malignancy = master_join4["malignancy"][ idx_row_containing_malignancy ]
          
          malignancy = None
          # if idx_row_containing_malignancy != -1:
          #   esprint("3: " + str(idx_row_containing_malignancy))
            # malignancy = master_join4["DL_5_1", "DL_5_2", "DL_5_3", "DL_5_4", "DL_5_5"][ idx_row_containing_malignancy ]
            # malignancy = master_join4[ ["DL_5_1","DL_5_2","DL_5_3","DL_5_4","DL_5_5"] ][ imageSOP_UID ]  #["DL_5_1", "DL_5_2", "DL_5_3", "DL_5_4", "DL_5_5"]
            # malignancy = master_join4 [ master_join4["imageSOP_UID"] == "1.3.6.1.4.1.14519.5.2.1.6279.6001.796308805230198385262306509458" ][ ["DL_5_1", "DL_5_2", "DL_5_3", "DL_5_4", "DL_5_5"] ] 
          for index, data in (master_join4 [ master_join4["imageSOP_UID"] == imageSOP_UID ][ ["DL_5_1", "DL_5_2", "DL_5_3", "DL_5_4", "DL_5_5"] ] ).iterrows():
            # print ( data.tolist() )
            malignancy = data.tolist()

            # esprint("4: malignancy = " + str(malignancy))
        except StopIteration, e:
          esprint("** StopIteration **: " + str(e) + " -- malignancy: " + str([]) + " -- imageSOP_UID: " + imageSOP_UID + " -- filePath: " + filePath)
          continue
        except Exception, e:
          raise e
        else:
          
          if malignancy != None:
            # Append results in a consistent order.
            lock.acquire()
            # esprint("Appending: malignancy: " + str(malignancy) + " -- imageSOP_UID: " + imageSOP_UID + " -- filePath: " + filePath)
            # Append the image pixels and malignancy value to their respective arrays.
            # image_data.append( [ (numpy.transpose(numpy.reshape(im, -1))).tolist() ] )
            image_data.append( (numpy.reshape(im, -1)).tolist() )
            malignancy_rating = [0,0,0,0,0]
            # if malignancy in range(1,6):
            #   malignancy_rating[int(malignancy)-1] = 1
            malignancy_rating = malignancy
            # esprint("malignancy: " + str(malignancy_rating))
            malignancy_series.append(malignancy_rating)
            lock.release()

      # print(malignancy_series)
      output.put([image_data, malignancy_series])
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
      # print(allFilePaths[idxMin:idxMax])
      processes.append(mp.Process(target=wrappedMultiProcAllFiles, args=(image_data, malignancy_series, dicom_files, master_join4, allFilePaths[idxMin:idxMax], lock, output)))
    for p in processes:
      p.start()
    esprint("Processes started")
    # for p in processes:
    #   p.join()
    # esprint("Processes joined")
    for p in processes:
      image_result, label_result = output.get()
      # esprint(image_result)
      image_data = image_data + image_result
      malignancy_series = malignancy_series + label_result
      esprint("np.shape(image_data): " + str(np.shape(image_data)))
      esprint("np.shape(malignancy_series): " + str(np.shape(malignancy_series)))
      # esprint("Appended")

    # Structure the data so it can be processed the same way as the MNIST data.
    # esprint(image_data)
    # esprint(malignancy_series)
    esprint("Reshaping image data and converting malignancy data to NumPy...")
    np_image_data = numpy.array(image_data, dtype=numpy.uint8)
    np_malignancy_series = numpy.array(malignancy_series, dtype=numpy.uint8)


    # WARNING: THIS MIGHT NOT WORK. OUTPUT STRUCTURE IS DIFFERENT, BUT AM TRYING ANYWAY (supposedly doesn't make a difference: http://stackoverflow.com/a/5954747):
    #   MINE (as row vectors):
          # >>> lidc.read_all_images("../data/LIDC/resized_images/LIDC-IDRI-0001/", "*.tiff")
          # array([[[ 57,  57,  57, ..., 103, 103, 103]],

          #        [[ 57,  57,  57, ..., 101, 101, 101]],

          #        [[ 57,  57,  57, ..., 101, 101, 101]],

          #        ..., 
          #        [[ 74,  74,  74, ...,  74,  74,  74]],

          #        [[ 70,  70,  70, ...,  70,  70,  70]],

          #        [[ 63,  63,  63, ...,  63,  63,  63]]], dtype=uint8)

    #   ORIGINAL (as column vectors):
          # >>> b = np.array([[ [[3], [2], [4]], [[5],[7],[2]] ]], dtype=np.uint8)
          # >>> b
          # array([[[[3],
          #          [2],
          #          [4]],

          #         [[5],
          #          [7],
          #          [2]]]], dtype=uint8)

    # print(np_image_data)
    # print(np_malignancy_series)
    esprint('np_image_data.shape = ' + str(np_image_data.shape) + "; np_malignancy_series.shape = " + str(np_malignancy_series.shape))
    return [np_image_data, np_malignancy_series]



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