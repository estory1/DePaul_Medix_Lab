import os
import sys
import timeit

import numpy
import numpy as np

import theano
import theano.tensor as T

import traceback

import fnmatch
from time import strftime
import math
from multiprocessing import Process, Lock
import multiprocessing as mp



# def load_data_LIDC(dataset):
def load_data(images_root_dir_path, images_file_glob, master_join4_file_path, patient_dicom_root_dir_path, patient_dicom_file_glob):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # # Download the MNIST dataset if it is not present
    # data_dir, data_file = os.path.split(dataset)
    # if data_dir == "" and not os.path.isfile(dataset):
    #     # Check if dataset is in the data directory.
    #     new_path = os.path.join(
    #         os.path.split(__file__)[0],
    #         "..",
    #         "data",
    #         dataset
    #     )
    #     if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
    #         dataset = new_path

    # if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
    #     import urllib
    #     origin = (
    #         'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    #     )
    #     print 'Downloading data from %s' % origin
    #     urllib.urlretrieve(origin, dataset)

    # print '... loading data'

    # # Load the dataset
    # f = gzip.open(dataset, 'rb')
    # train_set, valid_set, test_set = cPickle.load(f)
    # f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        esprint("shapes: " + str(np.shape(data_x)) + "; " + str(np.shape(data_y)))
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    # Read the LIDC data.
    train_set, valid_set, test_set = read_data_sets(images_root_dir_path, images_file_glob, master_join4_file_path, patient_dicom_root_dir_path, patient_dicom_file_glob)

    # Set up the training, validation, and test sets.
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def read_data_sets(images_root_dir_path, images_file_glob, master_join4_file_path, patient_dicom_root_dir_path, patient_dicom_file_glob, fake_data=False, one_hot=False):

  images, labels = read_images_and_labels(images_root_dir_path, images_file_glob, master_join4_file_path, patient_dicom_root_dir_path, patient_dicom_file_glob)

  n = numpy.shape(images)[0]
  TRAIN_SIZE = math.floor(n * 0.7)
  VALIDATION_SIZE = math.floor(TRAIN_SIZE * 0.1)
  # TEST_SIZE = n - TRAIN_SIZE

  train_images = images[0:(TRAIN_SIZE - VALIDATION_SIZE)]
  train_labels = labels[0:(TRAIN_SIZE - VALIDATION_SIZE)]

  validation_images = images[(TRAIN_SIZE - VALIDATION_SIZE):TRAIN_SIZE]
  validation_labels = labels[(TRAIN_SIZE - VALIDATION_SIZE):TRAIN_SIZE]

  test_images = images[TRAIN_SIZE:]
  test_labels = labels[TRAIN_SIZE:]

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
      master_join4 = pd.read_csv(f, delimiter=",", header=1, usecols=["imageSOP_UID", "roi_Id", "readingSession_Id", "malignancy"])
      master_join4 = master_join4.fillna(-1)

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

        try:
          # 2) query DICOM extract file for the imageSOP_UID.
          dicom_data = dicom_files["imageSOP_UID-filePath-dicominfo-"+XmlStudyNode+".csv"]
          # finding first match on criteria: http://stackoverflow.com/questions/9868653/find-first-list-item-that-matches-criteria
          idx_row_containing_imagesSOP_UID = next( i for i in range(0,len(dicom_data)) if orig_dicom_filename in dicom_data["DICOM_original_fullPath"][i] and (dicom_data["XmlStudyNode"][i] == XmlStudyNode) and (dicom_data["StudyInstanceUID"][i] == StudyInstanceUID) and (dicom_data["SeriesInstanceUID"][i] == SeriesInstanceUID) )
          imageSOP_UID = dicom_data["imageSOP_UID"][idx_row_containing_imagesSOP_UID]

          # 3) query the semantic ratings file (master_join4.csv) for the malignancy.
          # esprint('filePath: ' + filePath + ', imageSOP_UID: ' + imageSOP_UID)
          idx_row_containing_malignancy = next( i for i in range(0,len(master_join4)) if (imageSOP_UID == master_join4["imageSOP_UID"][i]) and (roi_id == master_join4["roi_Id"][i]) and (rs_id == master_join4["readingSession_Id"][i]) )
          malignancy = master_join4["malignancy"][ idx_row_containing_malignancy ]
        except StopIteration, e:
          esprint("** StopIteration **: " + str(e) + " -- malignancy: " + str(malignancy) + " -- imageSOP_UID: " + imageSOP_UID + " -- filePath: " + filePath)
          continue
        except Exception, e:
          raise e
        else:
          # Append results in a consistent order.
          lock.acquire()
          # esprint("Appending: malignancy: " + str(malignancy) + " -- imageSOP_UID: " + imageSOP_UID + " -- filePath: " + filePath)
          # Append the image pixels and malignancy value to their respective arrays.
          # image_data.append( [ (numpy.transpose(numpy.reshape(im, -1))).tolist() ] )
          image_data.append( (numpy.reshape(im, -1)).tolist() )

          # TensorFlow (and more canonical) approach: binary classification for each possible state...
          # malignancy_rating = [0,0,0,0,0]
          # if malignancy in range(1,6):
          #   malignancy_rating[int(malignancy)-1] = 1
          # malignancy_series.append(malignancy_rating)
          malignancy_series.append(malignancy)
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
    # esprint('np_malignancy_series: ' + str(np_malignancy_series))
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