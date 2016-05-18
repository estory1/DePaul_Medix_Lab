# A partial (matrix only; no vectors) translation of the mat2gray function from MATLAB to Python.
#
# Author: Evan Story (estory0@gmail.com)
# Date created: 20160306
#
# MATLAB src: http://lasp.colorado.edu/cism/CISM_DX/code/CISM_DX-0.50/required_packages/octave-forge/main/image/mat2gray.m
#
# Mmin = min (min (M));
# Mmax = max (max (M));
#
#  I = (M < Mmin) .* 0;
#  I = I + (M >= Mmin & M < Mmax) .* (1 / (Mmax - Mmin) * (M - Mmin));
#  I = I + (M >= Mmax);

import numpy as np

def mat2gray(m):
  m = (np.asmatrix(m))
  Mmin = np.min(m)
  Mmax = np.max(m)
  I = np.zeros(np.shape(m))
  divisorMat = float(Mmax - Mmin) * (m - Mmin)
  if (np.max(divisorMat) > 0):
    I = np.add(I, np.multiply( np.logical_and( np.greater_equal(m, Mmin), np.less(m, Mmax)), (1 / float(Mmax - Mmin) * (m - Mmin)) ) )
  I = np.add(I, (np.greater_equal(m, Mmax)))

  return I