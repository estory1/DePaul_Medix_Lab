#!/bin/bash

## Src: https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html
#export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/tensorflow-0.9.0rc0-py2-none-any.whl
#sudo /opt/local/bin/pip-2.7 install --upgrade $TF_BINARY_URL

export CUDA_HOME=/usr/local/cuda
export DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:$CUDA_HOME/lib"
export PATH="$CUDA_HOME/bin:$PATH"

sudo mv cudnn/include/cudnn.h /Developer/NVIDIA/CUDA-7.5/include/
sudo mv cudnn/lib/libcudnn* /Developer/NVIDIA/CUDA-7.5/lib
sudo ln -s /Developer/NVIDIA/CUDA-7.5/lib/libcudnn* /usr/local/cuda/lib/

git clone https://github.com/tensorflow/tensorflow

cd tensorflow
./configure

bazel build -c opt //tensorflow/tools/pip_package:build_pip_package

# To build with GPU support:
bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package

bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

# The name of the .whl file will depend on your platform.
sudo /opt/local/bin/pip-2.7 install /tmp/tensorflow_pkg/tensorflow-0.9.0rc0-py2-none-any.whl
