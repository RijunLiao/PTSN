#!/bin/bash

TOOLS=../../build/tools

export HDF5_DISABLE_VERSION_CHECK=1
export PYTHONPATH=.

#GLOG_logtostderr=1  $TOOLS/caffe train -solver lstm_solver_RGB.prototxt -weights center_loss_sample_version_2_iter_1090.caffemodel
GLOG_logtostderr=1  $TOOLS/caffe train -solver lstm_solver_RGB.prototxt -snapshot center_loss_sample_version_2_iter_88.solverstate
echo "Done."
