#!/usr/bin/env sh

TOOLS=caffe/build/tools

$TOOLS/compute_image_mean train_lmdb \
  dataset_mean.binaryproto

echo "Done."
