#!/usr/bin/env python

import tensorflow as tf
import os
import sys

def generate_tfrecords(input_filename, output_filename):
  print("Start to convert {} to {}".format(input_filename, output_filename))
  writer = tf.python_io.TFRecordWriter(output_filename)

  for line in open(input_filename, "r"):
    data = line.split(" ")
    label = float(data[0])
    features={}
    features["label"] = tf.train.Feature(float_list=tf.train.FloatList(value= [label]))
    for fea in data[1:]:
      index, val = fea.split(":")
      features[index] = tf.train.Feature(float_list=tf.train.FloatList(value=[float(val)]))
    
    # Write each example one by one
    example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(example.SerializeToString())

  writer.close()
  print("Successfully convert {} to {}".format(input_filename,
                                               output_filename))


def main():
  if len(sys.argv) < 2:
    print ('Usage: file_name')
    exit(0)

  source_file = sys.argv[1]
  dst_file = source_file + ".tfrecords"
  generate_tfrecords(source_file, dst_file)


if __name__ == "__main__":
  main()

