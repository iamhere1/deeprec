#coding=utf8
"""
preprocess data in libsvm format by multi-thread
- for continous features, padding the missed feature with value zero
- for features with index begin from none zero(most situations with one), reset feature index begin with 0
"""

import sys
import glob
import argparse
from multiprocessing import Pool as ThreadPool
    
parser = argparse.ArgumentParser()
parser.add_argument(
        "--thread_num",
        type = int,
        default = 4,
        help = "thread num"
        )
parser.add_argument(
        "--input_dir",
        type = str,
        default = "../data/raw_data/test_temp/",
        help = "input data dir"
        )
parser.add_argument(
        "--output_dir",
        type = str,
        default = "../data/raw_data/test_temp/",
        help = "output data dir"
        ) 
parser.add_argument(
        "--feature_num",
        type = int,
        default = 120,
        help = "feature number"
        ) 
parser.add_argument(
        "--begin_index",
        type = int,
        default = 1,
        help = "feature begin index"
        ) 


"""
process data function
- input_name: input feature file name provided by second element of pool.map 
"""
def format_data(input_name):
    infile = open(input_name, 'r')
    outfile = open(FLAGS.output_dir + "/"+ input_name.split('/')[-1] + '.res', 'w')
    feature_num = FLAGS.feature_num
    begin_index = FLAGS.begin_index

    for line in infile:
        line = line.strip()
        str_vec = line.split(' ')
        feature_dict = {}
        for i in range(1, len(str_vec)):
            temp_vec = str_vec[i].split(':')
            feature_index = int(temp_vec[0]) - begin_index
            if feature_index < 0:
                continue
            feature_value = round(float(temp_vec[1]), 3)
            feature_dict[feature_index] = feature_value
        for i in range(feature_num):
            if False == feature_dict.has_key(i):
                feature_dict[i] = 0 
        res_line = str_vec[0]
        for (key, value) in sorted(feature_dict.items(), key = lambda pair:pair[0]):
            res_line = res_line + " " + str(key) + ":" + str(value)
        outfile.write(res_line + "\n")
    infile.close()
    outfile.close()

if __name__ == "__main__":
    FLAGS, unparsed = parser.parse_known_args()
    file_list = glob.glob(FLAGS.input_dir+'/*')
    print('file_list size ', len(file_list))
    print('file_list', file_list)
    pool = ThreadPool(FLAGS.thread_num)
    pool.map(format_data, file_list)
    pool.close()
    pool.join()
