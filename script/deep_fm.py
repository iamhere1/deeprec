# Copyright (c) 2018 WangYongJie
# data reading and referenced by "https://github.com/lambdaji/tf_repos/blob/master/deep_ctr/Model_pipeline/DeepFM.py"
"""dnn model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import sys

#argument parser
parser = argparse.ArgumentParser()

#parameters for data set, logging, running
parser.add_argument('--field_size', default = 120, type = int, help = 'field size')
parser.add_argument('--feature_size', default = 120, type = int, help = 'feature size')
parser.add_argument('--batch_size', default = 1000, type = int, help = 'batch size')
parser.add_argument('--num_epochs', default = 1, type = int, help = 'number of epoches')
parser.add_argument('--train_steps', default = 10000000, type = int, help = 'train_steps')
parser.add_argument('--perform_shuffle', default = True, type = int, help = 'shuffle or not before trainning')
parser.add_argument('--log_steps', default = 100, type = int, help = 'logsteps and summary steps')
parser.add_argument('--num_threads', default = 4, type = int, help = 'number threads')
parser.add_argument('--train_dir', default = '../data/train.libsvm', type = str, help = 'train file')
parser.add_argument('--test_dir', default = '../data/test.libsvm', type = str, help = 'test file')
parser.add_argument('--model_dir', default = '../model/deep_fm', type = str, help = 'model dir')
parser.add_argument('--server_model_dir', default = '../model/deep_fm/export', type = str, help = 'server model dir')
parser.add_argument('--result_dir', default = '../result', type = str, help = 'prediction result dir')
parser.add_argument('--task_type', default = 'train', type = str, help = 'type: train, evaluate, predict, export')

#model hyper parameters
parser.add_argument('--hidden_units', default = '66_56_56_36', type = str, help = 'hidden units number of each layer')
parser.add_argument('--embedding_size', default = 36, type = int, help = 'embedding size')
parser.add_argument('--drop_rates', default = '0.5_0.5_0.5_0.5', type = str, help = 'drop rates of each layer')
parser.add_argument('--drop_out', default = True, type = bool, help = 'model with drop out or not')
parser.add_argument('--batch_norm', default = True, type = bool, help = 'model with batch normalization or not')
parser.add_argument('--optimizer', default = 'Adam', type = str, help = '{Adam, Adagrad, GD, Momentum, RMSProp}')
parser.add_argument('--learning_rate', default = 0.01, type = float, help = 'learning rate for optimization')
parser.add_argument('--l2_reg', default = 0.01, type = float, help = 'l2 regularization')


#read data in libsvm format, each field encoded with one-hot encoding
def input_rec(filenames, batch_size = 1000, num_epochs = 1, perform_shuffle = True):
    print('Parsing ', filenames)
    print('batch_size:', batch_size)
    print('epoch:', num_epochs)
    print('perform_shuffle:', perform_shuffle)
    #parse one line
    def parse_line(line):
        columns = tf.string_split([line], ' ')
        label = tf.string_to_number(columns.values[0], out_type = tf.float32)
        feature = tf.string_split(columns.values[1:], ':')
        index_val = tf.reshape(feature.values, feature.dense_shape)
        index, val = tf.split(index_val, num_or_size_splits = 2,axis = 1)
        index = tf.string_to_number(index, out_type = tf.int32)
        val = tf.string_to_number(val, out_type = tf.float32)
        return {'feature_index':index, 'feature_value':val}, label
    #generate and process TextLineDataset
    dataset = tf.data.TextLineDataset(filenames).map(parse_line, num_parallel_calls = 4)
    if perform_shuffle == True:
        dataset = dataset.shuffle(100000)
    dataset = dataset.repeat(num_epochs).batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    #sess = tf.Session()
    #print(sess.run([batch_features, batch_labels]))
    return batch_features, batch_labels


#deep-fm model
def deep_fm_rec(features, labels, mode, params):
    ##################################################
    #weight variables for fm modle
    fm_b = tf.get_variable(name = 'fm_b', shape = [1], initializer = tf.glorot_normal_initializer())
    fm_w = tf.get_variable(name = 'fm_w', shape = [params['feature_size']], initializer = tf.glorot_normal_initializer())
    fm_v = tf.get_variable(name = 'fm_v', shape = [params['feature_size'], params['embedding_size']], initializer = tf.glorot_normal_initializer())
    ##################################################
    #feature index and vals
    feature_index = tf.reshape(features['feature_index'], shape = [-1, params['field_size']])
    feature_value = tf.reshape(features['feature_value'], shape = [-1, params['field_size']])
    #linear module of fm model
    #y_w = tf.nn.embedding_lookup_sparse(fm_w, feature_index, feature_val, combiner = 'sum')
    e_w = tf.nn.embedding_lookup(fm_w, feature_index) #N,F
    y_w = tf.reduce_sum(tf.multiply(e_w, feature_value),1) #N
 
    #cross module of fm model
    #embeddings = tf.nn.embedding_lookup_sparse(fm_v, feature_index, feature_value)
    embeddings = tf.nn.embedding_lookup(fm_v, feature_index) #N, F, k
    feature_value = tf.reshape(feature_value, shape = [-1, params['field_size'], 1])
    embeddings = tf.multiply(embeddings, feature_value) #N, F, k
    sum_square = tf.square(tf.reduce_sum(embeddings,1)) #N,F
    square_sum = tf.reduce_sum(tf.square(embeddings), 1) #N,F
    y_v = 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum), 1) #N
   
    ##################################################
    #deep part of deepfm model
    #create fully connected layers, perform weight decay, batch normlization, drop_out with each hidden layer
    net = tf.reshape(embeddings, shape = [-1, params['field_size'] * params['embedding_size']])
    hidden_index = 0
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units = int(units), activation = tf.nn.relu,
                              kernel_regularizer = tf.contrib.layers.l2_regularizer(params['l2_reg']))
        if params['batch_norm']:
            net = tf.layers.batch_normalization(net, training = (mode == tf.estimator.ModeKeys.TRAIN))
        if params['drop_out']:
            net = tf.layers.dropout(net, rate = float(params['drop_rates'][hidden_index]), training = (mode == tf.estimator.ModeKeys.TRAIN))

        hidden_index = hidden_index + 1
    y_deep = tf.layers.dense(net, 1, activation = tf.nn.relu)
    y_b = fm_b * tf.ones_like(y_deep, dtype=tf.float32)
    y_w = tf.reshape(y_w, shape = [-1, 1])
    y_v = tf.reshape(y_v, shape = [-1, 1])
    logits = y_b + y_w + y_v + y_deep
    #logits = tf.reshape(logits, shape = [-1 ,1])
    
    
    #predict and export
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
        'prob': tf.nn.sigmoid(logits),
        'logits': logits
        }
        export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}
        return tf.estimator.EstimatorSpec(mode, predictions = predictions, export_outputs = export_outputs)
    #evaluate and train
    labels = tf.reshape(labels, (-1, 1))
    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_op):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits=logits))
        auc = tf.metrics.auc(labels = labels,
                             predictions = tf.nn.sigmoid(logits),
                             name = 'auc_op')
        metrics = {'auc': auc}
        tf.summary.scalar('auc', auc[1])
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss = loss, eval_metric_ops = metrics)

        # train.
        assert mode == tf.estimator.ModeKeys.TRAIN
        if 'Adam' == params['optimizer']:
            optimizer = tf.train.AdamOptimizer(learning_rate = params['learning_rate'], 
                beta1 = 0.9, beta2 = 0.999, epsilon = 1e-08, use_locking = False, name = 'Adam') 
        elif 'RMSProp' == params['optimizer']:
            optimizer = tf.train.RMSPropOptimizer(learning_rate = params['learning_rate'], 
                decay = 0.9, momentum = 0.0, epsilon = 1e-10)
        elif 'Momentum' == params['optimizer']:
            optimizer = tf.train.MomentumOptimizer(learning_rate = params['learning_rate'],
                momentum = 0.9)
        elif 'Adagrad' == params['optimizer']:
            optimizer = tf.train.AdagradOptimizer(learning_rate = params['learning_rate'], 
                initial_accumulator_value = 0.1)
        else:
            assert 'GD' == params['optimizer']
            optimizer = tf.train.GradientDescentOptimizer(learning_rate = params['learning_rate'])
        train_op = optimizer.minimize(loss, global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss = loss, train_op = train_op)

def main(argv):
    args = parser.parse_args(argv[1:])
    print("arguments")
    print(parser.parse_args())

    hidden_units_list = args.hidden_units.split('_')
    drop_rates_list = args.drop_rates.split('_')
    config = tf.estimator.RunConfig().replace(session_config = tf.ConfigProto(device_count = {'GPU':0, 'CPU':args.num_threads}),
            log_step_count_steps = args.log_steps, save_summary_steps = args.log_steps)
    
    # build deep-fm recommanding estimator
    predictor = tf.estimator.Estimator(
        model_fn = deep_fm_rec,
        model_dir = args.model_dir,
        params = {
            'hidden_units': hidden_units_list,
            'drop_out': args.drop_out,
            'drop_rates': drop_rates_list,
            'optimizer': args.optimizer,
            'learning_rate': args.learning_rate,
            'l2_reg': args.l2_reg,
            'batch_norm': args.batch_norm,
            'field_size': args.field_size,
            'feature_size': args.feature_size,
            'embedding_size': args.embedding_size
        },
        config = config)
    train_spec = tf.estimator.TrainSpec(
        input_fn = lambda: input_rec(args.train_dir, 
                                     batch_size = args.batch_size,
                                     num_epochs = args.num_epochs,
                                     perform_shuffle = args.perform_shuffle),
        max_steps = args.train_steps
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn = lambda: input_rec(args.test_dir,
                                     batch_size = args.batch_size,
                                     num_epochs = 1,
                                     perform_shuffle = False), steps = None, start_delay_secs = 5, throttle_secs = 60)
    if "train" == args.task_type:
        tf.estimator.train_and_evaluate(predictor, train_spec, eval_spec)
    elif "evaluate" == args.task_type:
        eval_result = predictor.evaluate(
            input_fn = lambda:input_rec(args.test_dir, 
                                        batch_size = args.batch_size, 
                                        num_epochs = 1, 
                                        perform_shuffle = False))
        print('\nTest set auc: {auc:0.3f}\n'.format(**eval_result))
    elif "predict" == args.task_type:
        preds = predictor.predict(
                    input_fn = lambda:input_rec(args.test_dir,
                                                batch_size = args.batch_size,
                                                num_epochs = 1,
                                                perform_shuffle = False),
                    predict_keys = "prob")
        with open(args.result_dir + "/pred.txt", "w") as fo:
            for res in preds:
                fo.write("%f\n" % (res['prob']))
    elif "export" == args.task_type:
        feature_spec = {}
        feature_spec['feature_index'] = tf.placeholder(dtype = tf.int32, shape = [None, args.field_size], name = 'feature_index')
        feature_spec['feature_value'] = tf.placeholder(dtype = tf.float32, shape = [None, args.field_size], name = 'feature_value')
        print(feature_spec)
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        predictor.export_savedmodel(args.server_model_dir, serving_input_receiver_fn, as_text = False)
      

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
