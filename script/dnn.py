# Copyright (c) 2018 WangYongJie

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
parser.add_argument('--batch_size', default = 1000, type = int, help = 'batch size')
parser.add_argument('--num_epochs', default = 10, type = int, help = 'number of epoches')
parser.add_argument('--train_steps', default = 10000000, type = int, help = 'train_steps')
parser.add_argument('--feature_size', default = 121, type = int, help = 'number of features')
parser.add_argument('--perform_shuffle', default = True, type = int, help = 'shuffle or not before trainning')
parser.add_argument('--log_steps', default = 100, type = int, help = 'logsteps and summary steps')
parser.add_argument('--num_threads', default = 4, type = int, help = 'logsteps and summary steps')
parser.add_argument('--train_dir', default = '/data/lantian/xgboost/demo/paster_recommand/join_feature.valid.test.tfrecords', type = str, help = 'test file')
parser.add_argument('--test_dir', default = '/data/lantian/xgboost/demo/paster_recommand/join_feature.valid.test.tfrecords', type = str, help = 'test file')
parser.add_argument('--model_dir', default = '../model/dnn', type = str, help = 'model dir')
parser.add_argument('--server_model_dir', default = '../model/dnn/export', type = str, help = 'server model dir')
parser.add_argument('--result_dir', default = '../result', type = str, help = 'prediction result dir')
parser.add_argument('--task_type', default = 'train', type = str, help = 'type: train, evaluate, predict, export')

#model hyper parameters
parser.add_argument('--hidden_units', default = '66_56_56_36', type = str, help = 'hidden units number of each layer')
parser.add_argument('--drop_rates', default = '0.5_0.5_0.5_0.5', type = str, help = 'drop rates of each layer')
parser.add_argument('--drop_out', default = True, type = bool, help = 'model with drop out or not')
parser.add_argument('--batch_norm', default = True, type = bool, help = 'model with batch normalization or not')
parser.add_argument('--optimizer', default = 'Adam', type = str, help = '{Adam, Adagrad, GD, Momentum, RMSProp}')
parser.add_argument('--learning_rate', default = 0.01, type = float, help = 'learning rate for optimization')
parser.add_argument('--l2_reg', default = 0.01, type = float, help = 'l2 regularization')

#dict for parsing tf record file
tf_record_dict = {}

#read data in tf-record format
def input_rec(filenames, batch_size = 1000, num_epochs = 1, perform_shuffle = True):
    print('Parsing ', filenames)
    print('batch_size:', batch_size)
    print('epoch:', num_epochs)
    print('perform_shuffle:', perform_shuffle)
    #parse one record
    def parse_record(example_proto):
        parsed_example = tf.parse_single_example(example_proto, tf_record_dict)
        for key in tf_record_dict.keys():
            parsed_example[key] = tf.reshape(parsed_example[key], ())
        label = parsed_example.pop('label')
        return parsed_example, label
    #generate and process TFRecordDataset
    dataset = tf.data.TFRecordDataset(filenames).map(parse_record, num_parallel_calls = 4)
    if perform_shuffle == True:
        dataset = dataset.shuffle(100000)
    dataset = dataset.repeat(num_epochs).batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


#recommandation dnn model    
def dnn_rec(features, labels, mode, params):
    # create fully connected layers, perform weight decay, batch normlization, drop_out with each hidden layer
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    hidden_index = 0
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units = int(units), activation = tf.nn.relu,
                              kernel_regularizer = tf.contrib.layers.l2_regularizer(params['l2_reg']))
        if params['batch_norm']:
            net = tf.layers.batch_normalization(net, training = (mode == tf.estimator.ModeKeys.TRAIN))
        if params['drop_out']:
            net = tf.layers.dropout(net, rate = float(params['drop_rates'][hidden_index]), training = (mode == tf.estimator.ModeKeys.TRAIN))

        hidden_index = hidden_index + 1
    logits = tf.layers.dense(net, 1, activation = None)
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
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits = logits))
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
    #feature columns indicating which column to use
    feature_columns = []
    for i in range(args.feature_size):
        feature_columns.append(tf.feature_column.numeric_column(key = str(i)))
        tf_record_dict[str(i)] = tf.FixedLenFeature(shape = (), dtype = tf.float32, default_value = 0)
    tf_record_dict['label'] = tf.FixedLenFeature(shape = (), dtype = tf.float32, default_value = 0)

    hidden_units_list = args.hidden_units.split('_')
    drop_rates_list = args.drop_rates.split('_')
    config = tf.estimator.RunConfig().replace(session_config = tf.ConfigProto(device_count = {'GPU':0, 'CPU':args.num_threads}),
            log_step_count_steps = args.log_steps, save_summary_steps = args.log_steps)
    
    # build dnn recommanding estimator
    predictor = tf.estimator.Estimator(
        model_fn = dnn_rec,
        model_dir = args.model_dir,
        params = {
            'feature_columns': feature_columns,
            'hidden_units': hidden_units_list,
            'drop_out': args.drop_out,
            'drop_rates': drop_rates_list,
            'optimizer': args.optimizer,
            'learning_rate': args.learning_rate,
            'l2_reg': args.l2_reg,
            'batch_norm': args.batch_norm
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
        for i in range(args.feature_size):
            feature_spec[str(i)] = tf.placeholder(dtype = tf.float32, shape = [None,], name = 'feature' + str(i))
        print(feature_spec)
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        predictor.export_savedmodel(args.server_model_dir, serving_input_receiver_fn, as_text = False)
      

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
