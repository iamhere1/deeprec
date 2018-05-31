# Copyright (c) 2018 WangYongJie

"""wide deep model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import sys

#argument parser
parser = argparse.ArgumentParser()

#parameters for data set, logging, running
parser.add_argument('--category_feature_size', default = 2, type = int, help = 'category feature size')
parser.add_argument('--number_feature_size', default = 3, type = int, help = 'number feature size')
parser.add_argument('--batch_size', default = 1000, type = int, help = 'batch size')
parser.add_argument('--num_epochs', default = 200000, type = int, help = 'number of epoches')
parser.add_argument('--train_steps', default = 100000, type = int, help = 'train_steps')
parser.add_argument('--perform_shuffle', default = True, type = int, help = 'shuffle or not before trainning')
parser.add_argument('--log_steps', default = 100, type = int, help = 'logsteps and summary steps')
parser.add_argument('--num_threads', default = 4, type = int, help = 'number threads')
parser.add_argument('--train_dir', default = '../data/wdl/train.txt', type = str, help = 'train file')
parser.add_argument('--test_dir', default = '../data/wdl/test.txt', type = str, help = 'test file')
parser.add_argument('--model_dir', default = '../model/wdl', type = str, help = 'model dir')
parser.add_argument('--server_model_dir', default = '../model/wdl/export', type = str, help = 'server model dir')
parser.add_argument('--result_dir', default = '../result', type = str, help = 'prediction result dir')
parser.add_argument('--task_type', default = 'train', type = str, help = 'type: train, evaluate, predict, export')
parser.add_argument('--model_type', default = 'wide_deep', type = str, help = 'wide, deep, wide_deep')

#model hyper parameters
parser.add_argument('--hidden_units', default = '66_56_56_36', type = str, help = 'hidden units number of each layer')
parser.add_argument('--embedding_size', default = 36, type = int, help = 'embedding size')
parser.add_argument('--drop_rate', default = '0.5', type = float, help = 'drop out rate')
parser.add_argument('--batch_norm', default = True, type = bool, help = 'model with batch normalization or not')
parser.add_argument('--optimizer', default = 'Adam', type = str, help = '{Adam, Adagrad, GD, Momentum, RMSProp}')
parser.add_argument('--learning_rate', default = 0.01, type = float, help = 'learning rate for optimization')
parser.add_argument('--l2_reg', default = 0.01, type = float, help = 'l2 regularization')
parser.add_argument('--epochs_between_evals', default = 1, type = int, help = 'epoches between evaluation')
parser.add_argument('--class_number', default = 2, type = int, help = 'class number')

#columns name and default values
CSV_COLUMNS = []
CSV_COLUMNS_DEFAULT = []

#builde feature columns
def build_feature_columns(category_feature_size, number_feature_size):
    #column names and default values
    categorical_columns = [str(i) for i in range(category_feature_size)]
    number_columns = [str(i) for i in range(category_feature_size, category_feature_size + number_feature_size)]
    label_colunm = ['label']
    CSV_COLUMNS.extend(label_colunm)
    CSV_COLUMNS.extend(categorical_columns)
    CSV_COLUMNS.extend(number_columns)
    CSV_COLUMNS_DEFAULT.extend([[0]])
    for i in range(category_feature_size):
        CSV_COLUMNS_DEFAULT.extend([['0']])
    for i in range(category_feature_size, category_feature_size + number_feature_size):
        CSV_COLUMNS_DEFAULT.extend([[0]])
    
    #feature columns
    number_features = [tf.feature_column.numeric_column(column) for column in number_columns]
    categorical_features = [tf.feature_column.categorical_column_with_hash_bucket(key = column, hash_bucket_size = 1000, dtype=tf.string) for column in categorical_columns]
    crossed_features = []
    for i in range(len(categorical_columns) - 1):
        for j in range(i + 1, len(categorical_columns)):
          crossed_features.append(tf.feature_column.crossed_column([categorical_columns[i], categorical_columns[j]], hash_bucket_size = 1000))
    embedding_features = [tf.feature_column.embedding_column(feature, dimension = 8) for feature in categorical_features + crossed_features]
    wide_features = number_features +  categorical_features + crossed_features
    deep_features = number_features + embedding_features 
    return [wide_features, deep_features]

#build model estimater
def build_estimator(model_dir, model_type, wide_columns, deep_columns, params):
    hidden_units = params['hidden_units']
    run_config = params['run_config']
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
    
    if model_type == 'wide':
        return tf.estimator.LinearClassifier(
            model_dir = model_dir,
            feature_columns = wide_columns,
            n_classes = params['n_classes'],
            optimizer = optimizer,
            config = run_config)
    elif model_type == 'deep':
        return tf.estimator.DNNClassifier(
            model_dir = model_dir,
            feature_columns = deep_columns,
            hidden_units = hidden_units,
            n_classes = params['n_classes'],
            optimizer = optimizer,
            activation_fn = tf.nn.relu,
            dropout = params['dropout'],
            config = run_config)
    else:
        return tf.estimator.DNNLinearCombinedClassifier(
            model_dir = model_dir,
            linear_feature_columns = wide_columns,
            linear_optimizer = optimizer,
            dnn_feature_columns = deep_columns,
            dnn_optimizer = optimizer,
            dnn_hidden_units = hidden_units,
            dnn_activation_fn = tf.nn.relu,
            dnn_dropout = params['dropout'],
            n_classes = params['n_classes'],
            config = run_config)

#data generation function
def input_rec(data_file, num_epochs, perform_shuffle, batch_size):
    def parse_csv(value):
        print('Parsing', data_file)
        columns = tf.decode_csv(value, record_defaults = CSV_COLUMNS_DEFAULT)
        features = dict(zip(CSV_COLUMNS, columns))
        label = features.pop('label')
        return features, label
    dataset = tf.data.TextLineDataset(data_file).map(parse_csv, num_parallel_calls = 4)
    if perform_shuffle:
        dataset = dataset.shuffle(100000)
    dataset = dataset.repeat(num_epochs).batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    #sess = tf.Session()
    #print(sess.run([batch_features, batch_labels]))
    return batch_features, batch_labels

#main function
def main(argv):
    args = parser.parse_args(argv[1:])
    print("arguments")
    print(parser.parse_args())
    #build features
    wide_features, deep_features = build_feature_columns(args.category_feature_size, args.number_feature_size)
    #prepair parameters
    hidden_units_list = [int(unit_num) for unit_num in args.hidden_units.split('_')]
    run_config = run_config = tf.estimator.RunConfig().replace(
        session_config = tf.ConfigProto(device_count = {'GPU': 0, 'CPU':args.num_threads}),
        save_checkpoints_steps = args.log_steps,
        log_step_count_steps = args.log_steps,
        save_summary_steps = args.log_steps)
    params = {'hidden_units': hidden_units_list,
              'run_config': run_config,
              'n_classes': args.class_number,
              'dropout': args.drop_rate,
              'optimizer': args.optimizer,
              'learning_rate': args.learning_rate
    } 

    #build model estimater
    model = build_estimator(args.model_dir, args.model_type, wide_features, deep_features, params)
    #train and evaluation spec
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
                                     perform_shuffle = False), 
        steps = None, 
        start_delay_secs = 60, 
        throttle_secs = 60)
    #train
    if "train" == args.task_type:
        tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    #evaluate
    elif "evaluate" == args.task_type:
        eval_result = model.evaluate(
            input_fn = lambda:input_rec(args.test_dir,
                                        batch_size = args.batch_size,
                                        num_epochs = 1,
                                        perform_shuffle = False))
        print('-' * 60)
        for key in sorted(eval_result):
            print('%s: %s' % (key, eval_result[key]))
    #predict
    elif "predict" == args.task_type:
        preds = model.predict(
            input_fn = lambda:input_rec(args.test_dir,
                                        batch_size = args.batch_size,
                                        num_epochs = 1,
                                        perform_shuffle = False),
            predict_keys = "probabilities")
        with open(args.result_dir + "/pred.txt", "w") as fo:
            for res in preds:
                fo.write("%f\n" % (res['probabilities'][1]))
    #export model for serving
    else:
        assert "export" == args.task_type
        if "wide" == args.model_type:
            features = wide_features
        elif "deep" == args.model_type:
            features = deep_features
        else:
            assert "wide_deep" == args.model_type
            features = wide_features + deep_features
        feature_spec = tf.feature_column.make_parse_example_spec(features)
        print("-" * 10)
        print(feature_spec)
        print("-" * 10)
        serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        model.export_savedmodel(args.server_model_dir, serving_input_receiver_fn)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)



