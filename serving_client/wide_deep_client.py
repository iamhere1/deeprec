#coding=utf8
from grpc.beta import implementations
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import prediction_service_pb2

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS
host, port = FLAGS.server.split(':')

channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

sample = "1,1,1,1.0,1.0,1.0"
#sample = "0,1,1,10,0,1"
categoriy_feature_size = 2
number_feature_size = 3
request = classification_pb2.ClassificationRequest()
feature_vec = sample.split(',')
feature_dict = {}
for i in range(1, categoriy_feature_size + number_feature_size + 1):
  if i < categoriy_feature_size + 1:
    feature_dict[str(i - 1)] = tf.train.Feature(bytes_list = tf.train.BytesList(value = [str(feature_vec[i])])) #string feature
  else:
    feature_dict[str(i - 1)] = tf.train.Feature(float_list = tf.train.FloatList(value = [float(feature_vec[i])])) #float feature

example = tf.train.Example(
     features = tf.train.Features(
         feature = feature_dict
     )
)

examples = []
examples.append(example)
request.input.example_list.examples.extend(examples)
request.model_spec.name = 'wide_deep'
result = stub.Classify(request, 10.0) # 10 secs timeout

print(sample)
print(feature_dict)
print(result)

