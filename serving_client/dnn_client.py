#coding=utf8
from grpc.beta import implementations
import numpy as np
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS

host, port = FLAGS.server.split(':')
channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)


sample = "1 2:1 4:1 7:1 19:1 21:1 25:0.258 26:0.049 27:0.014 28:0.043 29:0.313 30:0.373 31:0.263 32:0.089 34:0.154 35:0.204 36:0.039 37:0.308 38:0.171 39:0.028 41:0.256 42:0.053 43:0.077 44:0.022 46:0.208 47:0.081 48:0.037 49:0.425 50:0.108 52:0.340 53:1 68:1 77:1 82:1 93:1 97:1 100:0.054 101:0.069 102:0.042 103:0.089 105:0.040 106:0.044 107:0.039 109:0.045 110:0.026 111:0.103 112:0.055 113:0.037 114:0.038 115:0.063 116:0.093 117:0.057 118:0.008 119:0.048 120:0.049"
#sample = "0 2:1 4:1 7:1 11:1 12:1 13:1 15:1 19:1 22:1 25:0.109 26:0.050 27:0.020 29:0.456 31:0.113 35:0.076 36:0.038 37:0.653 38:0.305 39:0.066 41:0.322 42:0.025 43:0.031 44:0.005 45:0.039 46:0.088 47:0.072 48:0.063 49:0.768 50:0.036 64:1 77:1 82:1 93:1 97:1 100:0.054 101:0.069 102:0.042 103:0.089 105:0.040 106:0.044 107:0.039 109:0.045 110:0.026 111:0.103 112:0.055 113:0.037 114:0.038 115:0.063 116:0.093 117:0.057 118:0.008 119:0.048 120:0.049"


data = []
for i in range(121):
    data.append(0.0)
str_vec = sample.split(' ')
for i in range(1, len(str_vec)):
    temp_vec = str_vec[i].split(':')
    data[int(temp_vec[0])] = float(temp_vec[1])

request = predict_pb2.PredictRequest() 
request.model_spec.name = 'dnn'

feature_num = 121
for i in range(feature_num):
    request.inputs[str(i)].CopyFrom(tf.contrib.util.make_tensor_proto(data[i]))

#request.inputs['x'].CopyFrom(tf.contrib.util.make_tensor_proto(x_data, shape=[100, 1])) 
result = stub.Predict(request, 10.0) # 10 secs timeout
print data
print result
print result.outputs['prob'].float_val

