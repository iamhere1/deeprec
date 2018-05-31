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

#sample = "0 0:0 1:1.0 2:0 3:1.0 4:0 5:0 6:1.0 7:0 8:0 9:0 10:1.0 11:0 12:1.0 13:0 14:0 15:0 16:0 17:0 18:1.0 19:0 20:1.0 21:0 22:0 23:0 24:0.39 25:0.011 26:0.031 27:0 28:0 29:0 30:0.037 31:0 32:0 33:0 34:0.004 35:0.149 36:0.729 37:0.204 38:0.019 39:0 40:0 41:0.047 42:0.352 43:0.074 44:0 45:0.569 46:0 47:0.16 48:0 49:0.041 50:0.74 51:0 52:1.0 53:0 54:0 55:0 56:0 57:0 58:0 59:0 60:0 61:0 62:0 63:0 64:0 65:0 66:0 67:1.0 68:0 69:0 70:0 71:0 72:0 73:1.0 74:0 75:0 76:0 77:0 78:0 79:0 80:0 81:0 82:0 83:0 84:1.0 85:0 86:0 87:0 88:0 89:0 90:0 91:0 92:0 93:0 94:0 95:1.0 96:0 97:0 98:0 99:0.029 100:0.048 101:0.016 102:0.017 103:0 104:0.05 105:0.061 106:0.027 107:0 108:0.045 109:0.015 110:0.282 111:0.019 112:0.131 113:0.011 114:0.077 115:0.022 116:0.097 117:0 118:0.034 119:0.014"
sample = "1 0:0 1:1.0 2:0 3:0 4:0 5:0 6:1.0 7:0 8:0 9:0 10:1.0 11:0 12:0 13:0 14:0 15:0 16:0 17:0 18:1.0 19:0 20:0 21:0 22:0 23:0 24:0.992 25:0 26:0 27:0 28:0 29:0 30:0 31:0 32:0 33:0 34:1.0 35:0 36:0 37:0 38:0.227 39:0 40:0 41:0 42:0 43:0 44:0 45:0 46:0 47:1.0 48:0 49:0.008 50:0 51:0 52:0 53:0 54:0 55:0 56:0 57:0 58:0 59:1.0 60:0 61:0 62:1.0 63:0 64:0 65:0 66:0 67:0 68:0 69:0 70:0 71:1.0 72:0 73:0 74:0 75:0 76:0 77:0 78:0 79:0 80:0 81:1.0 82:0 83:0 84:0 85:1.0 86:0 87:0 88:0 89:0 90:0 91:0 92:0 93:0 94:1.0 95:0 96:0 97:0 98:0 99:0.056 100:0.031 101:0.107 102:0.038 103:0.0 104:0.04 105:0.156 106:0.018 107:0 108:0.057 109:0.05 110:0.073 111:0.011 112:0.075 113:0.008 114:0.017 115:0.026 116:0.025 117:0.019 118:0.105 119:0.09"

field_size = 120
index = []
value = []
str_vec = sample.split(' ')
assert len(str_vec) == field_size + 1
for i in range(1, len(str_vec)):
    temp_vec = str_vec[i].split(':')
    index.append(int(temp_vec[0]))
    value.append(float(temp_vec[1]))

request = predict_pb2.PredictRequest() 
request.model_spec.name='deep_fm'

request.inputs['feature_index'].CopyFrom(tf.make_tensor_proto(index, shape = [field_size]))
request.inputs['feature_value'].CopyFrom(tf.make_tensor_proto(value, shape = [field_size]))

#request.inputs['x'].CopyFrom(tf.contrib.util.make_tensor_proto(x_data, shape=[100, 1])) 
result = stub.Predict(request, 10.0) # 10 secs timeout
print(sample) 
print result
print result.outputs['prob'].float_val

