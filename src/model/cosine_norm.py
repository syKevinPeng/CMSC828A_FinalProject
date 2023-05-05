import tensorflow as tf
from keras.layers import Layer

class CosineLinear(Layer):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = self.add_weight(shape=(out_features, in_features), initializer='uniform', trainable=True)
        if sigma:
            self.sigma = self.add_weight(shape=(1,), initializer='ones', trainable=True)
        else:
            self.sigma = None

    def call(self, inputs):
        input_normalized = tf.nn.l2_normalize(inputs, axis=1)
        weight_normalized = tf.nn.l2_normalize(self.weight, axis=1)

        out = tf.linalg.matmul(input_normalized, weight_normalized, transpose_b=True)
        if self.sigma is not None:
            out = self.sigma * out
        return out

class SplitCosineLinear(Layer):
    def __init__(self, in_features, out_features1, out_features2, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features1 + out_features2
        self.fc1 = CosineLinear(in_features, out_features1, sigma=False)
        self.fc2 = CosineLinear(in_features, out_features2, sigma=False)
        if sigma:
            self.sigma = self.add_weight(shape=(1,), initializer='ones', trainable=True)
        else:
            self.sigma = None

    def call(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out = tf.concat([out1, out2], axis=1)
        if self.sigma is not None:
            out = self.sigma * out
        return out
