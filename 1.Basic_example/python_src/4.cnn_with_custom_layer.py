__author__ = "kdhht5022@gmail.com"
# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(2 ** 10)

import matplotlib.pyplot as plt

import tensorflow as tf
print("Tensorflow version is {}".format(tf.__version__))  # 1.11.0

tf.enable_eager_execution()

# Load mnist dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Pre-processing
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
x_train /= 255.
x_test /= 255.

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10).astype(np.float32)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10).astype(np.float32)

print("Shape of x_train is {}".format(np.shape(x_train)))  # (60000, 28, 28)
print("Shape of x_test is {}".format(np.shape(x_test)))  # (10000, 28, 28)

# Show some examples
plt.imshow(x_train[0])
plt.imshow(x_test[0])

# Build custom layer
class Dense_with_noise(tf.keras.layers.Layer):
    
    def __init__(self, output_dim, activation, **kwargs):
        self.output_dim = output_dim
        self.activation = activation
        super(Dense_with_noise, self).__init__(**kwargs)
        
    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.output_dim))
        
        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      initializer='uniform',
                                      trainable=True)
        # Add noise to kernel
        self.kernel = self.kernel + 1e-5 * tf.random_normal(shape=shape)
        super(Dense_with_noise, self).build(input_shape)
        
    def call(self, inputs):
        fc = tf.matmul(inputs, self.kernel)
        output = self.activation(fc)
        return output
    
    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)
    
    def get_config(self):
        base_config = super(Dense_with_noise, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config
    
    @classmethod
    def from_config(cls, config):GcS
        return cls(**config)

# Build a tf.keras model
inputs = tf.keras.Input(shape=(32, 32, 3))

conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, 
                               kernel_initializer="he_normal",
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                               padding="same", activation=tf.nn.relu)(inputs)
pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)
conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, 
                               kernel_initializer="he_normal",
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                               padding="same", activation=tf.nn.relu)(pool1)
pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)
conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, 
                               kernel_initializer="he_normal",
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                               padding="same", activation=tf.nn.relu)(pool2)

flatten = tf.keras.layers.Flatten()(conv3)
fc1 = Dense_with_noise(128, activation=tf.nn.relu)(flatten)
fc2 = Dense_with_noise(128, activation=tf.nn.relu)(fc1)
probabilities = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(fc2)

#probabilities = tf.keras.layers.Dense(10, )(logits)

model = tf.keras.Model(inputs=[inputs], outputs=[probabilities])
#model.summary()
print("Total parameters of model is {}".format(model.count_params()))

opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)  # SGD optimizer
model.compile(loss='categorical_crossentropy',
              optimizer=opt, metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=64, epochs=50, verbose=1,
          validation_data=(x_test, y_test))

