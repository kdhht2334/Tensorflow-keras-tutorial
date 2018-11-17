__author__ = "kdhht5022@gmail.com"
# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(2 ** 10)

import matplotlib.pyplot as plt

import tensorflow as tf
print("Tensorflow version is {}".format(tf.__version__))  # 1.11.0

tf.enable_eager_execution()

# Load cifar10 dataset
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

# Build a tf.keras model

# 1) Encoder layer for feature extraction
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

# 2) Decoder layer for reconstruction
trans_conv1 = tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3,
                                    kernel_initializer="he_normal",
                                    kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                                    strides=1, padding="same",
                                    activation=tf.nn.relu)(pool2)
unpool1 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last")(trans_conv1)
trans_conv2 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3,
                                    kernel_initializer="he_normal",
                                    kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                                    strides=1, padding="same",
                                    activation=tf.nn.relu)(unpool1)
unpool2 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last")(trans_conv2)
reconstruction = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3,
                                    kernel_initializer="he_normal",
                                    kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                                    strides=2, padding="same",
                                    activation=tf.nn.sigmoid, name="output_2")(unpool1)

# 3) Fc layer for classification
flatten = tf.keras.layers.Flatten()(pool2)
fc1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(flatten)
fc2 = tf.keras.layers.Dense(64, activation=tf.nn.relu)(fc1)
probabilities = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="output_1")(fc2)

model = tf.keras.Model(inputs=[inputs], outputs=[probabilities, reconstruction])
#model.summary()
print("Total parameters of model is {}".format(model.count_params()))  #468984
print("model output is {}".format(model.outputs))

opt = tf.train.RMSPropOptimizer(learning_rate=0.0001)
model.compile(loss=['categorical_crossentropy', 'mse'],
              optimizer=opt, loss_weights=[1., 0.05],
              metrics=["accuracy", "mse"])

BATCH_SIZE = 64
SHUFFLE_SIZE = 10000 

# Create the dataset
dataset = tf.data.Dataset.from_tensor_slices((x_train, [y_train, x_train]))
dataset = dataset.shuffle(SHUFFLE_SIZE)
dataset = dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, [y_test, y_train]))
test_dataset = test_dataset.shuffle(SHUFFLE_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

def _input_fn(x, y):

    def generator():
        for images, labels in zip(x, y):
            yield images, {"output_1": labels, "output_2": images}

    dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, {"output_1": tf.float32, "output_2": tf.float32}))
    dataset = dataset.shuffle(SHUFFLE_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset

for epoch in range(10):
    total_loss, prediction_loss, recon_loss, \
    prediction_acc, _, _, recon_error = model.train_on_batch(_input_fn(x_train, y_train))
    _, test_loss, _, test_acc, _, _, _ = model.test_on_batch(_input_fn(x_test, y_test))
    print('Epoch #{}\t Loss: #{}\tAccuracy: #{}'.format(epoch + 1, test_loss, test_acc))
