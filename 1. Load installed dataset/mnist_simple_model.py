__author__ = "kdhht5022@gmail.com"
# -*- coding: utf-8 -*-

import tensorflow as tf
print("Tensorflow version is {}".format(tf.__version__))  # 1.11.0

import numpy as np
np.random.seed(2 ** 10)

import matplotlib.pyplot as plt

# Load mnist dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# pre-processing
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
x_train = np.reshape(x_train, (len(x_train), 784))
x_test = np.reshape(x_test, (len(x_test), 784))
x_train /= 255.
x_test /= 255.

y_train = np.expand_dims(y_train, axis=1)
y_test = np.expand_dims(y_test, axis=1)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10).astype(np.float32)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10).astype(np.float32)

print("Shape of x_train is {}".format(np.shape(x_train)))  # (60000, 28, 28)
print("Shape of x_test is {}".format(np.shape(x_test)))  # (10000, 28, 28)

# Build a tf.keras model
inputs = tf.keras.Input(shape=(784,))

fc_1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(inputs)
fc_2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)(fc_1)
probabilities = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(fc_2)

#probabilities = tf.keras.layers.Dense(10, )(logits)

model = tf.keras.Model(inputs=[inputs], outputs=[probabilities])
#model.summary()
print("Total parameters of model is {}".format(model.count_params()))  #468984

opt = tf.train.RMSPropOptimizer(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=opt, metrics=['accuracy'])

BATCH_SIZE = 64

SHUFFLE_SIZE = 10000 

# Create the dataset
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.shuffle(SHUFFLE_SIZE)
dataset = dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.shuffle(SHUFFLE_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

print(dataset)  # <BatchDataset shapes: ((?, 784), (?, 1, 10)), types: (tf.float32, tf.float32)>

for epoch in range(10):
    for (train_images, train_labels), (test_images, test_labels) in zip(dataset, test_dataset):
        loss, acc = model.train_on_batch(train_images, train_labels)
        test_loss, test_acc = model.test_on_batch(test_images, test_labels)
    print('Epoch #{}\t Loss: #{}\tAccuracy: #{}'.format(epoch + 1, test_loss, test_acc))
  