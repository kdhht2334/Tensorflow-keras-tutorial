__author__ = "kdhht5022@gmail.com"
# -*- coding: utf-8 -*-
import sys, subprocess
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from IPython import display
import pandas as pd

import tensorflow as tf
print(tf.__version__)


def run_subprocess_command(cmd):
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    for line in process.stdout:
        print(line.decode().strip())
        
        
#IN_COLAB = "google.colab" in sys.modules
#colab_requirements = ["pip install tf-nightly-gpu-2.0-preview==2.0.0.dev20190513"]
#if IN_COLAB:
#    for i in colab_requirements:
#        run_subprocess_command(i)
        
        
class AE(tf.keras.Model):
    """A basic autoencoder model using tf.keras 2.0 version"""
    
    def __init__(self, **kwargs):
        super(AE, self).__init__()
        self.__dict__.update(kwargs)
        
        self.enc = tf.keras.Sequential(self.enc)
        self.dec = tf.keras.Sequential(self.dec)
        
    @tf.function
    def encode(self, x):
        return self.enc(x)
    
    @tf.function
    def decode(self, x):
        return self.dec(x)
    
    @tf.function
    def compute_loss(self, x):
        z = self.encode(x)
        _x = self.decode(z)
        ae_loss = tf.reduce_mean(tf.square(x - _x))
        return ae_loss
    
    @tf.function
    def compute_gradients(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        return tape.gradient(loss, self.trainable_variables)
    
    def train(self, x_train):
        gradients = self.compute_gradients(x_train)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
# Create a fashion-MNIST dataset
TRAIN_BUF=60000
BATCH_SIZE=512
TEST_BUF=10000
input_shape = (28,28,1)
N_TRAIN_BATCHES =int(TRAIN_BUF/BATCH_SIZE)
N_TEST_BATCHES = int(TEST_BUF/BATCH_SIZE)

(x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype("float32")/255.
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype("float32")/255.

train_dataset = (
        tf.data.Dataset.from_tensor_slices(x_train)
        .shuffle(TRAIN_BUF)
        .batch(BATCH_SIZE)
        )
test_dataset = (
        tf.data.Dataset.from_tensor_slices(x_test)
        .shuffle(TEST_BUF)
        .batch(BATCH_SIZE)
        )

# Define the network architecture
N_Z = 64
encoder = [
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation="relu"),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation="relu"),       
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=N_Z)
        ]

decoder = [
        tf.keras.layers.Dense(units=7 * 7 * 64, activation="relu"),
        tf.keras.layers.Reshape(target_shape=(7, 7, 64)),
        tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"),
        tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"),
        tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation="sigmoid")
        ]

# Create model
optimizer = tf.keras.optimizers.Adam(1e-3)

model = AE(
        enc=encoder,
        dec=decoder,
        optimizer=optimizer)

# Train the model
example_data = next(iter(train_dataset))

def plot_reconstruction(model, example_data, nex=5, zm=3):

    example_data_reconstructed = model.decode(model.encode(example_data))
    fig, axs = plt.subplots(ncols=nex, nrows=2, figsize=(zm * nex, zm * 2))
    for exi in range(nex):
        axs[0, exi].matshow(
            example_data.numpy()[exi].squeeze(), cmap=plt.cm.Greys, vmin=0, vmax=1
        )
        axs[1, exi].matshow(
            example_data_reconstructed.numpy()[exi].squeeze(),
            cmap=plt.cm.Greys,
            vmin=0,
            vmax=1,
        )
    for ax in axs.flatten():
        ax.axis("off")
    plt.show()

# a pandas dataframe to save the loss information to
losses = pd.DataFrame(columns=['MSE'])

max_epochs = 50
for epoch in range(max_epochs):
    
    for batch, x_train in tqdm(
            zip(range(N_TRAIN_BATCHES), train_dataset), total=N_TRAIN_BATCHES):
        model.train(x_train)
        
    # test on holdout
    loss = []
    for batch, x_test in tqdm(
            zip(range(N_TRAIN_BATCHES), train_dataset), total=N_TRAIN_BATCHES):
        loss.append(model.compute_loss(x_train))
    losses.loc[len(losses)] = np.mean(loss, axis=0)
    display.clear_output()
    print("[INFO] Epoch: {} | MSE: {}".format(epoch, losses.MSE.values[-1]))
    plot_reconstruction(model, example_data)
    
plt.plot(losses.MSE.values)
    