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
        
        
class WGAN(tf.keras.Model):
    """
    I used https://github.com/timsainb/tensorflow2-generative-models/blob/master/3.0-WGAN-GP-fashion-mnist.ipynb
    
    Extends:
        tf.keras.Model
    """
    
    def __init__(self, **kwargs):
        super(WGAN, self).__init__()
        self.__dict__.update(kwargs)
        
        self.gen = tf.keras.Sequential(self.gen)
        self.disc = tf.keras.Sequential(self.disc)
        
    def generate(self, z):
        return self.gen(z)
    
    def discriminate(self, x):
        return self.disc(x)
    
    def compute_loss(self, x):
        """ passes through the network and computes loss."""
        
        z_samp = tf.random.normal([x.shape[0], 1, 1, self.n_Z])
        
        # run noise through generator
        x_gen = self.generate(z_samp)
        
        logits_x = self.discriminate(x)
        logits_x_gen = self.discriminate(x_gen)
        
        # gradient penalty
        d_regularizer = self.gradient_penalty(x, x_gen)
        
        disc_loss = (
                tf.reduce_mean(logits_x)
                - tf.reduce_mean(logits_x_gen)
                + d_regularizer * self.gradient_penalty_weight
        )
        
        gen_loss = tf.reduce_mean(logits_x_gen)
        
        return disc_loss, gen_loss
    
    def compute_gradients(self, x):
        """ passes through the network and computes loss."""
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            disc_loss, gen_loss = self.compute_loss(x)
            
        gen_gradients = gen_tape.gradient(gen_loss, self.gen.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.disc.trainable_variables)
        
        return gen_gradients, disc_gradients
    
    def apply_gradients(self, gen_gradients, disc_gradients):
        
        self.gen_optimizer.apply_gradients(
                zip(gen_gradients, self.gen.trainable_variables))
        self.disc_optimizer.apply_gradients(
                zip(disc_gradients, self.disc.trainable_variables))
        
    def gradient_penalty(self, x, x_gen):
        epsilon = tf.random.uniform([x.shape[0], 1, 1, 1], 0.0, 1.0)
        x_hat = epsilon * x + (1 - epsilon) * x_gen
        with tf.GradientTape() as t:
            t.watch(x_hat)
            d_hat = self.discriminate(x_hat)
        gradients = t.gradient(d_hat, x_hat)
        ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))
        d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)
        return d_regularizer
    
    @tf.function
    def train(self, x_train):
        gen_gradients, disc_gradients = self.compute_gradients(x_train)
        self.apply_gradients(gen_gradients, disc_gradients)
        
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

generator = [
    tf.keras.layers.Dense(units=7 * 7 * 64, activation="relu"),
    tf.keras.layers.Reshape(target_shape=(7, 7, 64)),
    tf.keras.layers.Conv2DTranspose(
        filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
    ),
    tf.keras.layers.Conv2DTranspose(
        filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
    ),
    tf.keras.layers.Conv2DTranspose(
        filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation="sigmoid"
    ),
]

discriminator = [
    tf.keras.layers.InputLayer(input_shape=input_shape),
    tf.keras.layers.Conv2D(
        filters=32, kernel_size=3, strides=(2, 2), activation="relu"
    ),
    tf.keras.layers.Conv2D(
        filters=64, kernel_size=3, strides=(2, 2), activation="relu"
    ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=1, activation="sigmoid"),
]

# Create model

# Optimizer
gen_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
disc_optimizer = tf.keras.optimizers.RMSprop(5e-4)

model = WGAN(
        gen = generator,
        disc = discriminator,
        gen_optimizer = gen_optimizer,
        disc_optimizer = disc_optimizer,
        n_Z = N_Z,
        gradient_penalty_weight = 10.0
)

# Train the model

# exampled data for plotting results
def plot_reconstruction(model, nex=8, zm=2):
    samples = model.generate(tf.random.normal(shape=(BATCH_SIZE, N_Z)))
    fig, axs = plt.subplots(ncols=nex, nrows=1, figsize=(zm * nex, zm))
    for axi in range(nex):
        axs[axi].matshow(
                    samples.numpy()[axi].squeeze(), cmap=plt.cm.Greys, vmin=0, vmax=1
                )
        axs[axi].axis('off')
    plt.show()

# a pandas dataframe to save the loss information to
losses = pd.DataFrame(columns=['disc_loss', 'gen_loss'])

n_epochs = 200
for epoch in range(n_epochs):
    # train
    for batch, train_x in tqdm(
        zip(range(N_TRAIN_BATCHES), train_dataset), total=N_TRAIN_BATCHES
    ):
        model.train(train_x)
    # test on holdout
    loss = []
    for batch, test_x in tqdm(
        zip(range(N_TEST_BATCHES), test_dataset), total=N_TEST_BATCHES
    ):
        loss.append(model.compute_loss(train_x))
    losses.loc[len(losses)] = np.mean(loss, axis=0)
    # plot results
    display.clear_output()
    print(
        "Epoch: {} | disc_loss: {} | gen_loss: {}".format(
            epoch, losses.disc_loss.values[-1], losses.gen_loss.values[-1]
        )
    )
    plot_reconstruction(model)