## Tensorflow and keras tutorial

We will use `tf.keras`, a way to more easily and efficiently use the tensorflow. 

This will show you how to make model design and tensorflow easier than ever before.

You will start in the following way.

```python
import tf.keras

```

Basically, you can use below 2 methods for tensorflow model construction.

```python
inp = tf.keras.layers.Input(shape=(32, 32, 3))
layer1 = tf.keras.layers.Dense(64, activation=tf.nn.relu)(inp)

model = tf.keras.Model(inputs=[inp], outputs=[layer1])

```

or

```python
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu,
                                input_shape=(32, 32, 3))

```

Please refer to the examples in this repository for more details.


## tf.probability

We will later learn how to easily design a probabilistic deep learning model by using the `tf.probability` tensorflow library with existing methods.


## Dependencies

* Ubuntu 16.04
* python >= 3.5 
* tensorflow >= 1.11.0


## TODO list

- [x] Basic deep learning training
- [ ] Advanced deep learning training
- [ ] Example of tf.probability
- [ ] Example of latest topic
- [ ] Example of probabilistic metrics

