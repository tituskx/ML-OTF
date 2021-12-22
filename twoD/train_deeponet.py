import tensorflow as tf
import numpy as np
import time
import os
import sys
import glob
import resource
from tensorflow.python.client import device_lib

class branch(tf.keras.layers.Layer):
    def __init__(self, p):
        super().__init__()
        self.input_layer = tf.keras.layers.Dense(200, activation = tf.nn.relu)
        self.final_layer = tf.keras.layers.Dense(p)

    @tf.function
    def call(self, inputs):
        hid_out = self.input_layer(inputs)
        branch_out = self.final_layer(hid_out)
        return branch_out


class trunk(tf.keras.layers.Layer):
    def __init__(self, p):
        super().__init__()
        self.input_layer = tf.keras.layers.Dense(200, activation = tf.nn.relu)
        self.hidden_layer = tf.keras.layers.Dense(300, activation = tf.nn.relu)
        self.final_layer = tf.keras.layers.Dense(p, activation = tf.nn.relu)

    @tf.function
    def call(self, inputs):
        hid_out = self.input_layer(inputs)
        hid_out = self.hidden_layer(hid_out)
        trunk_out = self.final_layer(hid_out)
        return trunk_out


class deep_o_net(tf.keras.Model):
    def __init__(self, p):
        super().__init__()
        self.branch = branch(p)
        self.trunk = trunk(p)
        self.dot_product = tf.keras.layers.Dot(axes = 1)

    @tf.function
    def call(self, input_branch, input_trunk):
        branch_out = self.branch(input_branch)
        trunk_out = self.trunk(input_trunk)
        output = self.dot_product([branch_out, trunk_out])
        return output


def memory_usage_resource():
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return mem


def parse(tfrecord):
    tfrecord_format = (
        {
            "feature_branch": tf.io.FixedLenFeature([num_sensors ** 2], tf.float32),
            "feature_trunk": tf.io.FixedLenFeature([num_vars], tf.float32),
            "solution": tf.io.FixedLenFeature([], tf.float32)
        }
    )
    example = tf.io.parse_single_example(tfrecord, features=tfrecord_format)
    return [*example.values()]


def train_step(input_function, vars, sol):
    with tf.GradientTape() as tape:
        predictions = net(input_function, vars, training = True)
        loss = compute_loss(predictions, sol)

    gradients = tape.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, net.trainable_variables))

    return loss


def test_step(input_function, vars, sol):
  predictions = net(input_function, vars, training = False)
  loss = compute_loss(predictions, sol)

  return loss


@tf.function
def distributed_train_step(input_function, vars, sol):
  per_replica_losses = strategy.run(train_step, args=(input_function, vars, sol,))
  return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)


@tf.function
def distributed_test_step(input_function, vars, sol):
  per_replica_losses = strategy.run(test_step, args=(input_function, vars, sol,))
  return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)


num_sensors = 20
num_vars = 2
n_otf = 26
p_otf = 2

storage_location = ''
model_name = ''

print('Devices: ', device_lib.list_local_devices())

training_dataset_storage = ''
testing_dataset_storage = ''

training_tfrecords = glob.glob(training_dataset_storage)
testing_tfrecords = glob.glob(testing_dataset_storage)

strategy = tf.distribute.MirroredStrategy()

BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

EPOCHS = 300

data_tic = time.perf_counter()

training_dataset = tf.data.TFRecordDataset(training_tfrecords)
training_dataset = training_dataset.map(parse)

buffer_tic = time.perf_counter()

BUFFER_SIZE = sum(1 for x in training_dataset)

train_dataset = training_dataset.shuffle(buffer_size = BUFFER_SIZE)
train_dataset = train_dataset.batch(GLOBAL_BATCH_SIZE)
train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

testing_dataset = tf.data.TFRecordDataset(filenames=testing_tfrecords)
testing_dataset = testing_dataset.map(parse)

testing_dataset = testing_dataset.batch(GLOBAL_BATCH_SIZE)
test_dist_dataset = strategy.experimental_distribute_dataset(testing_dataset)

print('MAX RSS after loading and shuffling testing dataset: ', memory_usage_resource())

with strategy.scope():
    def compute_loss(predictions, solutions):
        per_example_loss = (predictions - solutions) ** 2
        avg_loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)
        return avg_loss

    net = deep_o_net(200)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)


tic = time.perf_counter()
training_losses = []
testing_losses = []

for epoch in range(EPOCHS):
    epoch_tic = time.perf_counter()

    total_train_loss = 0.0
    num_batches = 0
    for x in train_dist_dataset:
        total_train_loss += distributed_train_step(*x)
        num_batches += 1

    train_loss = total_train_loss / num_batches
    training_losses.append(train_loss)

    total_test_loss = 0.0
    num_test_batches = 0
    for x in test_dist_dataset:
        total_test_loss += distributed_test_step(*x)
        num_test_batches += 1

    test_loss = total_test_loss / num_test_batches
    testing_losses.append(test_loss)

    if ((epoch % 100 == 0) or (epoch == EPOCHS - 1)):

        print('Running epoch %d took %.2f seconds' % (epoch, time.perf_counter() - epoch_tic))

        with open('%s/losses/training_%s_epoch%d.npy' % (storage_location, model_name, epoch), 'wb') as file:
            np.save(file, training_losses)

        with open('%s/%s_epoch%d.npy' % (storage_location, model_name, epoch), 'wb') as file:
            np.save(file, training_losses)

        training_losses = []

        with open('%s/losses/testing_%s_epoch%d.npy' % (storage_location, model_name, epoch), 'wb') as file:
            np.save(file, training_losses)

        testing_losses = []


        print('Max RSS after epoch %d: %d' % (epoch, memory_usage_resource()))

net.save('%s/%s_epoch%d' % (storage_location, model_name, epoch))

print('Training took %.2f seconds' % (time.perf_counter() - tic))

print('MAX RSS of running entire script: ', memory_usage_resource())
