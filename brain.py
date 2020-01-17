from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import tensorflow as tf
import numpy as np
from random import random, randint

from grid import Point


def cortex(input_size):
    inputs = Input(input_size)
    outputs = inputs
    i = 2
    while all(axis > ((2 - j) * (3 + i)) for j, axis in enumerate(outputs.shape[1:3])):
        outputs = Conv2D(32 * min(i, 4), kernel_size=min(3+i-2, 7))(outputs)
        outputs = ReLU()(outputs)
        if i % 2 == 1:
            outputs = MaxPool2D(min(i, 8))(outputs)
        i += 2
    outputs = Flatten()(outputs)
    return Model(inputs, outputs, name='main_cortex')


def q_value_module(input_shape):
    nmoves = 4
    inputs = Input(input_shape)
    outputs = inputs
    outputs = Dense(nmoves * 3)(outputs)
    outputs = BatchNormalization(trainable=False)(outputs)
    outputs = ReLU()(outputs)
    outputs = Dense(nmoves * 3)(outputs)
    outputs = BatchNormalization(trainable=False)(outputs)
    outputs = ReLU()(outputs)
    outputs = Dense(nmoves, activation='sigmoid')(outputs)
    outputs = tf.multiply(outputs, 2);
    outputs = tf.subtract(outputs, 2)  # map into [-1, 1]
    return Model(inputs, outputs, name='q_values_module')


def curiosity_model(input_shape):
    """
    This brain module predict the type of cells after the movement
    """
    inputs = Input(input_shape)
    outputs = inputs
    output_size = 5 * 8  # type cells * number of cells around
    outputs = Dense(output_size * 2)(outputs)
    outputs = BatchNormalization(trainable=False)(outputs)
    outputs = ReLU()(outputs)
    outputs = Dense(output_size * 2)(outputs)
    outputs = BatchNormalization(trainable=False)(outputs)
    outputs = ReLU()(outputs)
    outputs = Dense(output_size)(outputs)
    outputs = Reshape((8, 5))(outputs)
    outputs = tf.keras.activations.softmax(outputs, 1)
    return Model(inputs, outputs, name='curiosity_module')


def _outofbounds(state, position):
    return any(axis < 0 for axis in position) or \
           any(axis >= max_axis for axis, max_axis in zip(position, state.shape))


def curiosity_loss(state, player_position, output_model):
    ohe_matrix = np.eye(5)
    player_neightbours = get_player_neightbours(player_position, state)
    ohe_matrix = ohe_matrix[(player_neightbours * 4).astype('int')]
    loss = tf.reduce_sum(tf.losses.categorical_crossentropy(ohe_matrix, output_model), axis=-1)
    if len(loss.shape) > 0:
        loss = tf.reduce_mean(loss)
    return loss


def get_player_neightbours(player_position, state):
    player_neightbours = np.zeros((8,), dtype='float32')
    index = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i != 0 and j != 0:
                moved = player_position + Point(i, j)
                if _outofbounds(state, moved):
                    player_neightbours[index] = 0.0
                else:
                    player_neightbours[index] = state[moved].value
                index += 1
    return player_neightbours


def brain_v1(input_size):
    inputs = Input(input_size)
    main_cortex = cortex(input_size)
    outputs = inputs
    cortex_output = main_cortex(outputs)

    q_module = q_value_module(cortex_output.output_shape)
    q_value_est = q_module(cortex_output)
    qnetwork = Model(inputs, q_value_est, name='brain_v1')

    return qnetwork


def brain_v2(input_size):
    inputs = Input(input_size)
    main_cortex = cortex(input_size)
    outputs = inputs
    cortex_output = main_cortex(outputs)

    print(cortex_output.shape)

    q_module = q_value_module(cortex_output.shape[1:])
    q_value_est = q_module(cortex_output)

    curiosity = curiosity_model(cortex_output.shape[1:])
    curiosity_output = curiosity(cortex_output)

    brain = Model(inputs, [q_value_est, curiosity_output], name='brain_v2')

    brain.summary()

    return brain


def loss_v1(reward, est_reward, future_est_reward, discount_factor):
    return q_learning_loss(discount_factor, est_reward, future_est_reward, reward)


def q_learning_loss(discount_factor, est_reward, future_est_reward, reward):
    return reward + discount_factor * future_est_reward - est_reward


def loss_v2(reward, est_reward, future_est_reward, discount_factor, state, player_position, curiosity_output):
    return q_learning_loss(discount_factor, est_reward, future_est_reward, reward) + \
           curiosity_loss(state, player_position, curiosity_output)