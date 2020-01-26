from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import tensorflow as tf
import numpy as np
from random import random, randint

from grid import Point


def squeeze_excite_block(tensor, ratio=16):
    init = tensor
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = multiply([init, se])
    return x


def cortex(input_size):
    inputs = Input(input_size)
    outputs = inputs
    for i in range(2):
        outputs = Conv2D(32 ** (i + 1), kernel_size=3, strides=2)(outputs)
        outputs = ReLU()(outputs)
    outputs = Flatten()(outputs)

    return Model(inputs, outputs, name='main_cortex')


def q_value_module(input_shape):
    nmoves = 4
    inputs = Input(input_shape)
    outputs = inputs
    outputs = Dense(512)(outputs)
    outputs = BatchNormalization(trainable=False)(outputs)
    outputs = ReLU()(outputs)
    outputs = Dense(512)(outputs)
    outputs = BatchNormalization(trainable=False)(outputs)
    outputs = ReLU()(outputs)
    outputs = Dense(nmoves)(outputs)
    return Model(inputs, outputs, name='q_values_module')


def curiosity_model(input_shape):
    """
    This brain module predict the type of cells after the movement
    """
    inputs = Input(input_shape)
    outputs = inputs
    output_size = 4 * 4  # type cells * number of cells around
    outputs = Dense(output_size * 2)(outputs)
    outputs = BatchNormalization(trainable=False)(outputs)
    outputs = ReLU()(outputs)
    outputs = Dense(output_size * 2)(outputs)
    outputs = BatchNormalization(trainable=False)(outputs)
    outputs = ReLU()(outputs)
    outputs = Dense(output_size)(outputs)
    outputs = Reshape((4,4))(outputs)
    outputs = tf.keras.activations.softmax(outputs, 1)
    return Model(inputs, outputs, name='curiosity_module')


def _outofbounds(state, position):
    return any(axis < 0 for axis in position) or \
           any(axis >= max_axis for axis, max_axis in zip(position, state.shape))


def curiosity_loss(ohe_matrix, output_model):
    loss = tf.reduce_sum(tf.losses.categorical_crossentropy(ohe_matrix, output_model), axis=-1)
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
    loc_input = Input(2 + 2 + 4 * 4)  # mypos, destination pos, blocks around

    main_cortex = cortex(input_size)
    outputs = inputs
    cortex_output = main_cortex(outputs)

    q_module = q_value_module(cortex_output.shape[1:])
    q_value_est = q_module(cortex_output)
    q_value_est = Concatenate()([q_value_est, loc_input])
    q_value_est = Dense(4)(q_value_est)
    qnetwork = Model([inputs, loc_input], q_value_est, name='brain_v1')

    return qnetwork


def brain_v2(input_size):
    inputs = Input(input_size)
    loc_input = Input(2 + 2 + 4 * 4)  # mypos, destination pos, blocks around
    main_cortex = cortex(input_size)
    outputs = inputs
    cortex_output = main_cortex(outputs)

    print(cortex_output.shape)

    q_module = q_value_module(cortex_output.shape[1:])
    q_value_est = q_module(cortex_output)
    q_value_est = Concatenate()([q_value_est, loc_input])
    q_value_est = Dense(4)(q_value_est)

    curiosity = curiosity_model(cortex_output.shape[1:])
    curiosity_output = curiosity(cortex_output)

    brain = Model([inputs, loc_input], [q_value_est, curiosity_output], name='brain_v2')

    brain.summary()

    return brain


def loss_v1(reward, est_reward, future_est_reward, discount_factor):
    return q_learning_loss(discount_factor, est_reward, future_est_reward, reward)


def q_learning_loss(discount_factor, est_reward, future_est_reward, reward):
    return tf.losses.mse(reward + discount_factor * future_est_reward, est_reward)


def loss_v2(reward, est_reward, future_est_reward, discount_factor, neightbours, curiosity_output):
    return q_learning_loss(discount_factor, est_reward, future_est_reward, reward) + \
           curiosity_loss(neightbours, curiosity_output)
