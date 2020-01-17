from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import tensorflow as tf
import numpy as np
from random import random, randint


def build_brain(input_size, nmoves):
    inputs = Input(input_size)
    outputs = inputs
    i = 2
    while all(axis > ((2 - j) * (3 + i)) for j, axis in enumerate(outputs.shape[1:3])):
        outputs = Conv2D(32 * min(i, 4), kernel_size=((3 + i)*2, 3+i))(outputs)
        outputs = ReLU()(outputs)
        if i % 2 == 1:
            outputs = MaxPool2D()(outputs)
        i += 2
        i = min(i, 6)
    outputs = Flatten()(outputs)
    outputs = Dense(nmoves * 3)(outputs)
    outputs = BatchNormalization(trainable=False)(outputs)
    outputs = ReLU()(outputs)
    outputs = Dense(nmoves, activation='sigmoid')(outputs)
    outputs = tf.multiply(outputs, 2)
    outputs = tf.subtract(outputs, 2)
    qnetwork = Model(inputs, outputs, name='Conv_q_network')
    qnetwork.summary()
    return qnetwork


class QAgent:

    def __init__(self, grid, discount_factor=0.85, epsilon=0.1, experience_size=1024):
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        input_shape = list(grid.shape) + [1]
        self.brain = build_brain(input_shape, 4)  # up / down / right / left
        self._q_value_hat = 0
        self._gradients: list = []
        self.opt = tf.optimizers.Adam(10e-4, 0.9)
        self.episode = 0
        self.writer = tf.summary.create_file_writer('robot_logs')
        self.writer.set_as_default()
        self.experience_buffer = np.zeros((experience_size, 100 * 200 + 1 + 1 + 100 * 200), dtype='float32')
        self.__i_experience = 0
        self.experience_size = experience_size

    def decide(self, grid):
        if self.__i_experience == self.experience_size:
            print('learning from the past errors...')
            self.experience_update()
            self.__i_experience = 0
        grid = tf.Variable(grid.astype('float32'), trainable=False)
        grid = tf.expand_dims(grid, axis=0)
        grid = tf.expand_dims(grid, axis=-1)
        with tf.GradientTape() as gt:
            q_values = self.brain(grid)
        self._gradients = gt.gradient(q_values, self.brain.trainable_variables)
        q_values = tf.squeeze(q_values)
        if random() > self.epsilon:
            i = int(tf.argmax(q_values))
        else:
            i = randint(0, 3)
        self._q_value_hat = q_values[i]
        self.experience_buffer[self.__i_experience, :100*200] = grid.numpy().reshape((-1,))
        self.experience_buffer[self.__i_experience, 100*200] = i

        return i

    def get_reward(self, grid, reward):
        self.experience_buffer[self.__i_experience, 100*200+1] = reward
        self.experience_buffer[self.__i_experience, 100*200+2:] = grid.reshape((-1,))
        grid = tf.Variable(grid.astype('float32'), trainable=False)
        grid = tf.expand_dims(grid, axis=0)
        grid = tf.expand_dims(grid, axis=-1)
        q_values_t_1 = self.brain(grid)
        q_values_t_1 = tf.squeeze(q_values_t_1)
        q_max_t_1 = tf.reduce_max(q_values_t_1)
        gradient_loss = -2 * (reward + self.discount_factor * q_max_t_1 - self._q_value_hat)
        loss = (reward + self.discount_factor * q_max_t_1 - self._q_value_hat) ** 2
        for gradient_layer in self._gradients:
            gradient_layer *= gradient_loss
        for i in range(len(self._gradients)):
            self._gradients[i] = tf.clip_by_value(self._gradients[i], -3, 3)
        self.opt.apply_gradients(zip(self._gradients, self.brain.trainable_variables))

        for i, l in enumerate(self.brain.trainable_variables):
            tf.summary.histogram(f'Gradient {l.name}', self._gradients[i], self.episode)
            tf.summary.histogram(f'Weight distribution {l.name}', l, self.episode)
        tf.summary.scalar('loss', loss, step=self.episode)
        tf.summary.scalar('expected reward', self._q_value_hat, self.episode)
        tf.summary.scalar('true reward', reward, self.episode)
        self.episode += 1
        self.__i_experience += 1

    def experience_update(self):
        buffer = tf.data.Dataset.from_tensor_slices(self.experience_buffer).batch(32)
        for batch in buffer:
            batch = tf.squeeze(batch)
            (s_t, a_t, r_t, s_t1) = batch[:, :100*200], batch[:, 100*200], batch[:, 100*200+1], batch[:, 100*200+2:]
            a_t = tf.cast(a_t, tf.int32)
            s_t = tf.reshape(s_t, (32, 200, 100, 1))
            s_t1 = tf.reshape(s_t1, (32, 200, 100, 1))
            with tf.GradientTape() as gt:
                exp_rew_t = self.brain(s_t).numpy()
                exp_rew_t = exp_rew_t[:, a_t]
                exp_rew_t1 = self.brain(s_t1)
                exp_rew_t1 = tf.reduce_max(exp_rew_t1, axis=1)
                loss = tf.losses.mean_squared_error(r_t + self.discount_factor * exp_rew_t1, exp_rew_t)
                del s_t, a_t, r_t, s_t1
                loss = tf.reduce_mean(loss)
            gradient = gt.gradient(loss, self.brain.trainable_variables)
            self.opt.apply_gradients(zip(gradient, self.brain.trainable_variables))
            del gradient, exp_rew_t, exp_rew_t1, batch
        del buffer


