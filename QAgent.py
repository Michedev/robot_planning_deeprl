from math import ceil

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import tensorflow as tf
import numpy as np
from random import random, randint
from brain import *
from abc import ABC, abstractmethod
from path import Path
from numba import jit, jitclass, njit
BRAINFOLDER = Path(__file__).parent / 'brain'

class QAgent:

    def __init__(self, grid_shape, discount_factor=0.85, experience_size=1024):
        self.discount_factor = discount_factor
        self.grid_shape = list(grid_shape)
        if BRAINFOLDER.exists() and BRAINFOLDER.isdir():
            self.brain = tf.keras.models.load_model(BRAINFOLDER)
        else:
            self.brain = brain_v1(self.grid_shape)  # up / down / right / left
        self._q_value_hat = 0
        self.opt = tf.optimizers.SGD(10e-4, 0.7)
        self.episode = 0
        self.writer = tf.summary.create_file_writer('robot_logs')
        self.writer.set_as_default()

        self.__i_experience = 0
        self._curiosity_values = None
        self.experience_size = experience_size

        self.experience_buffer = [np.zeros([self.experience_size] + self.grid_shape, dtype='bool'), #s_t
                                  np.zeros([self.experience_size], dtype='int8'), #action
                                  np.zeros([self.experience_size], dtype='float32'), #reward
                                  np.zeros([self.experience_size] + self.grid_shape, dtype='bool')] #s_t1


    def brain_section(self, section):
        return self.brain.get_layer(section).trainable_variables

    def decide(self, grid):
        if self.__i_experience == self.experience_size:
            print('learning from the past errors...')
            self.experience_update(self.experience_buffer, self.discount_factor)
            self.__i_experience = 0
        self.experience_buffer[0][self.__i_experience] = grid
        grid = tf.Variable(grid.astype('float32'), trainable=False)
        grid = tf.expand_dims(grid, axis=0)
        q_values = self.brain(grid)
        q_values = tf.squeeze(q_values)
        epsilon = np.e ** (-0.001 * self.episode)
        if random() > epsilon:
            i = int(tf.argmax(q_values))
        else:
            i = randint(0, 3)
        self._q_value_hat = q_values[i]
        tf.summary.scalar('q value up', q_values[0], self.episode)
        tf.summary.scalar('q value down', q_values[1], self.episode)
        tf.summary.scalar('q value left', q_values[2], self.episode)
        tf.summary.scalar('q value right', q_values[3], self.episode)
        tf.summary.scalar('expected reward', self._q_value_hat, self.episode)
        tf.summary.scalar('action took', i, self.episode)
        self.experience_buffer[1][self.__i_experience] = i

        return i

    def get_reward(self, grid, reward, player_position):
        self.experience_buffer[2][self.__i_experience] = reward
        self.experience_buffer[3][self.__i_experience] = grid

        tf.summary.scalar('true reward', reward, self.episode)

        self.episode += 1
        self.__i_experience += 1

    def experience_update(self, data, discount_factor):
        for e in range(10):
            index = np.arange(0, self.experience_size)
            np.random.shuffle(index)
            batch_size = 64
            nbatch = ceil(len(index) / batch_size)
            for i_batch in range(nbatch):
                is_last = i_batch == nbatch - 1
                slice_batch = slice(i_batch * batch_size, (i_batch+1) * batch_size if not is_last else None)
                (s_t, a_t, r_t, s_t1) = [data[i][index[slice_batch]] for i in range(len(data))]
                a_t = tf.cast(a_t, tf.int32)
                s_t = tf.cast(s_t, tf.float32)
                s_t1 = tf.cast(s_t1, tf.float32)

                with tf.GradientTape() as gt:
                    exp_rew_t = self.brain(s_t)
                    exp_rew_t = exp_rew_t.numpy()
                    exp_rew_t = exp_rew_t[:, a_t]
                    exp_rew_t1 = self.brain(s_t1)
                    exp_rew_t1 = tf.reduce_max(exp_rew_t1, axis=1)
                    loss = loss_v1(r_t, exp_rew_t, exp_rew_t1, discount_factor)
                    del s_t, a_t, r_t, s_t1
                    loss = tf.reduce_mean(loss)
                tf.summary.scalar('loss', loss, self.episode)
                gradient = gt.gradient(loss, self.brain.trainable_variables)
                self.opt.apply_gradients(zip(gradient, self.brain.trainable_variables))
                del gradient, exp_rew_t, exp_rew_t1

    def reset(self):
        self.__i_experience = 0


