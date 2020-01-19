from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import tensorflow as tf
import numpy as np
from random import random, randint
from brain import *
from abc import ABC, abstractmethod
from numba import jit, jitclass, njit

class ExperienceReplay(ABC):

    def experience_replay(self, buffer):
        pass

    @abstractmethod
    def state_size(self):
        pass

class QAgent:

    def __init__(self, grid_shape, discount_factor=0.85, experience_size=32):
        self.epsilon = 1.0
        self.discount_factor = discount_factor
        self.grid_shape = grid_shape
        self.brain = brain_v1(self.grid_shape)  # up / down / right / left
        self._q_value_hat = 0
        self.opt = tf.optimizers.SGD(10e-4, 0.7)
        self.episode = 1
        self.writer = tf.summary.create_file_writer('robot_logs')
        self.writer.set_as_default()
        self.experience_buffer = np.zeros((experience_size, 100 * 200 + 1 + 1 + 100 * 200), dtype='float32')

        self.__i_experience = 0
        self._curiosity_values = None
        self.experience_size = experience_size

    def brain_section(self, section):
        return self.brain.get_layer(section).trainable_variables

    def decide(self, grid):
        if self.__i_experience == self.experience_size:
            print('learning from the past errors...')
            self.experience_update(self.experience_buffer, self.discount_factor)
            self.__i_experience = 0
        grid = tf.Variable(grid.astype('float32'), trainable=False)
        grid = tf.expand_dims(grid, axis=0)
        grid = tf.expand_dims(grid, axis=-1)
        q_values = self.brain(grid)

        q_values = tf.squeeze(q_values)
        epsilon = np.e ** (-0.01 * self.episode)
        if random() > epsilon:
            i = int(tf.argmax(q_values))
        else:
            i = randint(0, 3)
        self._q_value_hat = q_values[i]
        tf.summary.scalar('expected reward', self._q_value_hat, self.episode)
        tf.summary.scalar('action took', i, self.episode)

        self.experience_buffer[self.__i_experience, :100*200] = grid.numpy().reshape((-1,))
        self.experience_buffer[self.__i_experience, 100*200] = i

        return i

    def get_reward(self, grid, reward, player_position):
        self.experience_buffer[self.__i_experience, 100*200+1] = reward
        self.experience_buffer[self.__i_experience, 100*200+2:] = grid.reshape((-1,))
        tf.summary.scalar('true reward', reward, self.episode)

        self.episode += 1
        self.__i_experience += 1

    def experience_update(self, batch, discount_factor):
        batch = tf.squeeze(batch)
        (s_t, a_t, r_t, s_t1) = batch[:, :100*200], batch[:, 100*200], batch[:, 100*200+1], batch[:, 100*200+2:]
        a_t = tf.cast(a_t, tf.int32)
        s_t = tf.reshape(s_t, [self.experience_size] + self.grid_shape)
        s_t1 = tf.reshape(s_t1, [self.experience_size] + self.grid_shape)

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
        del gradient, exp_rew_t, exp_rew_t1, batch

    def reset(self):
        self.__i_experience = 0


