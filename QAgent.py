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

    def __init__(self, grid, discount_factor=0.85, experience_size=8):
        self.epsilon = 1.0
        self.discount_factor = discount_factor
        input_shape = list(grid.shape) + [1]
        self.brain = brain_v1(input_shape)  # up / down / right / left
        self._q_value_hat = 0
        self._gradients_1: list = []
        self._gradients_2: list = []
        self.opt = tf.optimizers.SGD(10e-4, 0.7)
        self.episode = 1
        self.writer = tf.summary.create_file_writer('robot_logs')
        self.writer.set_as_default()
        # self.experience_buffer = np.zeros((experience_size, 100 * 200 + 1 + 1 + 2 + 100 * 200), dtype='float32')
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
        del gt

        q_values = tf.squeeze(q_values)
        epsilon = np.e ** (-0.01 * self.episode)
        if random() > epsilon:
            i = int(tf.argmax(q_values))
        else:
            i = randint(0, 3)
        self._q_value_hat = q_values[i]
        # self._curiosity_values = predicted_values
        self.experience_buffer[self.__i_experience, :100*200] = grid.numpy().reshape((-1,))
        self.experience_buffer[self.__i_experience, 100*200] = i

        return i

    def get_reward(self, grid, reward, player_position):
        self.experience_buffer[self.__i_experience, 100*200+1] = reward
        # self.experience_buffer[self.__i_experience, 100*200+2] = player_position.x
        # self.experience_buffer[self.__i_experience, 100*200+3] = player_position.y
        # self.experience_buffer[self.__i_experience, 100*200+4:] = grid.reshape((-1,))
        self.experience_buffer[self.__i_experience, 100*200+2:] = grid.reshape((-1,))

        # grid = tf.Variable(grid.astype('float32'), trainable=False)
        # grid = tf.expand_dims(grid, axis=0)
        # grid = tf.expand_dims(grid, axis=-1)
        # q_values_t_1, _ = self.brain(grid)
        # q_values_t_1 = tf.squeeze(q_values_t_1)
        # q_max_t_1 = tf.reduce_max(q_values_t_1)
        # with tf.GradientTape(persistent=True) as gt:
        #     gt.watch(self._q_value_hat)
        #     gt.watch(self._curiosity_values)
        #     loss = loss_v2(reward, self._q_value_hat, q_max_t_1, self.discount_factor,
        #                    grid, player_position, self._curiosity_values)
        # curiosity_gradient = gt.gradient(loss, self._curiosity_values)
        # q_loss_gradient = gt.gradient(loss, self._q_value_hat)
        #
        # for l in self._gradients_1:
        #     l *= q_loss_gradient
        # self._gradients_2[-1] *= tf.reshape(curiosity_gradient, (40,))
        #
        # self.opt.apply_gradients(zip(self._gradients_1,
        #                              self.brain_section('main_cortex') + self.brain_section('q_values_module')))
        # self.opt.apply_gradients(zip(self._gradients_2,
        #                              self.brain_section('main_cortex') + self.brain_section('curiosity_module')))
        #
        # tf.summary.scalar('loss', loss, step=self.episode)
        # tf.summary.scalar('expected reward', self._q_value_hat, self.episode)
        # tf.summary.scalar('true reward', reward, self.episode)

        self.episode += 1
        self.__i_experience += 1

    def experience_update(self, batch, discount_factor):
        batch = tf.squeeze(batch)
        (s_t, a_t, r_t, s_t1) = batch[:, :100*200], batch[:, 100*200], batch[:, 100*200+1], batch[:, 100*200+2:]

        # (s_t, a_t, r_t, x_player, y_player, s_t1) = batch[:, :100*200], batch[:, 100*200], batch[:, 100*200+1], batch[:, 100*200+2], batch[:, 100*200+3], batch[:, 100*200+4:]
        a_t = tf.cast(a_t, tf.int32)
        print(batch)
        s_t = tf.expand_dims(s_t, -1)
        s_t1 = tf.expand_dims(s_t1, -1)
        print(s_t1)
        with tf.GradientTape() as gt:
            exp_rew_t, curiosity_values = self.brain(s_t)
            exp_rew_a = tf.zeros((self.experience_size,))
            for i, a in enumerate(a_t):
                exp_rew_a[i] = exp_rew_t[i, a]
            # exp_rew_t = exp_rew_t[:, a_t]
            exp_rew_t1, _ = self.brain(s_t1)
            exp_rew_t1 = tf.reduce_max(exp_rew_t1, axis=1)
            # loss = loss_v2(r_t, exp_rew_a, exp_rew_t1, discount_factor, s_t, Point(x_player, y_player), curiosity_values)
            loss = loss_v1(r_t, exp_rew_a, exp_rew_t1, discount_factor)

            del s_t, a_t, r_t, s_t1
            loss = tf.reduce_mean(loss)
        gradient = gt.gradient(loss, self.brain.trainable_variables)
        self.opt.apply_gradients(zip(gradient, self.brain.trainable_variables))
        del gradient, exp_rew_t, exp_rew_t1, batch


