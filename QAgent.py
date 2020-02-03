from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
from random import random, randint
from brain import *
from abc import ABC, abstractmethod
from path import Path
import gc
from numba import jit, jitclass, njit

from grid import Direction

FOLDER = Path(__file__).parent
BRAINFILEINDEX = FOLDER / 'brain.tf.index'
BRAINFILE = FOLDER / 'brain.tf'
LASTSTEP = FOLDER / 'laststep.txt'


class QAgent:

    def __init__(self, grid_shape, discount_factor=0.8, experience_size=1_000_000, update_q_fut=1000,
                 sample_experience=128, update_freq=4, no_update_start=500):
        '''
        :param grid_shape:
        :param discount_factor:
        :param experience_size:
        :param update_q_fut:
        :param sample_experience: sample size drawn from the buffer
        :param update_freq: number of steps for a model update
        :param no_update_start: number of initial steps which the model doesn't update
        '''
        ### TODO implement in decide update_freq and no_update_start
        self.no_update_start = no_update_start
        self.update_freq = update_freq
        self.sample_experience = sample_experience
        self.update_q_fut = update_q_fut
        self.epsilon = 1.0
        self.discount_factor = discount_factor
        self.grid_shape = list(grid_shape)
        self.brain = brain_v1(self.grid_shape)  # up / down / right / left
        if BRAINFILEINDEX.exists():
            self.brain.load_weights(BRAINFILE)
        self.q_future = tf.keras.models.clone_model(self.brain)
        self._q_value_hat = 0
        self.opt = tf.optimizers.RMSprop(0.00025,  0.95, 0.95, 0.01)
        self.step = 1
        self.episode = 0
        self.step_episode = 0
        self.brain.summary()
        self.writer = tf.summary.create_file_writer('robot_logs')
        self.writer.set_as_default()
        self.epsilon = 1.0
        self.__i_experience = 0
        self._curiosity_values = None
        self.experience_max_size = experience_size
        self.experience_size = 0
        self.same_values = tf.zeros((4,))
        self.same_counter = 0
        self.destination_position = None

        self.experience_buffer = [np.zeros([self.experience_max_size] + self.grid_shape, dtype='bool'),  # s_t
                                  np.zeros([self.experience_max_size, 2 + 2 + 4 * 4]),  # extra_data_t
                                  np.zeros([self.experience_max_size], dtype='int8'),  # action
                                  np.zeros([self.experience_max_size], dtype='float32'),  # reward
                                  np.zeros([self.experience_max_size] + self.grid_shape, dtype='bool'),  # s_t1
                                  np.zeros([self.experience_max_size, 2 + 2 + 4 * 4])]  # extra_data_t

    def brain_section(self, section):
        return self.brain.get_layer(section).trainable_variables

    def extra_features(self, grid, my_pos, destination_pos):
        result = np.zeros(2 + 2 + 4 * 4)
        w, h = grid.shape[1:3]
        result[:2] = [my_pos[0] / w, my_pos[1] / h]
        result[2:4] = [(destination_pos[0] - my_pos[0]) / w, (destination_pos[1] - my_pos[1]) / h]
        i = 0
        for direction in [Direction.North, Direction.South, Direction.Est, Direction.West]:
            val_direction = direction.value
            n_pos = my_pos + val_direction
            cell_type = grid[n_pos.x, n_pos.y, 1:]
            result[4 + i * 4: 4 + (i + 1) * 4] = cell_type
            i += 1
        result = np.expand_dims(result, axis=0)
        return result

    def decide(self, grid, my_pos, destination_pos):
        self.destination_position = destination_pos
        if self.__i_experience == self.experience_max_size:
            self.__i_experience = 0
        if self.no_update_start < self.__i_experience and self.__i_experience % self.update_freq == 0:
            self.experience_update(self.experience_buffer, self.discount_factor)
        self.experience_buffer[0][self.__i_experience] = grid
        grid = tf.Variable(grid.astype('float32'), trainable=False)
        extradata = self.extra_features(grid, my_pos, destination_pos)
        grid = tf.expand_dims(grid, axis=0)
        q_values = self.brain([grid, extradata])
        q_values = tf.squeeze(q_values)
        if random() > self.epsilon:
            i = int(tf.argmax(q_values))
        else:
            i = randint(0, 3)
        self._q_value_hat = q_values[i]
        tf.summary.scalar('q value up', q_values[0], self.step)
        tf.summary.scalar('q value down', q_values[1], self.step)
        tf.summary.scalar('q value left', q_values[2], self.step)
        tf.summary.scalar('q value right', q_values[3], self.step)
        tf.summary.scalar('expected reward', self._q_value_hat, self.step)
        tf.summary.scalar('action took', i, self.step)
        self.experience_buffer[1][self.__i_experience] = extradata
        self.experience_buffer[2][self.__i_experience] = i
        self.epsilon = max(0.05, self.epsilon - 0.0004)
        if self.step_episode % 1000 == 0:
          self.epsilon = 1.0
        if self.same_counter == 100:
            print('chosen the wrong path, going back to the random...')
            self.epsilon = 1.5
            self.same_counter = 0
        elif self.epsilon == 0.02 and np.all(tf.abs(self.same_values - q_values) < 0.00001):
            self.same_counter += 1
        else:
            self.same_counter = 0
            self.same_values = q_values
        return i

    def get_reward(self, grid, reward, player_position):
        self.experience_buffer[3][self.__i_experience] = reward
        self.experience_buffer[4][self.__i_experience] = grid
        extra = self.extra_features(grid, player_position, self.destination_position)
        self.experience_buffer[5][self.__i_experience] = extra

        if self.step % 1000 == 0 and self.step > 0:
            del self.q_future
            gc.collect()
            self.q_future = tf.keras.models.clone_model(self.brain)
        tf.summary.scalar('true reward', reward, self.step)
        self.step += 1
        self.step_episode += 1
        self.__i_experience += 1
        if self.experience_max_size > self.experience_size:
            self.experience_size += 1

    def experience_update(self, data, discount_factor):
          index = np.random.randint(0, self.experience_size, (self.sample_experience,))
          batch_size = 32
          nbatch = np.ceil(len(index) / batch_size)
          nbatch = int(nbatch)
          for i_batch in range(nbatch):
              is_last = i_batch == nbatch - 1
              slice_batch = slice(i_batch * batch_size, ((i_batch + 1) * batch_size if not is_last else None))
              (s_t, extra_t, a_t, r_t, s_t1, extra_t1) = [data[i][index[slice_batch]] for i in range(len(data))]
              a_t = tf.cast(a_t, tf.int32)
              s_t = tf.cast(s_t, tf.float32)
              s_t1 = tf.cast(s_t1, tf.float32)
              n_t1 = tf.reshape(extra_t1[:, -16:], (-1, 4,4))
              with tf.GradientTape() as gt:
                  exp_rew_t = self.brain([s_t, extra_t])
                  exp_rew_t = exp_rew_t * tf.one_hot(a_t, depth=4)
                  exp_rew_t = tf.reduce_max(exp_rew_t, axis=1)
                  exp_rew_t1 = self.q_future([s_t1, extra_t1])
                  exp_rew_t1 = tf.reduce_max(exp_rew_t1, axis=1)
                  qloss = tf.losses.mse(r_t + discount_factor * exp_rew_t1, exp_rew_t)
                  del s_t, a_t, r_t, s_t1
                  qloss = tf.reduce_mean(qloss)
              if self.step % 10 == 0:
                  tf.summary.scalar('q loss', tf.reduce_mean(qloss), self.step)
              gradient = gt.gradient(qloss, self.brain.trainable_variables)
              if self.step % 100 == 0:
                  for l, g in zip(self.brain.trainable_variables, gradient):
                      tf.summary.histogram('gradient ' + l.name, g, self.step)
                      tf.summary.histogram(l.name, l, self.step)
              self.opt.apply_gradients(zip(gradient, self.brain.trainable_variables))
              del gradient, exp_rew_t, exp_rew_t1

    def reset(self):
        self.epsilon = max(1.0 - 0.001 * self.episode, 0.1)
        self.step_episode = 0

    def on_win(self):
        tf.summary.scalar('steps per episode', self.step_episode, self.episode)
        self.episode += 1
