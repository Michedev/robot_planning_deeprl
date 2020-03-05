import gc

from path import Path
# from torchsummary import summary
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from brain import BrainV1, QValueModule, VisualCortex
import torch
from typing import List
from grid import Direction
from random import random, randint
#from modelsummary import summary
from experience_buffer import ExperienceBuffer
import json
from opt import RAdam
from torch.optim import SGD, rmsprop

has_gpu = torch.cuda.is_available()

FOLDER = Path(__file__).parent
BRAINFILE = FOLDER / 'brain.pth'
AGENTDATA = FOLDER / 'agent.json'


class QAgent:

    def __init__(self, grid_shape, discount_factor=0.9, experience_size=500_000, update_q_fut=1000,
                 sample_experience=64, update_freq=4, no_update_start=500, meta_learning=False):
        '''
        :param grid_shape:
        :param discount_factor:
        :param experience_size:
        :param update_q_fut:
        :param sample_experience: sample size drawn from the buffer
        :param update_freq: number of steps for a model update
        :param no_update_start: number of initial steps which the model doesn't update
        '''
        self.no_update_start = no_update_start
        self.update_freq = update_freq
        self.sample_experience = sample_experience
        self.update_q_fut = update_q_fut
        self.epsilon = 1.0
        self.discount_factor = discount_factor
        self.grid_shape = list(grid_shape)
        self.grid_shape[-1], self.grid_shape[0] = self.grid_shape[0], self.grid_shape[-1]
        self.grid_shape[0] -= 1
        self.extra_shape = 2 + 2 + 4 * 4
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.brain = BrainV1([self.extra_shape]).to(self._device)  # up / down / right / left
        self.brain.to(self._device)
        self.q_future = BrainV1([self.extra_shape]).to(self._device)
        if BRAINFILE.exists():
            brain_state = torch.load(BRAINFILE)
            self.brain.load_state_dict(brain_state)
            self.q_future.load_state_dict(brain_state)
            del brain_state
        self._q_value_hat = 0
        self.task_opt = RAdam(self.brain.parameters(), lr=0.0004)
        self.global_opt = RAdam(self.brain.parameters(), lr=0.0003)

        self.mse = torch.nn.MSELoss(reduction='mean')
        self.step = 1
        self.episode = 0
        self.decrease_epsilon = 0
        self.step_episode = 0
        self.writer = SummaryWriter('robot_logs')
        self.epsilon = 1.0
        self._curiosity_values = None
        self.experience_max_size = experience_size
        self.destination_position = None
        self.meta_learning = meta_learning
        if AGENTDATA.exists():
            with open(AGENTDATA) as f:
                data = json.load(f)
            self.step = data['step']
            self.episode = data['episode']

        self.experience_buffer = ExperienceBuffer(self.grid_shape, self.extra_shape)

    # noinspection PyArgumentList
    def extra_features(self, grid, my_pos, destination_pos):
        result = torch.zeros(2 + 2 + 4 * 4, device='cpu')
        w, h = grid.shape[1:3]
        result[:4] = torch.FloatTensor([my_pos[0] / w, my_pos[1] / h,
                                        (destination_pos[0] - my_pos[0]) / w,
                                        (destination_pos[1] - my_pos[1]) / h])
        i = 0
        for direction in [Direction.North, Direction.South, Direction.Est, Direction.West]:
            val_direction = direction.value
            n_pos = my_pos + val_direction
            if not n_pos.out_of_bound(w, h):
                cell_type = grid[:4, n_pos.x, n_pos.y]
                result[4 + i * 4: 4 + (i + 1) * 4] = cell_type.squeeze()
            i += 1
        result = torch.unsqueeze(result, dim=0)
        return result

    def decide(self, grid_name, grid, my_pos, destination_pos):
        if self.episode % 30 == 0:
          self.writer.add_image(grid_name + '_episode_' + str(self.episode), np.expand_dims(np.sum(grid * np.arange(1,5).reshape((1,1,4)), axis=-1) / 4.0, axis=0), self.step_episode)
        if not self.meta_learning:
            grid_name = 'grid'
        self.brain.eval()
        grid, extradata = self.build_data(destination_pos, grid, grid_name, my_pos)
        self.brain.eval()
        q_values = self.brain(grid, extradata).squeeze()
        if random() > self.epsilon:
            i = int(torch.argmax(q_values))
        else:
            i = randint(0, 3)
        self._q_value_hat = q_values[i]
        self._write_values_tb(i, q_values)
        self.experience_buffer.put_s_t(grid_name, grid)
        self.experience_buffer.put_extra_t(grid_name, extradata)
        self.experience_buffer.put_a_t(grid_name, i)
        self.epsilon = max(0.0, self.epsilon - 0.0001)
        return i

    def build_data(self, destination_pos, grid, grid_name, my_pos):
        grid = grid.reshape(self.grid_shape)
        self.destination_position = destination_pos
        if self.no_update_start < self.step and self.step % self.update_freq == 0:
            self.experience_update(self.discount_factor)
        grid = torch.from_numpy(grid.astype('float32'))
        extradata = self.extra_features(grid, my_pos, destination_pos)
        grid = torch.unsqueeze(grid, dim=0)
        grid = grid.to(self._device)
        extradata = extradata.to(self._device)
        return grid, extradata

    def _write_values_tb(self, action, q_values):
        self.writer.add_scalar('q_value/up', q_values[0], self.step)
        self.writer.add_scalar('q_value/down', q_values[1], self.step)
        self.writer.add_scalar('q_value/left', q_values[2], self.step)
        self.writer.add_scalar('q_value/right', q_values[3], self.step)
        self.writer.add_scalars('q_values',
                                {'up': q_values[0], 'down': q_values[1], 'left': q_values[2], 'right': q_values[3]},
                                self.step)
        self.writer.add_scalar('q_value/expected_reward', self._q_value_hat, self.step)
        self.writer.add_scalar('q_value/action', action, self.step)

    def get_reward(self, grid_name: str, grid, reward: float, player_position):
        if not self.meta_learning:
            grid_name = 'grid'
        grid = grid.reshape(self.grid_shape)
        grid = torch.from_numpy(grid)
        self._save_experience_reward(grid, grid_name, player_position, reward)
        if self.step % 1000 == 0 and self.step > 0:
            self.q_future.load_state_dict(self.brain.state_dict())
            gc.collect()
        self.writer.add_scalar('true reward', reward, self.step)
        self.step += 1
        self.step_episode += 1
        self.experience_buffer.increase_i(grid_name)

    def _save_experience_reward(self, grid, grid_name, player_position, reward):
        self.experience_buffer.put_r_t(grid_name, reward)
        self.experience_buffer.put_s_t1(grid_name, grid, decrease=0)
        extra = self.extra_features(grid, player_position, self.destination_position)
        self.experience_buffer.put_extra_t1(grid_name, extra, decrease=0)
        if self.experience_buffer.i_buffers[grid_name] >= 1:
            self.experience_buffer.put_r_t1(grid_name, reward)
            if reward != 1 and self.experience_buffer.get_r_t(grid_name, decrease=1) != 1:
                self.experience_buffer.put_s_t1(grid_name, grid, decrease=1)
                self.experience_buffer.put_extra_t1(grid_name, extra, decrease=1)
        if self.experience_buffer.i_buffers[grid_name] >= 2:
            self.experience_buffer.put_r_t2(grid_name, reward)
            if reward != 1 and \
                self.experience_buffer.get_r_t1(grid_name,decrease=1) != 1.0 and \
                self.experience_buffer.get_r_t(grid_name, decrease=2) != 1.0:
                    self.experience_buffer.put_s_t1(grid_name, grid, decrease=2)
                    self.experience_buffer.put_extra_t1(grid_name, extra, decrease=2)

    def experience_update(self, discount_factor):
        self.brain.train()
        for task in self.experience_buffer.task_names:
            s_t, extra_t, a_t, r_t, s_t1, extra_t1, r_t1, r_t2 = \
                self.experience_buffer.sample_same_task(task, self.sample_experience)
            qloss = self._train_step(s_t, extra_t, a_t, r_t, s_t1, extra_t1, r_t1, r_t2, discount_factor, is_task=True)
        if self.meta_learning:
          s_t, extra_t, a_t, r_t, s_t1, extra_t1, r_t1, r_t2 = self.experience_buffer.sample_all_tasks(16)
          qloss = self._train_step(s_t, extra_t, a_t, r_t, s_t1, extra_t1, r_t1, r_t2, discount_factor, is_task=False)

        if self.step % 10 == 0:
            self.writer.add_scalar('q loss', qloss, self.step)
        if self.step % 100 == 0:
            for lname, params in self.brain.state_dict().items():
                self.writer.add_histogram(lname.replace('.', '/'), params, global_step=self.step)

    def _train_step(self, s_t, extra_t, a_t, r_t, s_t1, extra_t1, r_t1, r_t2, discount_factor, is_task=True):
        opt = self.task_opt if is_task else self.global_opt
        opt.zero_grad()
        s_t, a_t, extra_t, extra_t1, s_t1, r_t, r_t1, r_t2 = \
            self.put_into_device(a_t, extra_t, extra_t1, r_t, r_t1, r_t2, s_t, s_t1)
        exp_rew_t, c_out = self.brain(s_t, extra_t, curiosity=True)
        exp_rew_t = exp_rew_t[a_t]
        is_finished_episode = ((torch.ne(r_t, 1.0) & torch.ne(r_t1, 1.0)) & torch.ne(r_t2, 1.0)).float().unsqueeze(0)
        exp_rew_t3 = is_finished_episode * self.q_future(s_t1, extra_t1)
        exp_rew_t3 = torch.max(exp_rew_t3, dim=1, )
        if isinstance(exp_rew_t3, tuple):
            exp_rew_t3 = exp_rew_t3[0]
        y = r_t + discount_factor * r_t1 + discount_factor ** 2 * r_t2 + discount_factor ** 3 * exp_rew_t3
        qloss = self.mse(y, exp_rew_t)
        curiosity_loss: torch.Tensor = - c_out * extra_t1[:, -16:].reshape((-1, 4, 4))
        curiosity_loss = curiosity_loss.sum(dim=[1,2]).mean(dim=0)
        qloss += curiosity_loss  #crossentropy, curiosity loss
        del s_t, extra_t, a_t, r_t, s_t1,  extra_t1, exp_rew_t, exp_rew_t3
        qloss = torch.mean(qloss)
        qloss.backward()
        gc.collect()
        opt.step()
        return qloss

    def put_into_device(self, a_t, extra_t, extra_t1, r_t, r_t1, r_t2, s_t, s_t1):
        s_t = s_t.float().to(self._device)
        extra_t = extra_t.to(self._device)
        a_t = (a_t.unsqueeze(-1) == torch.arange(4).unsqueeze(0)).to(self._device)  # ohe
        r_t = r_t.to(self._device)
        s_t1 = s_t1.float().to(self._device)
        extra_t1 = extra_t1.to(self._device)
        r_t1 = r_t1.to(self._device)
        r_t2 = r_t2.to(self._device)
        return s_t, a_t, extra_t, extra_t1, s_t1, r_t, r_t1, r_t2

    def reset(self):
        self.epsilon = max(1.0 - 0.004 * self.decrease_epsilon, 0.01)
        self.step_episode = 0

    def on_win(self):
        self.writer.add_scalar('steps per episode', self.step_episode, self.episode)
        self.episode += 1
        self.decrease_epsilon += 1
        with open(AGENTDATA, mode='w') as f:
            json.dump(dict(step=self.step, episode=self.episode), f)
        self.update_freq = min(int(self.update_freq + self.episode/10), 30)

