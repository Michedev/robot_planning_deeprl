import gc

from path import Path
# from torchsummary import summary
import numpy as np
from tensorboardX import SummaryWriter

from brain import BrainV1, QValueModule, VisualCortex
import torch
from typing import List
from grid import Direction
from random import random, randint
from modelsummary import summary
from experience_buffer import ExperienceBuffer
import json

has_gpu = torch.cuda.is_available()

FOLDER = Path(__file__).parent
BRAINFILE = FOLDER / 'brain.pth'
AGENTDATA = FOLDER / 'agent.json'


class QAgent:

    def __init__(self, grid_shape, discount_factor=0.8, experience_size=50_000, update_q_fut=1000,
                 sample_experience=128, update_freq=4, no_update_start=5_000, meta_learning=True):
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
        torch.cuda.empty_cache()
        self.discount_factor = discount_factor
        self.grid_shape = list(grid_shape)
        self.grid_shape[-1], self.grid_shape[0] = self.grid_shape[0], self.grid_shape[-1]
        self.grid_shape[0] -= 1
        self.grid_shape[0] += 2
        self.extra_shape = 2 + 2 + 4 * 4
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.brain = BrainV1(self.grid_shape, [self.extra_shape]).to(self._device)  # up / down / right / left
        self.brain.to(self._device)
        # summary(self.brain, self.grid_shape, (self.extra_shape,), batch_size=-1, show_input=False)
        if BRAINFILE.exists():
            self.brain.load_state_dict(torch.load(BRAINFILE))
        self.q_future = BrainV1(self.grid_shape, [self.extra_shape]).to(self._device)
        self._q_value_hat = 0
        self.task_opt = torch.optim.RMSprop(self.brain.parameters(), lr=0.0006, momentum=0.9)
        self.task_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.task_opt, step_size=30, gamma=0.1)

        self.global_opt = torch.optim.RMSprop(self.brain.parameters(), lr=0.0006, momentum=0.9)
        self.global_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.global_opt, step_size=30, gamma=0.1)

        self.mse = torch.nn.MSELoss(reduction='mean')

        self.step = 1
        self.episode = 0
        self.step_episode = 0
        self.writer = SummaryWriter('robot_logs')
        self.epsilon = 1.0
        self._curiosity_values = None
        self.experience_max_size = experience_size
        self.destination_position = None
        print(torch.cuda.memory_summary())
        self.meta_learning = meta_learning


        if AGENTDATA.exists():
            with open(AGENTDATA) as f:
                data = json.load(f)
            self.step = data['step']
            self.episode = data['episode']

        self.experience_buffer = ExperienceBuffer(self.grid_shape, self.extra_shape)

    def brain_section(self, section):
        raise NotImplementedError()

    def extra_features(self, grid, my_pos, destination_pos):
        result = torch.zeros(2 + 2 + 4 * 4, device='cpu')
        w, h = grid.shape[1:3]
        result[:4] = torch.FloatTensor([my_pos[0] / w, my_pos[1] / h, (destination_pos[0] - my_pos[0]) / w, (destination_pos[1] - my_pos[1]) / h])
        i = 0
        for direction in [Direction.North, Direction.South, Direction.Est, Direction.West]:
            val_direction = direction.value
            n_pos = my_pos + val_direction
            cell_type = grid[:4, n_pos.x, n_pos.y]
            result[4 + i * 4: 4 + (i + 1) * 4] = cell_type.squeeze()
            i += 1
        result = torch.unsqueeze(result, dim=0)
        return result

    def decide(self, grid_name, grid, my_pos, destination_pos):
        if not self.meta_learning:
            grid_name = 'a'
        self.brain.eval()
        grid = grid.reshape(self.grid_shape)
        self.destination_position = destination_pos
        if self.no_update_start < self.step and self.step % self.update_freq == 0:
            self.experience_update(self.discount_factor)
        self.experience_buffer.put_s_t(grid_name, torch.from_numpy(grid))
        grid = torch.from_numpy(grid.astype('float32'))
        extradata = self.extra_features(grid, my_pos, destination_pos)
        grid = torch.unsqueeze(grid, dim=0)
        grid = grid.to(self._device)
        extradata = extradata.to(self._device)
        self.brain.eval()
        q_values, _ = self.brain(grid, extradata)
        q_values = torch.squeeze(q_values)
        if random() > self.epsilon:
            i = int(torch.argmax(q_values))
        else:
            i = randint(0, 3)
        self._q_value_hat = q_values[i]
        self.writer.add_scalar('q value up', q_values[0], self.step)
        self.writer.add_scalar('q value down', q_values[1], self.step)
        self.writer.add_scalar('q value left', q_values[2], self.step)
        self.writer.add_scalar('q value right', q_values[3], self.step)
        self.writer.add_scalar('expected reward', self._q_value_hat, self.step)
        self.writer.add_scalar('action took', i, self.step)
        self.experience_buffer.put_extra_t(grid_name, extradata)
        self.experience_buffer.put_a_t(grid_name, i)
        self.epsilon = max(0.05, self.epsilon - 0.0004)
        if self.step_episode % 1000 == 0:
            self.epsilon = 1.0
        return i

    def get_reward(self, grid_name: str, grid, reward: float, player_position):
        if not self.meta_learning:
            grid_name = 'a'
        grid = grid.reshape(self.grid_shape)
        grid = torch.from_numpy(grid)
        self.experience_buffer.put_r_t(grid_name, reward)
        self.experience_buffer.put_s_t1(grid_name, grid)
        extra = self.extra_features(grid, player_position, self.destination_position)
        self.experience_buffer.put_extra_t1(grid_name, extra)

        if self.step % 1000 == 0 and self.step > 0:
            self.q_future.load_state_dict(self.brain.state_dict())
            gc.collect()
        self.writer.add_scalar('true reward', reward, self.step)
        self.step += 1
        self.step_episode += 1
        self.experience_buffer.increase_i(grid_name)

    def experience_update(self, discount_factor):
        self.brain.train()
        for task in self.experience_buffer.task_names:
            s_t, extra_t, a_t, r_t, s_t1, extra_t1 = self.experience_buffer.sample_same_task(task, 128)
            self._train_step(s_t, extra_t, a_t, r_t, s_t1, extra_t1, discount_factor, is_task=True)
        s_t, extra_t, a_t, r_t, s_t1, extra_t1 = self.experience_buffer.sample_all_tasks(16)
        qloss = self._train_step(s_t, extra_t, a_t, r_t, s_t1, extra_t1, discount_factor, is_task=False)
        if self.step % 10 == 0:
            self.writer.add_scalar('q loss', qloss, self.step)
        if self.step % 100 == 0:
            for i, l in enumerate(self.brain.parameters(recurse=True)):
                self.writer.add_histogram(f'{type(l)} {i}', l, self.step)

    def _train_step(self, s_t, extra_t, a_t, r_t, s_t1, extra_t1, discount_factor, is_task=True):
        s_t = s_t.float().to(self._device)
        extra_t = extra_t.to(self._device)
        a_t = (a_t.unsqueeze(-1) == torch.arange(4).unsqueeze(0)).to(self._device)
        r_t = r_t.to(self._device)
        s_t1 = s_t1.float().to(self._device)
        extra_t1 = extra_t1.to(self._device)
        exp_rew_t, aux_data = self.brain(s_t, extra_t)
        exp_rew_t = exp_rew_t[a_t]
        exp_rew_t1, _ = self.q_future(s_t1, extra_t1)
        exp_rew_t1 = torch.max(exp_rew_t1, dim=1)
        if isinstance(exp_rew_t1, tuple):
            exp_rew_t1 = exp_rew_t1[0]
        qloss = self.mse(r_t + discount_factor * exp_rew_t1, exp_rew_t)
        aux_loss = self.mse(aux_data[:, 0], sum(extra_t[4 + i * 4] for i in range(4))) +\
                   self.mse(aux_data[:, 1], sum(extra_t[5 + i * 4] for i in range(4))) +\
                   self.mse(aux_data[:, -2:], extra_t[:, -2:])
        qloss += aux_loss
        del s_t, extra_t, a_t, r_t, s_t1,  extra_t1, exp_rew_t, exp_rew_t1
        qloss = torch.mean(qloss)
        qloss.backward()
        opt = self.task_opt if is_task else self.global_opt
        opt.zero_grad()
        gc.collect()
        opt.step()
        return qloss

    def reset(self):
        self.epsilon = max(1.0 - 0.01 * self.episode, 0.1)
        self.step_episode = 0

    def on_win(self):
        self.writer.add_scalar('steps per episode', self.step_episode, self.episode)
        self.episode += 1
        self.task_lr_scheduler.step(self.episode)
        self.global_lr_scheduler.step(self.episode)
        with open(AGENTDATA, mode='w') as f:
            json.dump(dict(step=self.step, episode=self.episode), f)

        self.update_freq = min(self.update_freq + self.episode, 200)

