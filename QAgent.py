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

has_gpu = torch.cuda.is_available()

FOLDER = Path(__file__).parent
BRAINFILE = FOLDER / 'brain.pth'
LASTSTEP = FOLDER / 'laststep.txt'


class GPUData:

    def __init__(self, batch_size: int, grid_shape: List[int], extra_shape: int, device='cuda'):
        self.device = device
        self.s_t = torch.empty(batch_size, *grid_shape, dtype=torch.float32, device=device)
        self.extra_t = torch.empty(batch_size, extra_shape, dtype=torch.float32, device=device)
        self.a_t = torch.empty(batch_size, 4, dtype=torch.bool, device=device)
        self.s_t1 = torch.empty(batch_size, *grid_shape, dtype=torch.float32, device=device)
        self.extra_t1 = torch.empty(batch_size, extra_shape, dtype=torch.float32, device=device)
        self.r_t = torch.empty(batch_size, dtype=torch.float32, device=device)

    def load_in_gpu(self, s_t: torch.Tensor, extra_t, a_t: torch.Tensor, s_t1, extra_t1, r_t):
        self.s_t.set_(s_t.float().to(self.device))
        self.extra_t.set_(extra_t.to(self.device))
        self.a_t.set_((a_t.unsqueeze(-1) == torch.arange(4).unsqueeze(0)).to(self.device))
        self.s_t1.set_(s_t1.float().to(self.device))
        self.extra_t1.set_(extra_t1.to(self.device))
        self.r_t.set_(r_t.to(self.device))


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
        self.brain = BrainV1(self.grid_shape, [self.extra_shape]).to(self._device)  # up / down / right / left
        self.brain.to(self._device)
        # summary(self.brain, self.grid_shape, (self.extra_shape,), batch_size=-1, show_input=False)
        if BRAINFILE.exists():
            self.brain.load_state_dict(torch.load(BRAINFILE))
        self.q_future = BrainV1(self.grid_shape, [self.extra_shape]).to(self._device)
        self._q_value_hat = 0
        self.opt = torch.optim.SGD(self.brain.parameters(), lr=0.00025, momentum=0.8)
        self.lr_scheduler = torch.optim.lr_scheduler.CyclicLR(self.opt, 10e-5, 10e-3)
        self.torch_data = GPUData(32, self.grid_shape, self.extra_shape, device=self._device)
        self.step = 1
        self.episode = 0
        self.step_episode = 0
        self.writer = SummaryWriter('robot_logs')
        self.epsilon = 1.0
        self.__i_experience = 0
        self._curiosity_values = None
        self.experience_max_size = experience_size
        self.experience_size = 0
        self.destination_position = None

        self.experience_buffer = [torch.zeros(self.experience_max_size, *self.grid_shape, dtype=torch.bool, device='cpu'),  # s_t
                                  torch.zeros(self.experience_max_size, self.extra_shape, dtype=torch.float32, device='cpu'),  # extra_data_t
                                  torch.zeros(self.experience_max_size, dtype=torch.int8, device='cpu'),  # action
                                  torch.zeros(self.experience_max_size, dtype=torch.float32, device='cpu'),  # reward
                                  torch.zeros(self.experience_max_size, *self.grid_shape, dtype=torch.bool, device='cpu'),  # s_t1
                                  torch.zeros(self.experience_max_size, self.extra_shape, dtype=torch.float32, device='cpu')]  # extra_data_t

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

    def decide(self, grid, my_pos, destination_pos):
        self.brain.eval()
        grid = grid.reshape(self.grid_shape)
        self.destination_position = destination_pos
        if self.__i_experience == self.experience_max_size:
            self.__i_experience = 0
        if self.no_update_start < self.__i_experience and self.__i_experience % self.update_freq == 0:
            self.experience_update(self.experience_buffer, self.discount_factor)
        self.experience_buffer[0][self.__i_experience] = torch.from_numpy(grid)
        grid = torch.from_numpy(grid.astype('float32'))
        extradata = self.extra_features(grid, my_pos, destination_pos)
        grid = torch.unsqueeze(grid, dim=0)
        grid = grid.to(self._device)
        extradata = extradata.to(self._device)
        self.brain.eval()
        q_values = self.brain(grid, extradata)
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
        self.experience_buffer[1][self.__i_experience, :] = extradata
        self.experience_buffer[2][self.__i_experience] = i
        self.epsilon = max(0.05, self.epsilon - 0.0004)
        if self.step_episode % 1000 == 0:
            self.epsilon = 1.0
        return i

    def get_reward(self, grid, reward, player_position):
        grid = grid.reshape(self.grid_shape)
        grid = torch.from_numpy(grid)
        self.experience_buffer[3][self.__i_experience] = reward
        self.experience_buffer[4][self.__i_experience] = grid
        extra = self.extra_features(grid, player_position, self.destination_position)
        self.experience_buffer[5][self.__i_experience] = extra

        if self.step % 1000 == 0 and self.step > 0:
            self.q_future.load_state_dict(self.brain.state_dict())
            gc.collect()
        self.writer.add_scalar('true reward', reward, self.step)
        self.step += 1
        self.step_episode += 1
        self.__i_experience += 1
        if self.experience_max_size > self.experience_size:
            self.experience_size += 1

    def experience_update(self, data, discount_factor):
        self.brain.train()
        index = np.random.randint(0, self.experience_size, (self.sample_experience,))
        batch_size = 32
        nbatch = np.ceil(len(index) / batch_size)
        nbatch = int(nbatch)
        for i_batch in range(nbatch):
            is_last = i_batch == nbatch - 1
            self.opt.zero_grad()
            slice_batch = slice(i_batch * batch_size, ((i_batch + 1) * batch_size if not is_last else None))
            (s_t, extra_t, a_t, r_t, s_t1, extra_t1) = [data[i][index[slice_batch]] for i in range(len(data))]
            self.torch_data.load_in_gpu(s_t, extra_t, a_t, s_t1, extra_t1, r_t)
            del s_t, a_t, r_t, s_t1
            exp_rew_t = self.brain(self.torch_data.s_t, self.torch_data.extra_t)
            exp_rew_t = exp_rew_t[self.torch_data.a_t]
            exp_rew_t1 = self.q_future(self.torch_data.s_t1, self.torch_data.extra_t1)
            exp_rew_t1 = torch.max(exp_rew_t1, dim=1)
            if isinstance(exp_rew_t1, tuple):
                exp_rew_t1 = exp_rew_t1[0]
            qloss = torch.pow(self.torch_data.r_t + discount_factor * exp_rew_t1 - exp_rew_t, 2)
            qloss = torch.mean(qloss)
            self.opt.zero_grad()
            qloss.backward()
            if self.step % 10 == 0:
                self.writer.add_scalar('q loss', qloss, self.step)
            if self.step % 100 == 0:
                for l in self.brain.parameters(recurse=True):
                    self.writer.add_histogram(str(l), l, self.step)
            self.opt.step()
            self.lr_scheduler.step(self.step)


    def reset(self):
        self.epsilon = max(1.0 - 0.001 * self.episode, 0.1)
        self.step_episode = 0

    def on_win(self):
        self.writer.add_scalar('steps per episode', self.step_episode, self.episode)
        self.episode += 1
