from typing import Tuple, Dict, Union, List
from collections import defaultdict

import torch


class ExperienceBuffer:

    def __init__(self, grid_shape: Union[Tuple, List], extra_shape: int, experience_size=100_000):
        self.grid_shape = grid_shape
        self.extra_shape = extra_shape
        self.experience_max_size = experience_size
        self.experience_buffers: Dict[str, list] = defaultdict(lambda: self._new_buffer())
        self.i_buffers: Dict[str, int] = defaultdict(lambda: 0)
        self.experience_sizes: Dict[str, int] = defaultdict(lambda: 0)

    @property
    def task_names(self):
        return list(self.experience_buffers.keys())

    @property
    def num_tasks(self):
        return len(self.experience_buffers.keys())

    def _new_buffer(self, num_rows=None):
        num_rows = num_rows or self.experience_max_size
        return [
            torch.zeros(num_rows, *self.grid_shape, dtype=torch.float32, device='cpu'),  # s_t
            torch.zeros(num_rows, self.extra_shape, dtype=torch.float32, device='cpu'),  # extra_data_t
            torch.zeros(num_rows, dtype=torch.int8, device='cpu'),  # action
            torch.zeros(num_rows, dtype=torch.float32, device='cpu'),  # reward_t
            torch.zeros(num_rows, *self.grid_shape, dtype=torch.float32, device='cpu'),  # s_t1
            torch.zeros(num_rows, self.extra_shape, dtype=torch.float32, device='cpu'), # extra_data_t1
            torch.zeros(num_rows, dtype=torch.float32, device='cpu'),  # reward_t1
            torch.zeros(num_rows, dtype=torch.float32, device='cpu')  # reward_t2
        ]

    def put_s_t(self, name: str, value):
        self.experience_buffers[name][0][self.i_buffers[name]] = value

    def put_extra_t(self, name: str, value):
        self.experience_buffers[name][1][self.i_buffers[name]] = value

    def put_a_t(self, name: str, value):
        self.experience_buffers[name][2][self.i_buffers[name]] = value

    def put_r_t(self, name: str, value):
        self.experience_buffers[name][3][self.i_buffers[name]] = value

    def get_r_t(self, name: str, decrease=0):
        return self.experience_buffers[name][3][self.i_buffers[name]-decrease]

    def put_s_t1(self, name: str, value, decrease=0):
        self.experience_buffers[name][4][self.i_buffers[name]-decrease] = value

    def put_extra_t1(self, name: str, value, decrease=0):
        self.experience_buffers[name][5][self.i_buffers[name]-decrease] = value

    def put_r_t1(self, name: str, value):
        self.experience_buffers[name][6][self.i_buffers[name]-1] = value

    def get_r_t1(self, name: str, decrease=0):
        return self.experience_buffers[name][6][self.i_buffers[name]-decrease]

    def put_r_t2(self, name: str, value):
        self.experience_buffers[name][7][self.i_buffers[name]-2] = value

    def get_r_t2(self, name: str, decrease=0):
        return self.experience_buffers[name][7][self.i_buffers[name]-decrease]

    def increase_i(self, name: str):
        self.i_buffers[name] += 1
        self.experience_sizes[name] = max(self.experience_sizes[name], self.i_buffers[name])
        if self.i_buffers[name] == self.experience_max_size:
            self.i_buffers[name] = 0

    def sample_same_task(self, name: str, batch_size=128):
        i_batch = torch.randint(0, self.experience_sizes[name], [batch_size])
        return [self.experience_buffers[name][i][i_batch] for i in range(8)]

    def sample_all_tasks(self, per_batch_size=16):
        tasks = self.task_names
        result = self._new_buffer(per_batch_size * len(tasks))
        for i_task, t in enumerate(tasks):
            task_batch = self.sample_same_task(t, per_batch_size)
            for j in range(6):
                result[j][i_task * per_batch_size: (i_task + 1) * per_batch_size] = task_batch[j]
        return result
