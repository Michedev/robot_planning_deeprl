from typing import Iterable, List, Tuple, Union, Any

import torch
import numpy as np
from random import random, randint
from grid import Point
from functools import reduce
from abc import ABC, abstractmethod, abstractproperty
from operator import mul
from torch.nn import *


class VisualCortex(Module):

    def __init__(self, input_size: Union[List[int], Tuple[int]]):
        super().__init__()
        self.l1 = Sequential(Conv2d(4, 4, kernel_size=3, bias=True),
                             ReLU())

    def forward(self, input: torch.Tensor):
        output = self.l1(input)
        output = torch.flatten(output, start_dim=1)
        return output


class QValueModule(Module):

    def __init__(self, input_shape1: List[int], input_shape2: List[int]):
        super().__init__()
        self.l1 = Sequential(Linear(10 * 10 * 4, 64, bias=True),
                             ReLU(),
                             )
        self.l2 = Sequential(Linear(64 + input_shape2[-1], 4, bias=True))
        with torch.no_grad():
            for i in range(4):
                self.l2[0].weight[:, -3 - i * 4] = -0.05
                self.l2[0].weight[:, -1 - i * 4] = 0.30
            # order: [Direction.North, Direction.South, Direction.Est, Direction.West]
            #w, h
            #
            for i in range(4):
                if i in [2, 3]:
                    self.l2[0].weight[i, -17] = 0.01  # h
                else:
                    self.l2[0].weight[i, -17] = -0.01  # h
                if i in [0, 1]:
                    self.l2[0].weight[i, -18] = 0.01  # w
                else:
                    self.l2[0].weight[i, -18] = - 0.01  # w


    def forward(self, state, neightbours):
        output = self.l1(state)
        output = torch.cat([output, neightbours], dim=-1)
        return self.l2(output)


class BrainV1(Module):

    def __init__(self, state_size: Union[List[int], Tuple[int]], extradata_size: Union[List[int], Tuple[int]]):
        super().__init__()
        self.visual = VisualCortex(state_size)
        self.q_est = QValueModule(state_size, extradata_size)

    def forward(self, state, extra):
        vis_output = self.visual(state)
        output = self.q_est(vis_output, extra)
        return output
