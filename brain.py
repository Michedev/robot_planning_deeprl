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
        self.flatten_input_size = reduce(mul, input_size)
        self.l1 = Sequential(Linear(self.flatten_input_size, 400), BatchNorm1d(400), ReLU())

    def forward(self, input: torch.Tensor):
        output = torch.flatten(input, start_dim=1)
        output = self.l1(output)
        return output


class QValueModule(Module):

    def __init__(self, input_shape1: List[int], input_shape2: List[int]):
        super().__init__()
        self.l1 = Sequential(Linear(400, 256),
                             BatchNorm1d(256),
                             ReLU()
                             )
        self.l2 = Sequential(Linear(256, 4))

    def forward(self, state, neightbours):
        output = self.l1(state)
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
