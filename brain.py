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
        # self.high_details = Sequential(Conv2d(256, 256, kernel_size=3, bias=False), ReLU())

    def forward(self, input: torch.Tensor):
        output = torch.flatten(input, start_dim=1)
        output = self.l1(output)
        return output


class QValueModule(Module):

    def __init__(self, input_shape1: List[int], input_shape2: List[int]):
        super().__init__()
        self.l1 = Sequential(Linear(400, 256, bias=False),
                             BatchNorm1d(256),
                             ReLU()
                             )
        self.l2 = Sequential(Linear(256, 4, bias=False))

    def forward(self, state, neightbours):
        output = self.l1(state)
        return self.l2(output)

class AuxModule(Module, ABC):

    def input_size(self):
        return 400

    @abstractmethod
    def forward(self, *input: Any, **kwargs: Any):
        raise NotImplementedError


class Aux1Module(AuxModule):
    """
    This module estimate given the output of the visual cortex
    the robot position and the distance from the destination
    """

    def __init__(self):
        super().__init__()
        self.l1 = Sequential(Linear(self.input_size(), 256), BatchNorm1d(256), ReLU())
        self.l2 = Sequential(Linear(256, 4))

    def forward(self, input):
        return self.l2(self.l1(input))

class Aux2Module(AuxModule):
    """
    This module estimate given the output of the visual cortex
    the number of obstacles around the player
    """

    def __init__(self):
        super().__init__()
        self.l1 = Sequential(Linear(self.input_size(), 4), Sigmoid())

    def forward(self, input):
        return self.l1(input)

class Aux3Module(AuxModule):
    """
    This module estimate given the output of the visual cortex
    the number of empty blocks around the player
    """

    def __init__(self):
        super().__init__()
        self.l1 = Sequential(Linear(self.input_size(), 4), Sigmoid())

    def forward(self, input):
        return self.l1(input)



class BrainV1(Module):

    def __init__(self, state_size: Union[List[int], Tuple[int]], extradata_size: Union[List[int], Tuple[int]]):
        super().__init__()
        self.visual = VisualCortex(state_size)
        self.q_est = QValueModule(state_size, extradata_size)
        self.aux1 = Aux1Module()
        self.aux2 = Aux2Module()
        self.aux3 = Aux3Module()

    def forward(self, state, extra):
        vis_output = self.visual(state)
        output = self.q_est(vis_output, extra)
        pl_pos_distance_sol = self.aux1(vis_output)
        p_obs = self.aux2(vis_output)
        p_empty = self.aux3(vis_output)
        return output, pl_pos_distance_sol, p_obs, p_empty
