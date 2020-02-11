from typing import Iterable, List, Tuple, Union

import torch
import numpy as np
from random import random, randint
from grid import Point
from torch.nn import *


class VisualCortex(Module):

    def __init__(self, input_size: Union[List[int], Tuple[int]]):
        super().__init__()
        self.input_size = input_size
        self.c1 = Conv2d(input_size[0], 256, kernel_size=3, stride=3, bias=False)
        self.g1 = GroupNorm(4, 256)
        self.low_details = Sequential(self.c1, self.g1, ReLU())

        self.high_features = Sequential(
            Conv2d(256, 512, kernel_size=3, stride=3, bias=False),
            GroupNorm(4, 512),
            ReLU())

        # self.high_details = Sequential(Conv2d(256, 256, kernel_size=3, bias=False), ReLU())

    def forward(self, input: torch.Tensor):
        output1 = self.low_details(input)
        output2 = self.high_features(output1)
        output = torch.flatten(output2, start_dim=1)

        return output


class QValueModule(Module):

    def __init__(self, input_shape1: List[int], input_shape2: List[int]):
        super().__init__()
        self.l1 = Sequential(Linear(512, 256, bias=False),
                             BatchNorm1d(256),
                             ReLU()
                             )
        self.l2 = Sequential(Linear(256, 16, bias=False),
                             BatchNorm1d(16),
                             ReLU()
                             )
        self.l3 = Sequential(Linear(16, 4, bias=False))

    def forward(self, state, neightbours):
        output = self.l1(state)
        output = self.l2(output)
        return self.l3(output)


class AuxModule(Module):
    """
    This module estimate given the output of the visual cortex
    the number of empty, block cells around the robot
    the position of the robot and distance from the
    destination
    """

    def __init__(self):
        super().__init__()
        self.l1 = Sequential(Linear(512, 256), BatchNorm1d(256), ReLU())
        self.l2 = Sequential(Linear(256, 6), ReLU())

    def forward(self, input):
        return self.l2(self.l1(input))


class BrainV1(Module):

    def __init__(self, state_size: Union[List[int], Tuple[int]], extradata_size: Union[List[int], Tuple[int]]):
        super().__init__()
        self.visual = VisualCortex(state_size)
        self.q_est = QValueModule([512], extradata_size)
        self.aux = AuxModule()

    def forward(self, state, extra):
        vis_output = self.visual(state)
        output = self.q_est(vis_output, extra)
        aux_output = self.aux(vis_output)
        return output, aux_output
