from typing import Iterable, List, Tuple, Union, Any

import torch
import numpy as np
from random import random, randint
from grid import Point
from functools import reduce
from abc import ABC, abstractmethod, abstractproperty
from operator import mul
from torch.nn import *

class ResidualBlock(Module):

    def __init__(self, in_filters: int, out_filters: int, num_convs: int = 3, dropout=0.3):
        super(ResidualBlock, self).__init__()
        self.layers = []
        for i in range(num_convs):
            block = Sequential(Conv2d(in_filters if i == 0 else out_filters, out_filters, kernel_size=3, padding=1),
                               BatchNorm2d(out_filters),
                               LeakyReLU(0.1))
            self.layers.append(block)
        self.layers = Sequential(*self.layers)
        self.conv1x1 = Conv2d(in_filters, out_filters, kernel_size=1)
        self.dropout = Dropout2d(dropout)
        self.maxpool = MaxPool2d(2)

    def forward(self, input):
        output = input
        for l in self.layers:
            output = l(output)
        output2 = self.conv1x1(input)
        output += output2
        output = self.maxpool(output)
        output = self.dropout(output)
        return output


class VisualCortex(Module):

    def __init__(self, input_size: Union[List[int], Tuple[int]]):
        super().__init__()
        self.l1 = ResidualBlock(input_size[0], 64)
        self.l2 = ResidualBlock(64, 64)
        self.l3 = ResidualBlock(64, 64)



    def forward(self, input: torch.Tensor):
        output = self.l1(input)
        output = self.l2(output)
        output = self.l3(output)
        output = torch.flatten(output, start_dim=1)
        return output


class QValueModule(Module):

    def __init__(self, input_shape1: List[int], input_shape2: List[int]):
        super().__init__()
        self.l1 = Sequential(Linear(64, 32),
                             BatchNorm1d(32),
                             ReLU()
                             )
        self.l2 = Sequential(Linear(32, 4))

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
