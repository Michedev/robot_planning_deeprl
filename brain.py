from typing import Iterable, List, Tuple, Union

import torch
import numpy as np
from random import random, randint
from grid import Point


class VisualCortex(torch.nn.Module):

    def __init__(self, input_size: Union[List[int], Tuple[int]]):
        super().__init__()
        self.input_size = input_size
        self.c1 = torch.nn.Conv2d(input_size[0], 512, kernel_size=3, stride=3, bias=False)
        self.gn1 = torch.nn.GroupNorm(4, 512)

    def forward(self, input):
        output = input
        layers = [self.c1, self.gn1, torch.nn.ReLU()]
        for l in layers:
            output = l(output)
        output = torch.flatten(output, start_dim=1)
        return output


class QValueModule(torch.nn.Module):

    def __init__(self, input_shape1: List[int], input_shape2: List[int]):
        super().__init__()
        self.l1 = torch.nn.Sequential(torch.nn.Linear(8192, 1024, bias=False),
                                      torch.nn.BatchNorm1d(1024),
                                      torch.nn.ReLU()
                                      )
        self.l2 = torch.nn.Sequential(torch.nn.Linear(1024, 128, bias=False),
                                      torch.nn.BatchNorm1d(128),
                                      torch.nn.ReLU()
                                      )
        self.l3 = torch.nn.Sequential(torch.nn.Linear(128 + input_shape2[-1], 4, bias=False))

    def forward(self, state, neightbours):
        output = self.l1(state)
        output = self.l2(output)
        output = torch.cat([output, neightbours], dim=-1)
        return self.l3(output)


class BrainV1(torch.nn.Module):

    def __init__(self, state_size: Union[List[int], Tuple[int]], extradata_size: Union[List[int], Tuple[int]]):
        super().__init__()
        self.visual = VisualCortex(state_size)
        self.q_est = QValueModule([512], extradata_size)

    def forward(self, state, extra):
        output = self.visual(state)
        output = self.q_est(output, extra)
        return output
