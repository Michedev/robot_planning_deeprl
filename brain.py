from typing import List, Tuple, Union

import torch
from torch.nn import *


class VisualCortex(Module):

    def __init__(self):
        super().__init__()
        self.l1 = Sequential(Conv2d(4, 32, kernel_size=3, bias=True),
                             ReLU())
        self.l2 = Sequential(Conv2d(32, 256, kernel_size=3, bias=True, stride=2), BatchNorm2d(256), ReLU())
        self.l3 = Sequential(Conv2d(256, 512, kernel_size=3, bias=True, stride=2), BatchNorm2d(512), ReLU())

        self.l4 = Sequential(Linear(512, 64), BatchNorm1d(64), ReLU())

        self.residual = Sequential(Linear(12 * 12 * 4, 64), BatchNorm1d(64), ReLU())
        #

    def forward(self, input: torch.Tensor):
        output = self.l1(input)
        output = self.l2(output)
        output = self.l3(output)
        output = torch.flatten(output, start_dim=1)
        output = self.l4(output)
        output += self.residual(torch.flatten(input, start_dim=1))
        return output


class QValueModule(Module):

    def __init__(self, input_shape_extra: List[int]):
        super().__init__()
        self.l1 = Sequential(Linear(64, 16, bias=True),
                             ReLU(),
                             )
        self.l2 = Sequential(Linear(16 + input_shape_extra[-1], 4, bias=True))
        with torch.no_grad():
            for i in range(4):
                self.l2[0].weight[:, -3 - i * 4] = -0.1  # negative weight for block neighbors
                self.l2[0].weight[:, -1 - i * 4] = 0.1  # positive weight for target neighbors
            # order: [Direction.North, Direction.South, Direction.Est, Direction.West]
            #w, h
            # set right sign weight for distance from solution
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

    def __init__(self, extradata_size: Union[List[int], Tuple[int]]):
        super().__init__()
        self.visual = VisualCortex()
        self.q_est = QValueModule(extradata_size)
        self.curiosity_module = Sequential(Linear(64, 64), BatchNorm1d(64), ReLU(),
                                           Linear(64, 32), BatchNorm1d(32), ReLU(),
                                           Linear(32, 16))
        self.log_softmax_c = LogSoftmax(dim=1)

    def forward(self, state, extra, curiosity=False):
        vis_output = self.visual(state)
        output = self.q_est(vis_output, extra)
        if curiosity:
            c_output = self.curiosity_module(vis_output)
            c_output = c_output.reshape(-1, 4, 4)
            c_output = self.log_softmax_c(c_output)
            return output, c_output
        return output
