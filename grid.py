from cmath import sqrt
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Cell:

    @classmethod
    def to_bool_array(cls, value, explored=None):
        if explored is None:
            explored = False
        has_player = False
        destination = False
        empty = False
        obstacle = False
        if value == 0:
            empty = True
        if value == 1:
            obstacle = True
        if value == 2:
            has_player = True
            explored = True
        if value == 3:
            destination = True
            explored = True
        unknown = not explored
        return [unknown, empty, obstacle, has_player, destination]


@dataclass
class Point:
    x: int
    y: int

    def __add__(self, other):
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)

    def __iadd__(self, other):
        if isinstance(other, Point):
            self.x += other.x
            self.y += other.y
        return self

    def __iter__(self):
        yield self.x
        yield self.y

    def euclidean_distance(self, other: 'Point'):
        if not isinstance(other, Point):
            raise TypeError(f'Euclidean distance must be calculated between two points instances - '
                            f'other type is {type(other)}')
        distance = (other.x - self.x) ** 2 + (other.y - self.y) ** 2
        distance = sqrt(distance)
        return distance.real

    def manhattan_distance(self, other: 'Point'):
        if not isinstance(other, Point):
            raise TypeError(f'Manhattan distance must be calculated between two points instances - '
                            f'other type is {type(other)}')
        distance = abs(other.x - self.x) + abs(other.y - self.y)
        return distance

    def __getitem__(self, item):
        if item == 0:
            return self.x
        elif item == 1:
            return self.y
        else:
            raise ValueError(f"input must be 0 or 1 - your input is {item}")


class Direction(Enum):
    North = Point(0, 1)
    South = Point(0, -1)
    Est = Point(1, 0)
    West = Point(-1, 0)

    @classmethod
    def from_index(cls, i):
        __lst = [cls.North, cls.South, cls.Est, cls.West]
        return __lst[i]


class Grid:

    def __init__(self, grid):
        self._grid = grid
        self.shape = list(self._grid.shape)
        self.w, self.h = self.shape[:2]
        self.grid = self._public_grid(grid)

    def as_int(self, standardize=False):
        return self.grid[:, :, 1:]

    @classmethod
    def from_string(cls, string):
        lines = string.split('\n')
        if not lines[-1]:
            del lines[-1]
        w = len(lines[0]) + 2
        h = len(lines) + 2
        grid = np.ndarray((w, h, 5), dtype=np.bool)
        for i in range(len(lines)):
            lines[i] = '1' + lines[i] + '1'
        lines.insert(0, '1' * w)
        lines.append('1' * w)
        player_position = None
        destination_position = None
        for i, line in enumerate(lines):
            for j, char in enumerate(line):
                grid[i, j] = Cell.to_bool_array(int(char))
                if i == 0 or j == 0 or i == (w - 1) or j == (h - 1):
                    grid[i, j, 0] = False
                if grid[i, j, 3]:
                    player_position = Point(i, j)
                if grid[i, j, 4]:
                    destination_position = Point(i, j)
        instance = cls(grid)
        instance.destination_position = destination_position
        instance.initial_player_position = player_position
        return instance

    @classmethod
    def from_file(cls, fname: str):
        with open(fname) as f:
            txt = f.read()
        return cls.from_string(txt)

    def __getitem__(self, item):
        return self._grid[item[0], item[1]]

    __slots__ = ['destination_position', 'initial_player_position', '_grid', 'w', 'h', 'shape', 'grid']

    def explore(self, i, j):
        self._grid[i, j, 0] = False
        self.grid[i,j,:] = self._grid[i,j,:]

    def has_player(self, i, j):
        return self.grid[i, j, 3]

    def explored(self, i, j):
        return not self.grid[i, j, 0]

    def obstacle(self, i, j):
        return self.grid[i, j, 2]

    def empty(self, i, j):
        return self.grid[i, j, 1]

    def destination(self, i, j):
        return self.grid[i, j, 4]

    def set_player(self, i, j, value):
        if not value:
            self.grid[i, j, 3] = False
            self.grid[i, j, 1] = True
        else:
            self.grid[i, j, 3] = True
            self.grid[i, j, 1] = False

    def _public_grid(self, grid):
        public_grid = np.zeros(grid.shape, dtype='bool')
        for i in range(self.w):
            for j in range(self.h):
                public_grid[i, j, 0] = True
                if not grid[i, j, 0]: #i.e. if explored
                    public_grid[i,j,:] = grid[i,j,:]
        return public_grid


