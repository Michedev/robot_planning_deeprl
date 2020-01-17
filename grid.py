from cmath import sqrt
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

@dataclass
class Cell:


    _value: int
    explored: bool = False

    def __post_init__(self):
        """
        @:param value has the following meaning
        - -1: Unknown cell
        - 0: Empty cell
        - 1: Block cell
        - 2: Player cell
        - 3: Destination Cell
        """
        if self._value == 2 or self._value == 3:
            self.explored = True
        if not self.explored:
            self.value = -1
        else:
            self.value = self._value

    @property
    def empty(self):
        return self._value == 0

    @property
    def obstacle(self):
        return self._value == 1

    @property
    def has_player(self):
        return self._value == 2

    @property
    def destination(self):
        return self._value == 3

    def __setattr__(self, key, value):
        if key == 'explored' and isinstance(value, bool):
            self.__dict__['explored'] = value
            if value:
                self.value = self._value
        else:
            super(Cell, self).__setattr__(key, value)



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
            raise TypeError(f'Euclidean instance must be calculated between two points instances - '
                            f'other type is {type(other)}')
        distance = (other.x - self.x) ** 2 + (other.y - self.y) ** 2
        distance = sqrt(distance)
        return distance.real

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
        self.grid = grid
        self.shape = self.grid.shape
        self.w, self.h = self.shape

    def as_int(self, true_value=False, standardize=False):
        if true_value:
            f = lambda c: c._value
        else:
            f = lambda c: c.value
        f = np.vectorize(f, otypes=[np.int8])
        out = f(self.grid)
        if standardize:
            out = out.astype('float32')
            out -= -1
            out /= (4 - (-1))
        return out

    @classmethod
    def from_string(cls, string):
        lines = string.split('\n')
        if not lines[-1]:
            del lines[-1]
        w = len(lines[0])
        h = len(lines)
        grid = np.ndarray((w, h), dtype=Cell)
        player_position = None
        destination_position = None
        for j, line in enumerate(lines):
            for i, char in enumerate(line):
                grid[i, j] = Cell(int(char))
                if grid[i, j].has_player:
                    player_position = Point(i, j)
                if grid[i, j].destination:
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
        return self.grid[item[0], item[1]]

    __slots__ = ['destination_position', 'initial_player_position', 'grid', 'w', 'h', 'shape']
