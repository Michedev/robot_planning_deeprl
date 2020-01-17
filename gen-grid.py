from fire import Fire
import numpy as np
from random import randint


def gen_grid(w: int, h: int, nobstacles: int, x_agent: int, y_agent: int, x_destination: int, y_destination: int, output: str = "grid.txt"):
    """
    Generate the map for the motion planning robot
    :param w: width of the grid - type: int
    :param h: height of the grid - type: int
    :param nobstacles: number of obstacles to random put on the map - type: int
    :param x_agent: position of the agent on X axis - type: int
    :param y_agent: position of the agent on Y axis - type: int
    :param x_destination: position of the destination on X axis - type: int
    :param y_destination: position of the destination on Y axis - type: int
    :param output: output filename - type: string
    """
    grid = np.zeros((w, h), dtype='int8')
    grid[x_agent, y_agent] = 2
    grid[x_destination, y_destination] = 3
    for i in range(nobstacles):
        random_loc = (randint(0, w - 1), randint(0, h - 1))
        while grid[random_loc] != 0:
            random_loc = (randint(0, w - 1), randint(0, h - 1))
        grid[random_loc] = 1
    np.savetxt(output, grid, fmt='%d', delimiter='')


if __name__ == '__main__':
    Fire(gen_grid)
