from fire import Fire
import numpy as np
from random import randint



def gen_grid(w: int = 30, h: int = 30, nobstacles: int = 20, x_agent: int = 0, y_agent: int = 0, x_destination: int = 20, y_destination: int= 20, output: str = "grid.txt", set30x30=False, set10x10=False):
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
    :param set30x30: generate a set of 30x30 maps of increasing size of number of blocks
    """
    if set10x10:
        gen_grids_10x10()
        return
    elif set30x30:
        gen_grids_30x30()
        return
    grid = np.zeros((w, h), dtype='int8')
    grid[x_agent, y_agent] = 2
    grid[x_destination, y_destination] = 3
    for i in range(nobstacles):
        random_loc = (randint(0, w - 1), randint(0, h - 1))
        while grid[random_loc] != 0:
            random_loc = (randint(0, w - 1), randint(0, h - 1))
        grid[random_loc] = 1
    np.savetxt(output, grid, fmt='%d', delimiter='')

def gen_grids_10x10():
    x0, y0 = 0,0
    maps = 10
    difficulty = [('easy', 0.01), ('medium', 0.1), ('hard', 0.2), ('impossible', 0.4)]
    for d, perc_obs in difficulty:
        for i in range(maps):
            gen_grid(10, 10, int(10 * 10 * perc_obs), x0, y0, randint(5, 9), randint(5, 9), f'maps/grid_{d}_{i}.txt')


def gen_grids_30x30():
    x0, y0 = 0,0
    maps = 10
    difficulty = [('easy', 0.01), ('medium', 0.1), ('hard', 0.2), ('impossible', 0.4)]
    for d, perc_obs in difficulty:
        for i in range(maps):
            gen_grid(30, 30, int(30 * 30 * perc_obs), x0, y0, randint(15, 29), randint(15, 29), f'maps/grid_{d}_{i}.txt')


if __name__ == '__main__':
    Fire(gen_grid)
