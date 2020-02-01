from fire import Fire
from grid import Grid
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def viz(filename):
    g = Grid.from_file(filename)
    grid = g._grid[:, :, 1:].astype('int')
    grid = numeric_repr_grid(grid)
    sns.heatmap(grid, cbar=False)
    plt.show()


def numeric_repr_grid(grid):
    for i in range(grid.shape[-1]):
        grid[:, :, i] *= (i + 1)
    grid = np.sum(grid, axis=-1)
    return grid


if __name__ == '__main__':
    Fire(viz)
