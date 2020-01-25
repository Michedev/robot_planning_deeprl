from fire import Fire
from grid import Grid
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def viz(filename):
    g = Grid.from_file(filename)
    grid = g._grid[:, :, 1:].astype('int')
    print(grid[-1, -1])
    for i in range(grid.shape[-1]):
        grid[:, :, i] *= (i+1)
    grid = np.sum(grid, axis=-1)
    print(grid)
    sns.heatmap(grid, cbar=False)
    plt.show()

if __name__ == '__main__':
    Fire(viz)
