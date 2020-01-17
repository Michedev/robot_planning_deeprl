from fire import Fire
from grid import Grid
import seaborn as sns
import matplotlib.pyplot as plt

def viz(filename):
    g = Grid.from_file(filename)
    sns.heatmap(g.as_int(true_value=True), cbar=False)
    plt.show()

if __name__ == '__main__':
    Fire(viz)
