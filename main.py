from game import Game
import os
from path import Path
from random import shuffle, seed
import torch

FOLDER = Path(__file__).parent
MAPS = FOLDER / 'maps'

os.system('rm -rf robot_logs/')
g = None
maps = MAPS.files('*.txt')
seed(13)
first = True
for round in range(100):
    if first:
        maps = sorted(maps)
        first = False
    else:
        shuffle(maps)
    for file in maps:
        print(('=' * 100) + '\n\n')
        print('playing map ', file)
        print('\n\n' + ('=' * 100))
        if g is None:
            g = Game.from_file(file)
        else:
            g.load_from_file(file)
        g.play_game()

        torch.save(g.agent.brain, 'brain')
