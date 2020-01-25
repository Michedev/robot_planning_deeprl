from game import Game
import os
from path import Path
from random import shuffle, seed
FOLDER = Path(__file__).parent
MAPS = FOLDER / 'maps'

os.system('rm -rf robot_logs/')
g = None
maps = MAPS.files('*.txt')
seed(13)
for round in range(100):
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
        g.agent.brain.save('brain', include_optimizer=False, save_format='tf')
        g.agent.brain.save_weights('brain.tf')
