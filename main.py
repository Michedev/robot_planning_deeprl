from game import Game
import os
from path import Path
FOLDER = Path(__file__).parent
MAPS = FOLDER / 'maps'

os.system('rm -rf robot_logs/')
g = None
for file in MAPS.files('*.txt'):
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
