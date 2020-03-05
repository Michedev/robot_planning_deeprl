from fire import Fire

from QAgent import QAgent
from game import Game
from path import Path
from random import shuffle, seed
import torch

FOLDER = Path(__file__).parent
MAPS = FOLDER / 'maps'


def main(epochs=3000, discount_factor=0.9, experience_size=500_000, update_q_fut=1_000,
         sample_experience=128, update_freq=4, meta_learning=False, no_update_start=500, random_state=13):
    g = None
    maps = MAPS.files('*.txt')
    seed(random_state)
    agent = QAgent((10, 10, 5), discount_factor=discount_factor, experience_size=experience_size,
                   update_q_fut=update_q_fut, sample_experience=sample_experience,
                   update_freq=update_freq, no_update_start=no_update_start,
                   meta_learning=meta_learning)
    first = True
    for round in range(epochs):
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
                g = Game.from_file(agent, file)
            else:
                g.load_from_file(file)
            g.play_game()

            torch.save(agent.brain.state_dict(), 'brain.pth')


if __name__ == '__main__':
    Fire(main)
