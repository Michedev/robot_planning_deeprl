from game import Game
import os

os.system('rm -rf robot_logs/')
g = Game.from_file('grid.txt')
g.play_games()