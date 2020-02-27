from typing import List

from QAgent import QAgent
from grid import *
from numpy import sign
import viz
import seaborn as sns
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from hashlib import md5
import gc


class Game:

    def __init__(self, grid_string, grid_name=None):
        self.grid = Grid.from_string(grid_string)
        self.grid_name = grid_name or md5(grid_string.encode())
        self.agent = QAgent(self.grid.shape)
        self.player_position = self.grid.initial_player_position
        self.min_distance = (self.grid.w * self.grid.h) ** 2
        self.first_run = True
        self.first_turn = True
        self.counter_failures = 0
        # self.frames = []
        self.turn = 0
        self.counter_game = 0

    @classmethod
    def from_file(cls, grid_fname):
        with open(grid_fname) as f:
            txt = f.read()
        return cls(txt, grid_fname)

    def is_outofbounds(self, p):
        outofbounds = any(axis < 0 for axis in p) or \
                      any(axis >= max_axis for axis, max_axis in zip(p, self.grid.shape))
        return outofbounds

    def is_valid_move(self, p):
        if not self.is_outofbounds(p):
            into_obstacle = self.grid.obstacle(p.x, p.y)
            return not into_obstacle
        return False

    def explore_cells(self, position: Point):
        cells_explored = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                direction = Point(i, j)
                new_pos = position + direction
                if not self.is_outofbounds(new_pos):
                    if not self.grid.explored(new_pos.x, new_pos.y):
                        cells_explored += 1
                        self.grid.explore(new_pos.x, new_pos.y)
        return cells_explored

    def move(self, direction):
        old_position = self.player_position
        new_pos = old_position + direction
        if self.is_valid_move(new_pos):
            self.player_position = new_pos
            self.grid.set_player(old_position.x, old_position.y, value=False)
            self.grid.set_player(new_pos.x, new_pos.y, value=True)
            if self.grid.destination(new_pos.x, new_pos.y):
                return 1, 1
            cells_explored = self.explore_cells(new_pos)
            if new_pos in self.past_positions:
                return 0, -0.15
            else:
                self.past_positions.append(new_pos)
                reward = self.calc_extra_reward(cells_explored, new_pos, old_position)
                # reward =  extra
                return 0, reward

        return -1, -0.5

    def calc_extra_reward(self, cells_explored, new_position: Point, prev_position: Point):
        curr_distance = new_position.manhattan_distance(self.grid.destination_position)
        extra_reward = curr_distance / (self.grid.h + self.grid.w)
        extra_reward = -0.1 * (1 - extra_reward)
        return extra_reward

    def run_turn(self):
        if self.first_turn:
            self.first_turn = False
            self.explore_cells(self.player_position)
        int_grid = self.grid.as_int()
        # self.frames.append(int_grid.astype('int'))
        move = self.agent.decide(self.grid_name, int_grid, self.player_position, self.grid.destination_position)
        direction = Direction.from_index(move).value
        move_result, reward = self.move(direction)
        if self.turn % 100 == 0:
            print('Move result', move_result,
                  'Reward', reward,
                  'Player pos', self.player_position,
                  'epsilon', self.agent.epsilon)
        self.agent.get_reward(self.grid_name, self.grid.as_int(), reward, self.player_position)
        self.turn += 1
        return move_result, reward

    def play_game(self):
        self.agent.reset()
        self.player_position = self.grid.initial_player_position
        self.past_positions = [self.grid.initial_player_position]
        print('\n\n\tDestination is in ' + str(self.grid.destination_position) + f' - episode {self.agent.episode}' + '\n\n' + ('-' * 100))
        if self.first_run:
            self.first_run = False
        move_result = -1
        counter_moves = 0
        self.first_turn = True
        tot_reward = 0
        threshold_reward = -0.4 * self.grid.w * self.grid.h
        while move_result != 1:
            move_result, reward = self.run_turn()
            counter_moves += 1
            tot_reward += reward
            if threshold_reward > tot_reward:
                print('too much time to reach destination, fail')
                self.counter_failures += 1
                break
        # self._make_gif(self.frames)
        gc.collect()
        self.counter_game += 1
        self.agent.on_win()
        print('=' * 100)
        print(f'\n\tmoves to reach destination: {counter_moves}')
        print('=' * 100)
        if self.counter_game % 10 == 0:
            win_rate = 1.0 - self.counter_failures / 10
            if self.agent.epsilon < 0.3:
                self.agent.decrease_epsilon = max(0, self.agent.decrease_epsilon - int(30 * (1 - win_rate)))
            self.agent.writer.add_scalar('win rate', win_rate, global_step=self.counter_game)
            self.counter_failures = 0
        return counter_moves

    def load_from_file(self, fname):
        with open(fname) as f:
            grid_string = f.read()
        self.grid_name = fname
        self.grid = Grid.from_string(grid_string)

    def _make_gif(self, frames: List[np.ndarray]):
        f = lambda i: viz.numeric_repr_grid(frames[i])
        print('num frames', len(frames))
        fig: plt.Figure = plt.figure()
        im = plt.imshow(f(0))

        def init():
            im.set_data(f(0))

        def animate(i):
            im.set_data(f(i))

        anim = FuncAnimation(fig, animate, frames=len(frames), repeat=False, init_func=init)
        anim.save('animation.gif', writer='imagemagick', fps=60)
        raise -1
