from QAgent import QAgent
from grid import Direction, Point, Grid
from numba import jit, njit
from numpy import sign


class Game:

    def __init__(self, grid_string):
        self.grid_string = grid_string
        self.grid = Grid.from_string(grid_string)
        self.agent = QAgent(self.grid)
        self.player_position = self.grid.initial_player_position
        self.min_distance = (self.grid.w * self.grid.h) ** 2
        self.first_run = True
        self.first_turn = True

    @classmethod
    def from_file(cls, grid_fname):
        with open(grid_fname) as f:
            txt = f.read()
        return cls(txt)

    def is_outofbounds(self, position, direction):
        p_pos = position + direction
        outofbounds = any(axis < 0 for axis in p_pos) or any(
            axis >= max_axis for axis, max_axis in zip(p_pos, self.grid.shape))
        return outofbounds

    def is_valid_move(self, position, direction):
        p_pos = position + direction
        outofbounds = any(axis < 0 for axis in p_pos) or any(axis >= max_axis for axis, max_axis in zip(p_pos, self.grid.shape))
        if not outofbounds:
            into_obstacle = self.grid[p_pos].obstacle
            return not into_obstacle
        return False

    def explore_cells(self, position):
        cells_explored = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                direction = Point(i, j)
                if self.is_valid_move(position, direction):
                    new_position = position + direction
                    if not self.grid[new_position].explored:
                        cells_explored += 1
                    self.grid[new_position].explored = True
        return cells_explored

    def move(self, direction):
        old_position = self.player_position
        if self.is_valid_move(old_position, direction):
            new_position = old_position + direction
            self.player_position = new_position
            self.grid[old_position].value = 0
            self.grid[new_position].value = 2
            if self.grid[new_position].destination:
                return 1, 1
            cells_explored = self.explore_cells(new_position)
            extra = self.calc_extra_reward(cells_explored, new_position, old_position)
            return 0, extra
        return -1, -1

    def calc_extra_reward(self, cells_explored, new_position, prev_position):
        curr_distance = new_position.manhattan_distance(self.grid.destination_position)
        prev_distance = prev_position.manhattan_distance(self.grid.destination_position)
        extra_reward = curr_distance / 100 / 200 * sign(prev_distance - curr_distance)
        return extra_reward

    def run_turn(self):
        if self.first_turn:
            self.first_turn = False
            self.explore_cells(self.player_position)
        int_grid = self.grid.as_int(standardize=True)
        move = self.agent.decide(int_grid)
        direction = Direction.from_index(move).value
        print('current position', self.player_position, 'direction', direction, sep=' ')
        move_result, reward = self.move(direction)
        self.agent.get_reward(self.grid.as_int(standardize=True), reward, self.player_position)
        print('move_result', move_result, 'current position ', self.player_position, sep=' ')
        return move_result

    def play_game(self):
        if self.first_run:
            self.first_run = False
        move_result = -1
        counter_moves = 0
        self.first_turn = True
        while move_result != 1:
            move_result = self.run_turn()
            counter_moves += 1
        return counter_moves

    def play_games(self, n=1):
        if self.first_run:
            self.first_run = False
        else:
            self.grid = Grid.from_string(self.grid_string)
            self.player_position = self.grid.initial_player_position
            self.min_distance = (self.grid.w * self.grid.h) ** 2
        for i in range(n):
            moves = self.play_game()
            print('='*100)
            print(f'\n\tmoves to reach destination: {moves}')
            print('='*100)
