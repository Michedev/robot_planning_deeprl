from QAgent import QAgent
from grid import Direction, Point, Grid
from numba import jit, njit
from numpy import sign


class Game:

    def __init__(self, grid_string):
        self.grid_string = grid_string
        self.grid = Grid.from_string(grid_string)
        self.agent = QAgent(self.grid.shape)
        self.player_position = self.grid.initial_player_position
        self.min_distance = (self.grid.w * self.grid.h) ** 2
        self.first_run = True
        self.first_turn = True
        self.turn = 0

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
            into_obstacle = self.grid.obstacle(*p_pos)
            return not into_obstacle
        return False

    def explore_cells(self, position):
        cells_explored = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                direction = Point(i, j)
                if self.is_valid_move(position, direction):
                    new_position = position + direction
                    if not self.grid.explored(*new_position):
                        cells_explored += 1
                        self.grid.explore(*new_position)
        return cells_explored

    def move(self, direction):
        old_position = self.player_position
        if self.is_valid_move(old_position, direction):
            new_position = old_position + direction
            self.player_position = new_position
            self.grid.set_player(old_position.x, old_position.y, value=False)
            self.grid.set_player(new_position.x, new_position.y, value=True)
            if self.grid.destination(*new_position):
                return 1, 1
            cells_explored = self.explore_cells(new_position)
            extra = self.calc_extra_reward(cells_explored, new_position, old_position)
            return 0, extra
        return -1, -0.5

    def calc_extra_reward(self, cells_explored, new_position, prev_position):
        curr_distance = new_position.manhattan_distance(self.grid.destination_position)
        prev_distance = prev_position.manhattan_distance(self.grid.destination_position)
        extra_reward = curr_distance / 100 / 200 * sign(prev_distance - curr_distance)
        return extra_reward

    def run_turn(self):
        if self.first_turn:
            self.first_turn = False
            self.explore_cells(self.player_position)
        int_grid = self.grid.as_int()
        move = self.agent.decide(int_grid)
        direction = Direction.from_index(move).value
        move_result, reward = self.move(direction)
        self.agent.get_reward(self.grid.as_int(standardize=True), reward, self.player_position)
        if self.turn % 100 == 0:
            print('move_result', move_result, 'current position ', self.player_position, sep=' ')
        self.turn += 1
        return move_result

    def play_game(self):
        self.agent.reset()
        self.player_position = self.grid.initial_player_position
        print('\n\n\tDestination is in ' + str(self.grid.destination_position) + '\n\n' + ('-' * 100))
        if self.first_run:
            self.first_run = False
        move_result = -1
        counter_moves = 0
        self.first_turn = True
        while move_result != 1:
            move_result = self.run_turn()
            counter_moves += 1
        print('=' * 100)
        print(f'\n\tmoves to reach destination: {counter_moves}')
        print('=' * 100)
        return counter_moves

    def load_from_file(self, fname):
        self.grid = Grid.from_file(fname)

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
