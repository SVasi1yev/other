import os


class FieldRangeError(Exception):
    pass


class NotEmptyFieldError(Exception):
    pass


class TicTacGame:
    _TEAMS = ('крестики', 'нолики')

    def __init__(self):
        self.field = None
        self.turn = None
        self.cur_state = None

    def check_finished(self):
        if self.field[0][0] == self.field[1][1] == self.field[2][2] != 0:
            self.cur_state = self.field[0][0]
            return

        if self.field[0][2] == self.field[1][1] == self.field[2][0] != 0:
            self.cur_state = self.field[2][0]
            return

        for i in range(3):
            if self.field[i][0] == self.field[i][1] == self.field[i][2] != 0:
                self.cur_state = self.field[i][0]
                return
            if self.field[0][i] == self.field[1][i] == self.field[2][i] != 0:
                self.cur_state = self.field[0][i]
                return

        for i in range(3):
            for j in range(3):
                if self.field[i][j] == 0:
                    return
        self.cur_state = 0

    def get_input(self, error_message=''):
        _input = input(f'{error_message}Введите два целых числа от 1 до 3 через пробел: ')
        i, j = tuple(map(lambda x: int(x) - 1, _input.split(' ')))
        if not 0 <= i <= 2 or not 0 <= j <= 2:
            raise FieldRangeError
        if self.field[i][j] != 0:
            raise NotEmptyFieldError
        return i, j

    def make_turn(self):
        error = ''
        i, j = None, None
        while error is not None:
            try:
                i, j = self.get_input(error)
            except ValueError:
                error = 'Неправильный формат ввода // '
            except FieldRangeError:
                error = 'Значения не попали в необходимый диапозон // '
            except NotEmptyFieldError:
                error = 'Данное поле уже занято // '
            else:
                error = None

        if self.turn == 0:
            self.field[i][j] = 1
            self.turn = 1
        elif self.turn == 1:
            self.field[i][j] = 2
            self.turn = 0
        self.check_finished()

    def print_field(self):
        for i in range(3):
            for j in range(3):
                if self.field[i][j] == 0:
                    print('_', end='')
                elif self.field[i][j] == 1:
                    print('X',end='')
                elif self.field[i][j] == 2:
                    print('O', end='')
            print()

    def print_game_state(self):
        os.system('clear')
        if self.cur_state == -1:
            print(f'Игра продожается. {self._TEAMS[self.turn].capitalize()} ходят.')
        elif self.cur_state == 0:
            print('Игра закончилась ничьей.')
        elif self.cur_state == 1:
            print('Игра закончилась победой крестиков.')
        elif self.cur_state == 2:
            print('Игра закончилась победой ноликов.')
        print()
        self.print_field()

    def start_game(self):
        self.cur_state = -1
        self.turn = 0
        self.field = [[0] * 3 for i in range(3)]
        self.print_game_state()
        while self.cur_state == -1:
            self.make_turn()
            self.print_game_state()


if __name__ == '__main__':
    game = TicTacGame()
    game.start_game()
