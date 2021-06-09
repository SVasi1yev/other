import unittest
from unittest.mock import patch
import tic_tac_game


class TicTacTest(unittest.TestCase):
    def setUp(self):
        self.game_obj = tic_tac_game.TicTacGame()
        self.game_obj.cur_state = -1
        self.game_obj.turn = 0
        self.game_obj.field = [[0] * 3 for i in range(3)]

    @patch('builtins.input', return_value='1 1')
    def test_correct_input(self, _input):
        i, j = self.game_obj.get_input()
        self.assertEqual(i, 0)
        self.assertEqual(j, 0)

    @patch('builtins.input', return_value='qwe 1')
    def test_wrong_format_input(self, _input):
        self.assertRaises(ValueError, self.game_obj.get_input)

    @patch('builtins.input', return_value='-1 1')
    def test_range_error_input(self, _input):
        self.assertRaises(tic_tac_game.FieldRangeError, self.game_obj.get_input)

    @patch('builtins.input', side_effect=['1 1', '1 1'])
    def test_not_empty_input(self, _input):
        self.game_obj.make_turn()
        self.assertRaises(tic_tac_game.NotEmptyFieldError, self.game_obj.get_input)

    def test_not_finished(self):
        self.game_obj.field[1][1] = 1
        self.game_obj.field[2][2] = 2
        self.game_obj.field[0][1] = 1
        self.game_obj.check_finished()
        self.assertEqual(self.game_obj.cur_state, -1)

    def test_tic_finished(self):
        for i in range(3):
            self.game_obj.field[i][i] = 1
        self.game_obj.check_finished()
        self.assertEqual(self.game_obj.cur_state, 1)

    def test_tac_finished(self):
        for i in range(3):
            self.game_obj.field[i][i] = 2
        self.game_obj.check_finished()
        self.assertEqual(self.game_obj.cur_state, 2)

    def test_tie_finished(self):
        tic_tac = 0
        turns = [(2, 2), (1, 1), (1, 3), (3, 1), (2, 1), (2, 3), (1, 2), (3, 2), (3, 3)]
        for turn in turns:
            self.game_obj.field[turn[0] - 1][turn[1] - 1] = tic_tac + 1
            tic_tac = (tic_tac + 1) % 2
        self.game_obj.check_finished()
        self.assertEqual(self.game_obj.cur_state, 0)

    @patch('builtins.input',
           side_effect=['2 2', '1 1', '1 3', '3 1', '2 1', '2 3', '1 2', '3 2', '3 3'])
    def test_full_game(self, _input):
        self.game_obj.start_game()
        self.assertEqual(self.game_obj.cur_state, 0)