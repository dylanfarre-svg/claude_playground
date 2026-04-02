"""
Connect5 Game: 10x10 board, column-drop mechanics (like Connect Four), 5-in-a-row wins.

Board representation:
  - board[row][col], row 0 = bottom, row 9 = top
  - 1 = current player, -1 = opponent, 0 = empty
  - Canonical form always shows current player as 1
"""

import numpy as np

ROWS = 10
COLS = 10
WIN = 5


class Connect5Game:
    def __init__(self):
        self.rows = ROWS
        self.cols = COLS
        self.win = WIN

    def get_init_board(self):
        return np.zeros((self.rows, self.cols), dtype=np.int8)

    def get_board_size(self):
        return (self.rows, self.cols)

    def get_action_size(self):
        return self.cols  # one action per column

    def get_next_state(self, board, player, action):
        """Drop a piece in column `action`. Returns (new_board, next_player)."""
        b = board.copy()
        col = action
        for row in range(self.rows):
            if b[row][col] == 0:
                b[row][col] = player
                return b, -player
        raise ValueError(f"Column {col} is full")

    def get_valid_moves(self, board):
        """Returns binary vector of valid column drops."""
        valid = np.zeros(self.cols, dtype=np.float32)
        for col in range(self.cols):
            if board[self.rows - 1][col] == 0:  # top row empty = column not full
                valid[col] = 1
        return valid

    def get_game_ended(self, board, player):
        """
        Returns:
          1  if `player` won
         -1  if `player` lost
          0  if game ongoing
          1e-4 (small positive) for draw
        """
        b = board
        # Check all directions for WIN in a row
        for r in range(self.rows):
            for c in range(self.cols):
                if b[r][c] == 0:
                    continue
                owner = b[r][c]
                # Check 4 directions: horizontal, vertical, diag, anti-diag
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    if self._check_win(b, r, c, dr, dc, owner):
                        return owner * player  # 1 if player won, -1 if lost

        # Draw: no valid moves left
        if np.all(self.get_valid_moves(b) == 0):
            return 1e-4

        return 0

    def _check_win(self, board, r, c, dr, dc, player):
        for i in range(self.win):
            nr, nc = r + dr * i, c + dc * i
            if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
                return False
            if board[nr][nc] != player:
                return False
        return True

    def get_canonical_form(self, board, player):
        """Flip board so current player is always 1."""
        return player * board

    def get_symmetries(self, board, pi):
        """
        Returns list of (board, pi) pairs for data augmentation.
        Connect5 with column-drop: only horizontal mirror is valid.
        """
        syms = [(board, pi)]
        # Mirror horizontally
        mirror_board = board[:, ::-1].copy()
        mirror_pi = pi[::-1].copy()
        syms.append((mirror_board, mirror_pi))
        return syms

    def string_representation(self, board):
        return board.tobytes()

    def display(self, board):
        print()
        print("  " + " ".join(str(c) for c in range(self.cols)))
        for r in range(self.rows - 1, -1, -1):
            row_str = []
            for c in range(self.cols):
                v = board[r][c]
                row_str.append("X" if v == 1 else ("O" if v == -1 else "."))
            print(f"{r:2d} " + " ".join(row_str))
        print()
