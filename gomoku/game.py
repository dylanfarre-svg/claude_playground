"""
M,N,K game — gomoku/tic-tac-toe style.
Pieces placed freely on any empty cell (no gravity).

Default: 7x7 board, 4-in-a-row to win.
"""

import numpy as np

M = 7   # rows
N = 7   # cols
K = 4   # in-a-row to win


class MNKGame:
    def __init__(self, m=M, n=N, k=K):
        self.m = m  # rows
        self.n = n  # cols
        self.k = k  # win length
        self.action_size = m * n

    def get_init_board(self):
        return np.zeros((self.m, self.n), dtype=np.int8)

    def get_board_size(self):
        return (self.m, self.n)

    def action_to_rc(self, action):
        return divmod(action, self.n)

    def rc_to_action(self, row, col):
        return row * self.n + col

    def get_valid_moves(self, board):
        valid = np.zeros(self.action_size, dtype=np.float32)
        for r in range(self.m):
            for c in range(self.n):
                if board[r][c] == 0:
                    valid[self.rc_to_action(r, c)] = 1
        return valid

    def get_next_state(self, board, player, action):
        row, col = self.action_to_rc(action)
        if board[row][col] != 0:
            raise ValueError(f"Cell ({row},{col}) is already occupied")
        b = board.copy()
        b[row][col] = player
        return b, -player

    def get_game_ended(self, board, player):
        """
        Returns:
          1    if `player` has won
         -1    if `player` has lost
          0    if ongoing
          1e-4 if draw
        """
        for r in range(self.m):
            for c in range(self.n):
                if board[r][c] == 0:
                    continue
                owner = board[r][c]
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    if self._check_win(board, r, c, dr, dc, owner):
                        return owner * player

        if not np.any(self.get_valid_moves(board)):
            return 1e-4  # draw

        return 0

    def _check_win(self, board, r, c, dr, dc, player):
        for i in range(self.k):
            nr, nc = r + dr * i, c + dc * i
            if nr < 0 or nr >= self.m or nc < 0 or nc >= self.n:
                return False
            if board[nr][nc] != player:
                return False
        return True

    def get_canonical_form(self, board, player):
        return player * board

    def get_symmetries(self, board, pi):
        pi_board = pi.reshape(self.m, self.n)
        syms = []
        b = board
        p = pi_board
        for rot in range(4):
            syms.append((b, p.flatten()))
            syms.append((np.fliplr(b), np.fliplr(p).flatten()))
            b = np.rot90(b)
            p = np.rot90(p)
        return syms

    def string_representation(self, board):
        return board.tobytes()

    def display(self, board, last_move=None):
        col_labels = "  " + "  ".join(str(c) for c in range(self.n))
        print(col_labels)
        divider = "  " + "--" * self.n
        print(divider)
        for r in range(self.m - 1, -1, -1):
            row_str = []
            for c in range(self.n):
                v = board[r][c]
                cell = "X" if v == 1 else ("O" if v == -1 else ".")
                if last_move is not None and last_move == (r, c):
                    cell = f"[{cell}]"
                    row_str.append(cell.ljust(3))
                else:
                    row_str.append(f" {cell} ")
            print(f"{r}|" + " ".join(row_str))
        print(divider)
        print(col_labels)
        print()
