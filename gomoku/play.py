#!/usr/bin/env python3
"""
Two-player gomoku-style MNK game in the terminal.

Default: 7x7 board, 4-in-a-row to win.

Usage:
  python play.py
  python play.py --m 9 --n 9 --k 5    # 9x9 gomoku
  python play.py --m 3 --n 3 --k 3    # tic-tac-toe

Input: enter row and column separated by a space or comma.
  e.g.  3 4   or   3,4
"""

import argparse
import sys
import numpy as np
from game import MNKGame


PLAYER_NAMES = {1: "Player 1 (X)", -1: "Player 2 (O)"}


def parse_cli():
    parser = argparse.ArgumentParser(description="Two-player MNK game")
    parser.add_argument("--m", type=int, default=7, help="Rows (default 7)")
    parser.add_argument("--n", type=int, default=7, help="Cols (default 7)")
    parser.add_argument("--k", type=int, default=4, help="In-a-row to win (default 4)")
    return parser.parse_args()


def get_player_move(game, board, player):
    valid = game.get_valid_moves(board)
    name = PLAYER_NAMES[player]

    while True:
        try:
            raw = input(f"{name} — enter row col (e.g. 3 4): ").strip()
            if not raw:
                continue
            # Accept "3 4", "3,4", "3-4"
            parts = raw.replace(",", " ").replace("-", " ").split()
            if len(parts) != 2:
                print("  Enter two numbers: row and column.")
                continue
            row, col = int(parts[0]), int(parts[1])
            if row < 0 or row >= game.m or col < 0 or col >= game.n:
                print(f"  Out of range. Row: 0-{game.m-1}, Col: 0-{game.n-1}")
                continue
            action = game.rc_to_action(row, col)
            if valid[action] == 0:
                print("  That cell is already taken.")
                continue
            return action, (row, col)
        except ValueError:
            print("  Invalid input — enter two integers.")
        except (EOFError, KeyboardInterrupt):
            print("\nGame aborted.")
            sys.exit(0)


def print_header(m, n, k):
    print()
    print("=" * 40)
    print(f"  MNK GAME  —  {m}×{n} board, {k}-in-a-row wins")
    print("=" * 40)
    print("  Player 1 = X    Player 2 = O")
    print("  Enter: row col  (row 0 = bottom)")
    print("=" * 40)
    print()


def main():
    args = parse_cli()
    game = MNKGame(m=args.m, n=args.n, k=args.k)

    print_header(args.m, args.n, args.k)

    board = game.get_init_board()
    current_player = 1
    last_move = None
    move_count = 0

    game.display(board)

    while True:
        result = game.get_game_ended(board, current_player)
        if result != 0:
            if abs(result) < 0.01:  # draw
                print("It's a draw!")
            elif result > 0:
                print(f"{PLAYER_NAMES[current_player]} wins!")
            else:
                print(f"{PLAYER_NAMES[-current_player]} wins!")
            break

        action, last_move = get_player_move(game, board, current_player)
        board, current_player = game.get_next_state(board, current_player, action)
        move_count += 1

        print()
        game.display(board, last_move=last_move)
        print(f"  Move {move_count}: {PLAYER_NAMES[-current_player]} played ({last_move[0]}, {last_move[1]})")
        print()

    # Final board
    print()
    game.display(board, last_move=last_move)

    # Replay prompt
    try:
        again = input("Play again? (y/n): ").strip().lower()
        if again == "y":
            main()
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    main()
