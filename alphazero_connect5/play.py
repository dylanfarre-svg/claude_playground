#!/usr/bin/env python3
"""
Play Connect5 against the trained AlphaZero agent.

Usage:
  python play.py
  python play.py --checkpoint checkpoints/iter_0050.pt --sims 200
"""

import argparse
import numpy as np
import torch

from args import args
from game import Connect5Game
from model import NNetWrapper
from mcts import MCTS


def parse_cli():
    parser = argparse.ArgumentParser(description="Play against AlphaZero Connect5")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt")
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--human-first", action="store_true", default=True)
    return parser.parse_args()


def main():
    cli = parse_cli()
    args.numMCTSSims = cli.sims

    game = Connect5Game()
    nnet = NNetWrapper(game, args)

    try:
        import os
        folder = os.path.dirname(cli.checkpoint)
        filename = os.path.basename(cli.checkpoint)
        nnet.load_checkpoint(folder, filename)
        print(f"Loaded checkpoint: {cli.checkpoint}")
    except Exception as e:
        print(f"Could not load checkpoint: {e}")
        print("Playing with untrained network.")

    mcts = MCTS(game, nnet, args)

    board = game.get_init_board()
    human_player = 1
    ai_player = -1
    current_player = 1

    print("\nConnect5 — you are X, AI is O")
    print("Enter column number (0-9) to drop a piece.\n")

    while True:
        game.display(board)

        result = game.get_game_ended(board, current_player)
        if result != 0:
            if result == 1e-4:
                print("Draw!")
            elif result > 0:
                print("Current player wins!")
            else:
                print("Other player wins!")
            break

        if current_player == human_player:
            valid = game.get_valid_moves(board)
            valid_cols = [str(c) for c in range(game.cols) if valid[c]]
            print(f"Valid columns: {', '.join(valid_cols)}")
            while True:
                try:
                    col = int(input("Your move (column): "))
                    if 0 <= col < game.cols and valid[col]:
                        break
                    print("Invalid move, try again.")
                except (ValueError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    return
            action = col
        else:
            print("AI is thinking...")
            canonical = game.get_canonical_form(board, current_player)
            pi = mcts.get_action_probs(canonical, temp=0)
            action = np.argmax(pi)
            print(f"AI plays column {action}")

        board, current_player = game.get_next_state(board, current_player, action)

    game.display(board)


if __name__ == "__main__":
    main()
