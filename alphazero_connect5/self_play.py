"""
Self-play episode generator.
Plays one game using MCTS + current network, returns training examples.
"""

import numpy as np
from mcts import MCTS


def execute_episode(game, nnet, args):
    """
    Play one self-play game.
    Returns list of (canonical_board, pi, z) tuples where z is the outcome.
    """
    mcts = MCTS(game, nnet, args)
    examples = []
    board = game.get_init_board()
    current_player = 1
    step = 0

    while True:
        step += 1
        canonical = game.get_canonical_form(board, current_player)

        # Use temperature 1 for first `tempThreshold` moves, then greedy
        temp = 1 if step <= args.tempThreshold else 0

        pi = mcts.get_action_probs(canonical, temp=temp)

        # Data augmentation: symmetries
        syms = game.get_symmetries(canonical, pi)
        for sym_board, sym_pi in syms:
            examples.append([sym_board, sym_pi, None])  # value filled in later

        # Sample action from pi
        action = np.random.choice(len(pi), p=pi)
        board, current_player = game.get_next_state(board, current_player, action)

        result = game.get_game_ended(board, current_player)
        if result != 0:
            # Assign values: winner gets +1, loser gets -1
            final_examples = []
            for board_state, pi_state, _ in examples:
                # The canonical board was from the perspective of whoever played it.
                # We stored symmetries of the canonical form, so just assign result
                # relative to current_player at end.
                # result > 0 means current_player won, so the player who was
                # "current" at each step: alternating from the last player.
                final_examples.append((board_state, pi_state, result))
            return final_examples
