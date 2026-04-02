"""
Monte Carlo Tree Search for AlphaZero.

Each node stores:
  N[s][a] — visit count
  W[s][a] — total action value
  Q[s][a] — mean action value = W/N
  P[s][a] — prior probability from network
"""

import math
import numpy as np


EPS = 1e-8


class MCTS:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args

        self.Qsa = {}   # Q values for (s, a)
        self.Nsa = {}   # visit counts for (s, a)
        self.Ns = {}    # visit counts for s
        self.Ps = {}    # prior policy for s

        self.Es = {}    # game ended result for s
        self.Vs = {}    # valid moves for s

    def get_action_probs(self, canonical_board, temp=1):
        """
        Run `numMCTSSims` simulations from the given board.
        Returns action probability vector.
        temp=1: proportional to visit count (exploration)
        temp=0: greedy (pick most visited)
        """
        for _ in range(self.args.numMCTSSims):
            self._search(canonical_board)

        s = self.game.string_representation(canonical_board)
        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.get_action_size())]

        if temp == 0:
            best = np.argmax(counts)
            probs = np.zeros(len(counts))
            probs[best] = 1.0
            return probs

        counts_arr = np.array(counts, dtype=np.float64)
        counts_arr = counts_arr ** (1.0 / temp)
        total = counts_arr.sum()
        if total == 0:
            # Fallback: uniform over valid
            valid = self.game.get_valid_moves(canonical_board)
            return valid / valid.sum()
        return counts_arr / total

    def _search(self, board):
        s = self.game.string_representation(board)

        # Terminal check
        if s not in self.Es:
            self.Es[s] = self.game.get_game_ended(board, 1)
        if self.Es[s] != 0:
            return -self.Es[s]

        # Leaf node — expand
        if s not in self.Ps:
            self.Ps[s], v = self.nnet.predict(board)
            valid = self.game.get_valid_moves(board)
            self.Ps[s] = self.Ps[s] * valid
            mass = self.Ps[s].sum()
            if mass > 0:
                self.Ps[s] /= mass
            else:
                # All masked out — uniform fallback
                self.Ps[s] = valid / (valid.sum() + EPS)
            self.Vs[s] = valid
            self.Ns[s] = 0
            return -v

        valid = self.Vs[s]
        best_u = -float("inf")
        best_a = -1

        # UCB selection
        c_puct = self.args.cpuct
        sqrt_ns = math.sqrt(self.Ns[s] + EPS)
        for a in range(self.game.get_action_size()):
            if valid[a] == 0:
                continue
            q = self.Qsa.get((s, a), 0)
            n = self.Nsa.get((s, a), 0)
            u = q + c_puct * self.Ps[s][a] * sqrt_ns / (1 + n)
            if u > best_u:
                best_u = u
                best_a = a

        a = best_a
        next_board, next_player = self.game.get_next_state(board, 1, a)
        next_board = self.game.get_canonical_form(next_board, next_player)

        v = self._search(next_board)

        # Backup
        sa = (s, a)
        if sa in self.Qsa:
            self.Qsa[sa] = (self.Nsa[sa] * self.Qsa[sa] + v) / (self.Nsa[sa] + 1)
            self.Nsa[sa] += 1
        else:
            self.Qsa[sa] = v
            self.Nsa[sa] = 1

        self.Ns[s] += 1
        return -v
