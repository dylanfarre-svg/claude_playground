"""
AlphaZero training loop.

Each iteration:
  1. Self-play to generate training examples
  2. Train network on accumulated examples
  3. Save checkpoint
"""

import os
import time
import numpy as np
from collections import deque
from tqdm import tqdm

from game import Connect5Game
from model import NNetWrapper
from self_play import execute_episode


class Trainer:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.train_examples_history = deque(maxlen=args.maxlenOfQueue)

    def learn(self):
        start_iter = 1

        # Resume from checkpoint if available
        checkpoint_file = os.path.join(self.args.checkpoint, "latest.pt")
        history_file = os.path.join(self.args.checkpoint, "history.npy")
        if os.path.exists(checkpoint_file):
            print(f"Loading checkpoint from {checkpoint_file}")
            self.nnet.load_checkpoint(self.args.checkpoint, "latest.pt")
            if os.path.exists(history_file):
                saved = np.load(history_file, allow_pickle=True)
                self.train_examples_history = deque(saved.tolist(), maxlen=self.args.maxlenOfQueue)
                print(f"Loaded {len(self.train_examples_history)} example batches from history")

        for iteration in range(start_iter, self.args.numIters + 1):
            iter_start = time.time()
            print(f"\n{'='*60}")
            print(f"Iteration {iteration}/{self.args.numIters}")
            print(f"{'='*60}")

            # --- Self-play ---
            iteration_examples = []
            print(f"Self-play: {self.args.numEps} episodes")
            for ep in tqdm(range(self.args.numEps), desc="Self-play", ncols=80):
                examples = execute_episode(self.game, self.nnet, self.args)
                iteration_examples.extend(examples)

            self.train_examples_history.append(iteration_examples)
            print(f"  Generated {len(iteration_examples)} positions this iteration")
            print(f"  History size: {sum(len(e) for e in self.train_examples_history)} positions")

            # Flatten all history examples
            all_examples = []
            for batch in self.train_examples_history:
                all_examples.extend(batch)
            np.random.shuffle(all_examples)

            # --- Training ---
            print(f"\nTraining on {len(all_examples)} examples...")
            pi_loss, v_loss = self.nnet.train(all_examples)
            print(f"  Policy loss: {pi_loss:.4f}  Value loss: {v_loss:.4f}")

            # --- Checkpoint ---
            self.nnet.save_checkpoint(self.args.checkpoint, "latest.pt")
            self.nnet.save_checkpoint(self.args.checkpoint, f"iter_{iteration:04d}.pt")
            np.save(history_file, np.array(list(self.train_examples_history), dtype=object))

            elapsed = time.time() - iter_start
            print(f"\nIteration {iteration} complete in {elapsed:.1f}s")
            print(f"Checkpoint saved to {self.args.checkpoint}/latest.pt")
