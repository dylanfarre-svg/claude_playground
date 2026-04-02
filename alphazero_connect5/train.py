#!/usr/bin/env python3
"""
Entry point for AlphaZero Connect5 training.

Usage:
  python train.py
  python train.py --iters 200 --eps 100 --sims 200
"""

import argparse
import torch

from args import args
from game import Connect5Game
from model import NNetWrapper
from trainer import Trainer


def parse_cli():
    parser = argparse.ArgumentParser(description="AlphaZero Connect5 Training")
    parser.add_argument("--iters", type=int, help="Number of training iterations")
    parser.add_argument("--eps", type=int, help="Self-play episodes per iteration")
    parser.add_argument("--sims", type=int, help="MCTS simulations per move")
    parser.add_argument("--device", type=str, choices=["mps", "cpu", "cuda"], help="Device override")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint directory")
    return parser.parse_args()


def main():
    cli = parse_cli()
    if cli.iters:
        args.numIters = cli.iters
    if cli.eps:
        args.numEps = cli.eps
    if cli.sims:
        args.numMCTSSims = cli.sims
    if cli.device:
        args.device = cli.device
    if cli.checkpoint:
        args.checkpoint = cli.checkpoint

    print("=" * 60)
    print("AlphaZero Connect5 (10x10, 5-in-a-row)")
    print("=" * 60)
    print(f"Device:          {args.device}")
    print(f"Iterations:      {args.numIters}")
    print(f"Episodes/iter:   {args.numEps}")
    print(f"MCTS sims/move:  {args.numMCTSSims}")
    print(f"Network:         {args.num_channels}ch x {args.num_res_blocks} res blocks")
    print(f"Batch size:      {args.batch_size}")
    print(f"Checkpoint dir:  {args.checkpoint}")

    if args.device == "mps":
        print(f"MPS available:   {torch.backends.mps.is_available()}")

    print()

    game = Connect5Game()
    nnet = NNetWrapper(game, args)

    # Print param count
    total_params = sum(p.numel() for p in nnet.nnet.parameters())
    print(f"Network parameters: {total_params:,}")
    print()

    trainer = Trainer(game, nnet, args)
    trainer.learn()


if __name__ == "__main__":
    main()
