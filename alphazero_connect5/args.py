"""
Training hyperparameters, tuned for M4 Mac mini with MPS backend.
"""

import torch


class Args:
    # Device
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"

    # Self-play
    numIters: int = 100          # training iterations
    numEps: int = 50             # self-play episodes per iteration
    tempThreshold: int = 15      # use temp=1 for first N moves, then greedy
    maxlenOfQueue: int = 20      # keep last N iterations of self-play data

    # MCTS
    numMCTSSims: int = 100       # simulations per MCTS call
    cpuct: float = 1.5           # exploration constant

    # Network architecture
    num_channels: int = 128
    num_res_blocks: int = 4

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_step: int = 10            # decay LR every N iterations
    lr_gamma: float = 0.5
    batch_size: int = 256

    # Checkpointing
    checkpoint: str = "checkpoints"


args = Args()
