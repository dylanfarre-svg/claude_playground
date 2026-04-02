"""
Neural network for Connect5 AlphaZero.

Architecture:
  - Input: (3, 10, 10) — current player pieces, opponent pieces, valid moves mask
  - 4x residual conv blocks (128 filters)
  - Policy head: softmax over 10 columns
  - Value head: tanh scalar in [-1, 1]

Sized to train overnight on M4 MPS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class Connect5Net(nn.Module):
    def __init__(self, rows=10, cols=10, action_size=10, num_channels=128, num_res_blocks=4):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.action_size = action_size

        # Input: 3 planes (player, opponent, valid mask)
        self.input_conv = nn.Sequential(
            nn.Conv2d(3, num_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )

        self.res_blocks = nn.Sequential(*[ResBlock(num_channels) for _ in range(num_res_blocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * rows * cols, action_size)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(rows * cols, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: (batch, 3, rows, cols)
        x = self.input_conv(x)
        x = self.res_blocks(x)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        p = F.log_softmax(p, dim=1)

        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v


class NNetWrapper:
    """Wraps Connect5Net with training and inference helpers."""

    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.device = torch.device(args.device)

        self.nnet = Connect5Net(
            rows=game.rows,
            cols=game.cols,
            action_size=game.get_action_size(),
            num_channels=args.num_channels,
            num_res_blocks=args.num_res_blocks,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.nnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.lr_step, gamma=args.lr_gamma
        )

    def _board_to_tensor(self, board, valid_moves):
        """Convert numpy board to (1, 3, rows, cols) tensor."""
        player_plane = (board == 1).astype("float32")
        opp_plane = (board == -1).astype("float32")
        valid_plane = np.broadcast_to(valid_moves, (self.game.rows, self.game.cols)).astype("float32")
        x = torch.from_numpy(
            np.stack([player_plane, opp_plane, valid_plane])
        ).unsqueeze(0).to(self.device)
        return x

    def predict(self, board):
        """Returns (policy array, value scalar) for a single board state."""
        import numpy as np
        valid = self.game.get_valid_moves(board)
        self.nnet.eval()
        with torch.no_grad():
            x = self._board_to_tensor(board, valid)
            log_pi, v = self.nnet(x)
        pi = torch.exp(log_pi).cpu().numpy()[0]
        # Mask invalid moves
        pi = pi * valid
        s = pi.sum()
        if s > 0:
            pi /= s
        else:
            # uniform over valid
            pi = valid / valid.sum()
        return pi, v.item()

    def train(self, examples):
        """
        examples: list of (board, pi, v) tuples
        Returns (policy_loss, value_loss) averages.
        """
        import numpy as np
        self.nnet.train()
        total_pi_loss = 0.0
        total_v_loss = 0.0
        batch_size = self.args.batch_size
        np.random.shuffle(examples)

        for i in range(0, len(examples), batch_size):
            batch = examples[i: i + batch_size]
            boards, pis, vs = zip(*batch)

            boards_t = []
            for b in boards:
                valid = self.game.get_valid_moves(b)
                boards_t.append(self._board_to_tensor(b, valid).squeeze(0))
            boards_t = torch.stack(boards_t).to(self.device)
            pis_t = torch.tensor(np.array(pis), dtype=torch.float32).to(self.device)
            vs_t = torch.tensor(np.array(vs), dtype=torch.float32).unsqueeze(1).to(self.device)

            log_pi, v = self.nnet(boards_t)

            pi_loss = -(pis_t * log_pi).sum(dim=1).mean()
            v_loss = F.mse_loss(v, vs_t)
            loss = pi_loss + v_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), 1.0)
            self.optimizer.step()

            total_pi_loss += pi_loss.item()
            total_v_loss += v_loss.item()

        self.scheduler.step()
        n = max(1, len(examples) // batch_size)
        return total_pi_loss / n, total_v_loss / n

    def save_checkpoint(self, folder, filename):
        import os
        os.makedirs(folder, exist_ok=True)
        torch.save({
            "state_dict": self.nnet.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, os.path.join(folder, filename))

    def load_checkpoint(self, folder, filename):
        path = os.path.join(folder, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.nnet.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


import numpy as np
import os
import torch.nn.functional as F
