
"""
tafis.model

This module implements a **GPU-capable Takagi–Sugeno (TSK) ANFIS regression model**
using PyTorch.

Design goals:
  - Fixed number of fuzzy rules (user-controlled), avoiding rule explosion.
  - Gaussian membership functions (differentiable, stable).
  - TSK (first-order) consequents: linear regression per rule.
  - Fully compatible with GPU training via standard PyTorch optimizers.
  - Exposes internal parameters so rules can be extracted later.

Model structure (conceptual):
  For each rule i:
    IF x1 is A_i1 AND x2 is A_i2 AND ... AND xD is A_iD
    THEN y_i = w_i0 + sum_j w_ij * x_j

  Final output:
    y = sum_i (normalized_firing_i * y_i)

Notes:
  - We intentionally do NOT implement hybrid LSE + GD here.
    Pure gradient-based training is simpler, stable on GPU,
    and easier to integrate with modern pipelines.
  - This model is not a black box: all parameters map cleanly to fuzzy rules.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianMF(nn.Module):
    """
    Gaussian membership function:

      mu(x) = exp( - (x - c)^2 / (2 * sigma^2) )

    Parameters (learnable):
      - c     : center
      - sigma : width (constrained to be positive)
    """

    def __init__(self, num_rules: int, num_features: int):
        super().__init__()
        # Centers: (R, D)
        self.centers = nn.Parameter(torch.empty(num_rules, num_features))
        # Log-sigma: (R, D) -> ensures sigma > 0
        self.log_sigmas = nn.Parameter(torch.empty(num_rules, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        # Centers initialized near 0.5 assuming features are in [0, 1]
        nn.init.uniform_(self.centers, 0.25, 0.75)
        # Sigmas initialized moderately wide
        nn.init.constant_(self.log_sigmas, math.log(0.2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: (N, D) input features

        Returns:
          mu: (N, R, D) membership degrees per rule per feature
        """
        # x -> (N, 1, D)
        x_exp = x.unsqueeze(1)
        # centers, sigmas -> (1, R, D)
        c = self.centers.unsqueeze(0)
        sigma = torch.exp(self.log_sigmas).unsqueeze(0)

        # Gaussian membership
        mu = torch.exp(-((x_exp - c) ** 2) / (2.0 * sigma ** 2 + 1e-8))
        return mu


class TSKConsequent(nn.Module):
    """
    First-order TSK consequent:

      y_i = b_i + sum_j w_ij * x_j

    Parameters:
      - weight: (R, D)
      - bias  : (R,)
    """

    def __init__(self, num_rules: int, num_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_rules, num_features))
        self.bias = nn.Parameter(torch.zeros(num_rules))

        # Small random init to break symmetry
        nn.init.normal_(self.weight, mean=0.0, std=0.05)
        nn.init.normal_(self.bias, mean=0.0, std=0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: (N, D)

        Returns:
          y_rule: (N, R) output of each rule before aggregation
        """
        # (N, D) @ (D, R) -> (N, R)
        y = x @ self.weight.t() + self.bias
        return y


class ANFISRegressor(nn.Module):
    """
    Complete ANFIS regression model.

    Forward pass:
      1) Compute Gaussian memberships per feature per rule
      2) Aggregate memberships across features -> firing strength per rule
      3) Normalize firing strengths
      4) Compute TSK consequent outputs
      5) Weighted sum to produce final regression output
    """

    def __init__(
        self,
        num_features: int,
        num_rules: int,
        firing_eps: float = 1e-8,
    ):
        """
        Args:
          num_features: D
          num_rules: R (user-controlled)
          firing_eps: numerical stability constant
        """
        super().__init__()
        self.num_features = num_features
        self.num_rules = num_rules
        self.firing_eps = firing_eps

        self.mf = GaussianMF(num_rules=num_rules, num_features=num_features)
        self.consequent = TSKConsequent(num_rules=num_rules, num_features=num_features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x: (N, D)

        Returns:
          y_pred: (N,)
          firing: (N, R) normalized firing strengths (useful for rule extraction)
        """
        # Step 1: membership degrees (N, R, D)
        mu = self.mf(x)

        # Step 2: firing strength per rule
        # Product across feature dimension (AND operator)
        firing = torch.prod(mu, dim=2)  # (N, R)

        # Step 3: normalize firing strengths
        firing_sum = firing.sum(dim=1, keepdim=True)
        firing_norm = firing / (firing_sum + self.firing_eps)

        # Step 4: TSK consequent outputs per rule
        y_rule = self.consequent(x)  # (N, R)

        # Step 5: weighted sum
        y_pred = torch.sum(firing_norm * y_rule, dim=1)  # (N,)

        return y_pred, firing_norm

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience wrapper for inference-only usage.
        """
        y, _ = self.forward(x)
        return y

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}, num_rules={self.num_rules}"
