# Copyright (c) 2025, TARLoco Contributors
# SPDX-License-Identifier: BSD-3-Clause
#
# Standalone TARLoco policy inference wrapper for MuJoCo sim2sim.
# Zero dependency on IsaacLab or RSL-RL – only PyTorch + NumPy.

"""
TARLoco Policy Inference Wrapper
==================================
支持策略类型: ActorCriticTarRnn (LSTM 编码器 + MLP Actor)

推理流程
---------
1. 从 checkpoint 加载 LSTM 编码器、vel_estimator、actor MLP 及 obs 归一化统计量
2. 维护 num_hist 步的原始观测滚动缓冲区
3. 每步:
   a. 将 num_hist 步原始 obs 展平 → 归一化 → reshape [num_hist, obs_dim]
   b. 取最新一步 prop 送入 LSTM 编码器 → latent z
   c. vel_estimator([z, hist_short.flatten()]) → 估计速度 vel
   d. actor([z, prop, vel]) → 12 维关节位置 action
4. target_pos = default_pos + action * action_scale

观测维度 (45 维, 与 TarGo2 训练环境一致):
  base_ang_vel      3
  projected_gravity 3
  velocity_commands 3
  joint_pos_rel     12   (joint_pos - default_pos)
  joint_vel         12
  last_action       12
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# MLP factory: rebuild nn.Sequential from state_dict keys
# ─────────────────────────────────────────────────────────────────────────────

def _load_sequential_from_sd(sd: dict, prefix: str, act: nn.Module) -> nn.Sequential:
    """Reconstruct nn.Sequential of Linear layers from a state_dict.

    Linear layer indices are even (0, 2, 4, …). An activation is inserted
    between consecutive Linear layers.
    """
    layers: list[nn.Module] = []
    i = 0
    while f"{prefix}.{i}.weight" in sd:
        w = sd[f"{prefix}.{i}.weight"]
        b = sd[f"{prefix}.{i}.bias"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight = nn.Parameter(w.clone())
        lin.bias = nn.Parameter(b.clone())
        layers.append(lin)
        if f"{prefix}.{i + 2}.weight" in sd:
            layers.append(type(act)())
        i += 2
    if not layers:
        raise KeyError(f"No layers found with prefix '{prefix}' in state_dict")
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────────────
# Main inference class
# ─────────────────────────────────────────────────────────────────────────────

class TarRnnPolicy:
    """Standalone inference wrapper for ActorCriticTarRnn.

    Parameters
    ----------
    ckpt_path : str
        Path to the RSL-RL .pt checkpoint.
    action_scale : float
        Multiplier matching ``JointPositionActionCfg.scale`` at training time.
    device : str
        ``'cpu'`` or ``'cuda'``.
    """

    def __init__(
        self,
        ckpt_path: str,
        action_scale: float = 0.25,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.action_scale = action_scale

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd: dict = ckpt["model_state_dict"]

        # ── Infer network hyper-parameters from weight shapes ────────────
        self.obs_dim: int = sd["encoder.rnn.weight_ih_l0"].shape[1]
        self.rnn_hidden: int = sd["encoder.rnn.weight_hh_l0"].shape[1]
        self.latent_dim: int = sd["encoder.feature_extractor.weight"].shape[0]
        actor_max_idx = max(
            int(k.split(".")[2])
            for k in sd
            if k.startswith("actor.mlp.") and k.endswith(".weight")
        )
        self.action_dim: int = sd[f"actor.mlp.{actor_max_idx}.weight"].shape[0]

        obs_norm_sd: dict = ckpt["obs_norm_state_dict"]
        flat_dim: int = obs_norm_sd["_mean"].numel()
        self.num_hist: int = flat_dim // self.obs_dim

        print(
            f"[Policy] obs_dim={self.obs_dim}, num_hist={self.num_hist}, "
            f"latent_dim={self.latent_dim}, rnn_hidden={self.rnn_hidden}, "
            f"action_dim={self.action_dim}"
        )

        # ── Observation normalisation stats ──────────────────────────────
        self._obs_mean = obs_norm_sd["_mean"].view(-1).float().to(device)
        self._obs_std = obs_norm_sd["_std"].view(-1).float().to(device)

        # ── LSTM encoder ─────────────────────────────────────────────────
        self._lstm = nn.LSTM(
            input_size=self.obs_dim,
            hidden_size=self.rnn_hidden,
            num_layers=1,
            batch_first=False,
        )
        self._lstm.weight_ih_l0 = nn.Parameter(sd["encoder.rnn.weight_ih_l0"].clone())
        self._lstm.weight_hh_l0 = nn.Parameter(sd["encoder.rnn.weight_hh_l0"].clone())
        self._lstm.bias_ih_l0 = nn.Parameter(sd["encoder.rnn.bias_ih_l0"].clone())
        self._lstm.bias_hh_l0 = nn.Parameter(sd["encoder.rnn.bias_hh_l0"].clone())
        self._lstm.eval().to(device)

        # ── Feature extractor (LSTM hidden → latent z) ───────────────────
        self._feat_ext = nn.Linear(self.rnn_hidden, self.latent_dim)
        self._feat_ext.weight = nn.Parameter(sd["encoder.feature_extractor.weight"].clone())
        self._feat_ext.bias = nn.Parameter(sd["encoder.feature_extractor.bias"].clone())
        self._feat_ext.eval().to(device)

        # ── Velocity estimator MLP ───────────────────────────────────────
        self._vel_estimator = _load_sequential_from_sd(sd, "vel_estimator", nn.ELU())
        self._vel_estimator.eval().to(device)

        # ── Actor MLP ────────────────────────────────────────────────────
        self._actor = _load_sequential_from_sd(sd, "actor.mlp", nn.ELU())
        self._actor.eval().to(device)

        # ── Rolling history buffer ───────────────────────────────────────
        self._raw_buf: deque = deque(
            [np.zeros(self.obs_dim, dtype=np.float32)] * self.num_hist,
            maxlen=self.num_hist,
        )
        self._h: Optional[torch.Tensor] = None
        self._c: Optional[torch.Tensor] = None

    # --------------------------------------------------------------------- #

    def reset(self, initial_obs: Optional[np.ndarray] = None) -> None:
        """Reset observation history and LSTM hidden state.

        Parameters
        ----------
        initial_obs : np.ndarray, shape (obs_dim,), optional
            If provided, pre-fill the entire history buffer with this obs.
            This avoids starting with a zero-filled buffer where the projected
            gravity term is 0 instead of [0, 0, -1], which would cause the
            policy to output unstable actions from the very first step.
            If None, falls back to a "safe" standing obs: zeros except
            gravity = [0, 0, -1] at indices 3:6.
        """
        if initial_obs is not None:
            fill = initial_obs.astype(np.float32).copy()
        else:
            fill = np.zeros(self.obs_dim, dtype=np.float32)
            # At minimum, set correct projected gravity so policy knows it's upright
            if self.obs_dim >= 6:
                fill[3] = 0.0   # grav_x
                fill[4] = 0.0   # grav_y
                fill[5] = -1.0  # grav_z (pointing down in body frame = standing upright)
        self._raw_buf.clear()
        self._raw_buf.extend([fill.copy() for _ in range(self.num_hist)])
        self._h = None
        self._c = None

    # --------------------------------------------------------------------- #

    def _normalize(self, flat_obs: torch.Tensor) -> torch.Tensor:
        """Normalise flattened history vector (eps=0.01 matches EmpiricalNormalization)."""
        return (flat_obs - self._obs_mean) / (self._obs_std + 0.01)

    # --------------------------------------------------------------------- #

    @torch.no_grad()
    def step(self, raw_obs: np.ndarray) -> np.ndarray:
        """Run a single inference step.

        Parameters
        ----------
        raw_obs : np.ndarray, shape ``(obs_dim,)``
            Order must match training env:
            ``[ang_vel(3), proj_grav(3), cmd(3), jpos_rel(12), jvel(12), last_action(12)]``

        Returns
        -------
        np.ndarray, shape ``(action_dim,)``
            Joint position *offsets* (before adding default joint pos).
            ``target_pos = default_joint_pos + action * action_scale``
        """
        self._raw_buf.append(raw_obs.astype(np.float32))

        history_np = np.stack(list(self._raw_buf), axis=0)  # [num_hist, obs_dim]
        flat_t = torch.tensor(history_np.flatten(), dtype=torch.float32, device=self.device)
        flat_norm = self._normalize(flat_t)
        hist_norm = flat_norm.view(self.num_hist, self.obs_dim)

        prop = hist_norm[-1]  # latest normalised obs

        # LSTM encoder
        prop_in = prop.unsqueeze(0).unsqueeze(0)  # [1, 1, obs_dim]
        hx = (self._h, self._c) if self._h is not None else None
        lstm_out, (new_h, new_c) = self._lstm(prop_in, hx)
        self._h, self._c = new_h, new_c
        z = self._feat_ext(lstm_out.squeeze(0).squeeze(0))  # [latent_dim]

        # Velocity estimator
        hist_flat = hist_norm.flatten()
        vel = self._vel_estimator(torch.cat([z, hist_flat]))

        # Actor
        action = self._actor(torch.cat([z.detach(), prop, vel.detach()]))

        return action.cpu().numpy()
