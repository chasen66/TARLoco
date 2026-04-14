# Copyright (c) 2024, Unitree Robotics
# Adapted for TARLoco / IsaacLab integration.

from __future__ import annotations

import torch

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class UnitreeActuator(DelayedPDActuator):
    """Unitree actuator with torque-speed curve and friction model.

    The torque-speed curve::

            Torque Limit, N·m
                ^
    Y2──────────|
                |──────────────Y1
                |              │\\
                |              │ \\
                |              │  \\
                |              |   \\
    ------------+--------------|------> velocity: rad/s
                              X1   X2

    - Y1: Peak torque (torque and speed in the same direction)
    - Y2: Peak torque (torque and speed in the opposite direction)
    - X1: Maximum speed at full torque (T-N curve knee point)
    - X2: No-load speed
    """

    cfg: UnitreeActuatorCfg

    def __init__(self, cfg: UnitreeActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._effort_y1 = self._parse_joint_parameter(cfg.Y1, 1e9)
        self._effort_y2 = self._parse_joint_parameter(cfg.Y2, cfg.Y1)
        self._velocity_x1 = self._parse_joint_parameter(cfg.X1, 1e9)
        self._velocity_x2 = self._parse_joint_parameter(cfg.X2, 1e9)
        self._friction_static = self._parse_joint_parameter(cfg.Fs, 0.0)
        self._friction_dynamic = self._parse_joint_parameter(cfg.Fd, 0.0)
        self._activation_vel = self._parse_joint_parameter(cfg.Va, 0.01)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # save current joint vel for _clip_effort
        self._joint_vel[:] = joint_vel
        # PD + delay from parent
        control_action = super().compute(control_action, joint_pos, joint_vel)

        # apply friction model
        self.applied_effort -= (
            self._friction_static * torch.tanh(joint_vel / self._activation_vel)
            + self._friction_dynamic * joint_vel
        )

        # output only joint efforts (torque control)
        control_action.joint_positions = None
        control_action.joint_velocities = None
        control_action.joint_efforts = self.applied_effort

        return control_action

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        # determine max effort based on torque-speed direction
        same_direction = (self._joint_vel * effort) > 0
        max_effort = torch.where(same_direction, self._effort_y1, self._effort_y2)
        # apply T-N curve: linear drop-off between X1 and X2
        max_effort = torch.where(
            self._joint_vel.abs() < self._velocity_x1,
            max_effort,
            self._compute_effort_limit(max_effort),
        )
        return torch.clip(effort, -max_effort, max_effort)

    def _compute_effort_limit(self, max_effort: torch.Tensor) -> torch.Tensor:
        k = -max_effort / (self._velocity_x2 - self._velocity_x1)
        limit = k * (self._joint_vel.abs() - self._velocity_x1) + max_effort
        return limit.clip(min=0.0)


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------


@configclass
class UnitreeActuatorCfg(DelayedPDActuatorCfg):
    """Base configuration for Unitree actuators with T-N curve."""

    class_type: type = UnitreeActuator

    X1: float = 1e9
    """Maximum speed at full torque (T-N curve knee point), rad/s."""

    X2: float = 1e9
    """No-load speed, rad/s."""

    Y1: float = 1e9
    """Peak torque (same direction as speed), N·m."""

    Y2: float | None = None
    """Peak torque (opposite direction to speed), N·m.  Defaults to Y1."""

    Fs: float = 0.0
    """Static friction coefficient."""

    Fd: float = 0.0
    """Dynamic friction coefficient."""

    Va: float = 0.01
    """Velocity at which friction is fully activated, rad/s."""


@configclass
class UnitreeActuatorCfg_Go2HV(UnitreeActuatorCfg):
    """Go2 high-voltage motor (GO-M8010-6)."""

    X1: float = 13.5
    X2: float = 30.0
    Y1: float = 20.2
    Y2: float = 23.4
