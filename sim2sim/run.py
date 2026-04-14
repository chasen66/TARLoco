# Copyright (c) 2025, TARLoco Contributors
# SPDX-License-Identifier: BSD-3-Clause
#
# MuJoCo sim-to-sim verification for TARLoco Go2 locomotion policies.
# Loads a trained checkpoint and runs explicit PD + T-N curve torque control.

"""
MuJoCo Sim2Sim Runner
=====================
用法:
    python -m sim2sim.run --load_run <log_dir> [--checkpoint <ckpt_name>]

快捷键:
    W / S  — 前进 / 后退
    A / D  — 左转 / 右转
    Q / E  — 左平移 / 右平移
    Space  — 零速 (停止)
    R      — 重置
    ESC    — 退出
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np

from sim2sim.policy import TarRnnPolicy

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Go2 joint names — IsaacLab policy order (FL, FR, RL, RR leg-by-leg)
# This is the order used in observations, actions, and default_joint_pos.
JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

# Unitree SDK2 / real-robot motor order: FR, FL, RR, RL (hardware motor IDs 0-11)
# Mapping from IsaacLab policy index → real robot motor index
POLICY_TO_REAL = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
# Mapping from real robot motor index → IsaacLab policy index
REAL_TO_POLICY = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

# Default joint positions (matching training env UNITREE_GO2_CFG.init_state)
DEFAULT_JOINT_POS = np.array([
    0.1,  0.8, -1.5,   # FL
   -0.1,  0.8, -1.5,   # FR
    0.1,  1.0, -1.5,   # RL
   -0.1,  1.0, -1.5,   # RR
], dtype=np.float64)

# PD gains — must match training (stiffness=25, damping=0.5)
KP = 25.0
KD = 0.5

# Simulation timing
SIM_DT = 0.005          # 200 Hz physics
DECIMATION = 4          # policy runs at 50 Hz
ACTION_SCALE = 0.25

# DCMotor actuator parameters (matching training env.yaml for this checkpoint)
# DCMotorCfg: effort_limit=23.5, saturation_effort=23.5, velocity_limit=30.0
EFFORT_LIMIT = 23.5
SATURATION_EFFORT = 23.5
VELOCITY_LIMIT = 30.0

# Soft joint position limit factor (matching training env)
# IsaacLab applies soft_joint_pos_limit_factor=0.9 to clip action targets.
SOFT_JOINT_LIMIT_FACTOR = 0.9

# Initial base height for reset
# Computed from MJCF kinematics: with training joint defaults, foot sphere
# centers are ~0.143-0.153m above floor when base is at 0.445m (mj_resetData).
# Lower base by 0.131m (rear foot gap) so all 4 feet are near the ground.
# Warmup phase settles any residual gap via gravity + PD.
INIT_HEIGHT = 0.325  # metres (verified: front feet ~0mm, rear ~0.011m above floor)

# MuJoCo joint armature (rotational inertia, kg·m²).
# The official go2.xml uses 0.01. Keep this value for numerical stability.
MJ_ARMATURE = 0.01

# PD-only warmup steps before activating policy.
# With LSTM warmup active, PD warmup is unnecessary and may hurt stability
# by creating obs distribution shift. Set to 0 when using LSTM warmup.
WARMUP_STEPS = 0

# LSTM warmup: run policy inference N times with standing obs before simulation.
# The LSTM starts with zero hidden state which produces unreasonably large
# actions (max ~3.2). Pre-running with standing obs lets the hidden state
# converge to steady-state (actions drop to ~1.0).
LSTM_WARMUP_STEPS = 100

# Action EMA (exponential moving average) filter coefficient.
# Smooths policy outputs to reduce high-frequency oscillations caused by
# PhysX↔MuJoCo contact model differences. Lower α = more smoothing.
# Set to 1.0 to disable (pass-through).
ACTION_EMA_ALPHA = 0.2

# Fall detection
FALL_HEIGHT = 0.12  # metres

# Velocity command limits
VX_RANGE = (-1.0, 1.0)
VY_RANGE = (-1.0, 1.0)
WZ_RANGE = (-1.0, 1.0)
CMD_STEP = 0.1


# ─────────────────────────────────────────────────────────────────────────────
# DCMotor Torque Saturation (matching training actuator)
# ─────────────────────────────────────────────────────────────────────────────

def clip_effort_dcmotor(torques: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """Clip torques using IsaacLab's DCMotor velocity-dependent saturation model.

    Matches ``DCMotor._clip_effort`` from training:
      max_effort = sat * (1 - vel/vel_limit),  clipped to [0, effort_limit]
      min_effort = sat * (-1 - vel/vel_limit),  clipped to [-effort_limit, 0]
      output = clip(torque, min_effort, max_effort)
    """
    max_effort = SATURATION_EFFORT * (1.0 - velocities / VELOCITY_LIMIT)
    max_effort = np.clip(max_effort, 0.0, EFFORT_LIMIT)

    min_effort = SATURATION_EFFORT * (-1.0 - velocities / VELOCITY_LIMIT)
    min_effort = np.clip(min_effort, -EFFORT_LIMIT, 0.0)

    return np.clip(torques, min_effort, max_effort)


# ─────────────────────────────────────────────────────────────────────────────
# PD Control
# ─────────────────────────────────────────────────────────────────────────────

def pd_control(
    target_pos: np.ndarray,
    current_pos: np.ndarray,
    current_vel: np.ndarray,
    kp: float = KP,
    kd: float = KD,
) -> np.ndarray:
    """Explicit PD torque computation."""
    return kp * (target_pos - current_pos) - kd * current_vel


# ─────────────────────────────────────────────────────────────────────────────
# Gravity Vector Helper
# ─────────────────────────────────────────────────────────────────────────────

def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector *v* by the inverse of quaternion *q* (scalar-first: w,x,y,z)."""
    w, x, y, z = q
    # q_inv = conjugate for unit quaternion
    t = 2.0 * np.cross(np.array([-x, -y, -z]), v)
    return v + w * t + np.cross(np.array([-x, -y, -z]), t)


def projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    """Project gravity [0,0,-1] into body frame using quaternion (w,x,y,z)."""
    return quat_rotate_inverse(quat_wxyz, np.array([0.0, 0.0, -1.0]))


# ─────────────────────────────────────────────────────────────────────────────
# MJCF Locator
# ─────────────────────────────────────────────────────────────────────────────

def find_mjcf() -> str:
    """Search for go2 scene XML in common locations.

    Prefers scene.xml over go2.xml — scene.xml contains the ground plane,
    sky box, and lights required for a functional simulation.
    go2.xml alone has no floor; the robot falls through void immediately.
    """
    # Priority: local assets first, then known external repos.
    # Always prefer scene.xml over bare go2.xml.
    candidates = [
        # Local copy (preferred — fastest to iterate on)
        Path(__file__).parent / "assets" / "scene.xml",
        Path(__file__).parent / "assets" / "go2.xml",
        # Official unitree_mujoco repo inside this project
        Path(__file__).parent.parent / "unitree_mujoco" / "unitree_robots" / "go2" / "scene.xml",
        Path(__file__).parent.parent / "unitree_mujoco" / "unitree_robots" / "go2" / "go2.xml",
        # Official unitree_mujoco repo at home (best physics parameters)
        Path.home() / "unitree_mujoco" / "unitree_robots" / "go2" / "scene.xml",
        Path.home() / "unitree_mujoco" / "unitree_robots" / "go2" / "go2.xml",
        # mujoco_menagerie
        Path.home() / "ubuntu20.04" / "dreamwk" / "mujoco_menagerie" / "unitree_go2" / "scene.xml",
        Path.home() / "ubuntu20.04" / "dreamwk" / "mujoco_menagerie" / "unitree_go2" / "go2.xml",
        Path.home() / "mujoco_menagerie" / "unitree_go2" / "scene.xml",
        Path.home() / "mujoco_menagerie" / "unitree_go2" / "go2.xml",
    ]
    for p in candidates:
        if p.is_file():
            if p.name == "go2.xml":
                print(
                    f"[Sim2Sim] WARNING: Using bare go2.xml — no floor/lights.\n"
                    f"         Consider using scene.xml instead (same directory)."
                )
            return str(p)
    raise FileNotFoundError(
        "Cannot find go2 scene XML.\n"
        "  Option A: git clone https://github.com/unitreerobotics/unitree_mujoco ~/unitree_mujoco\n"
        "  Option B: place scene.xml + go2.xml + assets/ in sim2sim/assets/"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint Locator
# ─────────────────────────────────────────────────────────────────────────────

def find_checkpoint(load_run: str, checkpoint: Optional[str] = None) -> str:
    """Locate checkpoint file from a training log directory.

    Parameters
    ----------
    load_run : str
        Path to the RSL-RL log directory (e.g. ``logs/rsl_rl/TarRnn/.../``).
        Can be absolute or relative to the project root.
    checkpoint : str, optional
        Specific checkpoint filename (e.g. ``model_5000.pt``).
        If None, automatically selects the latest ``model_*.pt``.
    """
    run_dir = Path(load_run)
    if not run_dir.is_absolute():
        run_dir = Path(__file__).parent.parent / run_dir

    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    if checkpoint is not None:
        ckpt = run_dir / checkpoint
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return str(ckpt)

    # Auto-detect latest model_*.pt
    pts = sorted(run_dir.glob("model_*.pt"))
    if not pts:
        raise FileNotFoundError(f"No model_*.pt found in {run_dir}")
    return str(pts[-1])


# ─────────────────────────────────────────────────────────────────────────────
# Main Simulation Loop
# ─────────────────────────────────────────────────────────────────────────────

class Go2Sim:
    """MuJoCo simulation wrapper for Go2 with explicit PD + DCMotor clipping."""

    def __init__(self, model_path: str, policy: TarRnnPolicy) -> None:
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.policy = policy

        # Override timestep to match training
        self.model.opt.timestep = SIM_DT

        # ── Override passive joint dynamics ────────────────────────────
        # The official Unitree go2.xml has damping=0.1, armature=0.01,
        # frictionloss=0.2 on each joint. In IsaacLab/PhysX training,
        # damping/frictionloss are zero — all damping comes from the PD
        # controller (Kd). Keep original armature=0.01 for stability.
        for i in range(self.model.njnt):
            dof = self.model.jnt_dofadr[i]
            if dof >= 0:
                self.model.dof_damping[dof] = 0.0
                self.model.dof_armature[dof] = MJ_ARMATURE
                self.model.dof_frictionloss[dof] = 0.0

        # Build joint index mapping
        self.joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
        )
        # qpos / qvel address for each joint
        self.qpos_addr = np.array([self.model.jnt_qposadr[j] for j in self.joint_ids])
        self.qvel_addr = np.array([self.model.jnt_dofadr[j] for j in self.joint_ids])

        # Base body ID for rotation matrix lookup
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

        # Actuator indices (assumed same name without _joint suffix)
        actuator_names = [n.replace("_joint", "") for n in JOINT_NAMES]
        self.act_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in actuator_names]
        )

        # ── Joint position limits for action clipping ────────────────────
        # In training, JointPositionActionCfg clips targets to
        # soft_joint_pos_limit_factor * [lower, upper].
        raw_lower = np.array([self.model.jnt_range[j, 0] for j in self.joint_ids])
        raw_upper = np.array([self.model.jnt_range[j, 1] for j in self.joint_ids])
        mid = 0.5 * (raw_lower + raw_upper)
        half = 0.5 * (raw_upper - raw_lower)
        self.joint_pos_lower = mid - SOFT_JOINT_LIMIT_FACTOR * half
        self.joint_pos_upper = mid + SOFT_JOINT_LIMIT_FACTOR * half
        print(f"[Sim2Sim] Joint soft limits (90%):")
        for i, n in enumerate(JOINT_NAMES):
            print(f"  {n:24s}  [{self.joint_pos_lower[i]:+.3f}, {self.joint_pos_upper[i]:+.3f}]")

        # Velocity commands
        self.cmd = np.zeros(3, dtype=np.float64)  # [vx, vy, wz]

        # Action buffer
        self.last_action = np.zeros(12, dtype=np.float32)
        # EMA-filtered action (for smoothing)
        self._filtered_action = np.zeros(12, dtype=np.float32)

        # Step counter
        self.step_count = 0

    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset simulation to default standing pose.

        Steps:
        1. Set base height to INIT_HEIGHT so feet are near the floor.
        2. Pre-fill policy obs buffer with standing obs (gravity=[0,0,-1]).
        3. LSTM warmup: converge hidden state before simulation starts.
        4. Optional PD-only warmup to settle robot contacts.
        """
        mujoco.mj_resetData(self.model, self.data)

        # ── 1. Set explicit base pose matching training init_state ────────
        self.data.qpos[0] = 0.0          # x
        self.data.qpos[1] = 0.0          # y
        self.data.qpos[2] = INIT_HEIGHT  # z (feet near floor)
        self.data.qpos[3] = 1.0          # quat w (identity)
        self.data.qpos[4] = 0.0          # quat x
        self.data.qpos[5] = 0.0          # quat y
        self.data.qpos[6] = 0.0          # quat z

        # ── 2. Set joint positions to training defaults ───────────────────
        for i, addr in enumerate(self.qpos_addr):
            self.data.qpos[addr] = DEFAULT_JOINT_POS[i]

        # Reset velocities and controls
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.last_action = np.zeros(12, dtype=np.float32)
        self.cmd[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # ── 3. Pre-fill policy buffer with actual standing obs ────────────
        # This ensures projected gravity = [0,0,-1] in the buffer from the start,
        # preventing the policy from seeing phantom wrong orientation history.
        standing_obs = self.get_obs()
        self.policy.reset(standing_obs)

        # ── 4. LSTM warmup: converge hidden state before simulation ───────
        # The LSTM starts with zero hidden state, producing large actions.
        # Pre-running with standing obs lets it converge to steady-state.
        if LSTM_WARMUP_STEPS > 0:
            for _ in range(LSTM_WARMUP_STEPS):
                _ = self.policy.step(standing_obs.copy())
            self.last_action[:] = 0.0
            self._filtered_action[:] = 0.0

        # ── 5. PD-only warmup: settle robot before policy activates ───────
        for _ in range(WARMUP_STEPS):
            cur_pos = self.data.qpos[self.qpos_addr]
            cur_vel = self.data.qvel[self.qvel_addr]
            torques = pd_control(DEFAULT_JOINT_POS, cur_pos, cur_vel)
            torques = clip_effort_dcmotor(torques, cur_vel)
            self.data.ctrl[self.act_ids] = torques
            mujoco.mj_step(self.model, self.data)

        self.step_count = 0

    # ------------------------------------------------------------------ #

    def get_obs(self) -> np.ndarray:
        """Construct 45-dim observation matching training env.

        Order: [ang_vel(3), proj_grav(3), cmd(3), jpos_rel(12), jvel(12), last_action(12)]
        """
        # Base angular velocity (body frame)
        # MuJoCo free joint qvel[3:6] is angular velocity in WORLD frame.
        # IsaacLab training uses root_ang_vel_b (body frame).
        # Transform: ang_vel_body = R^T @ ang_vel_world
        quat_wxyz = self.data.qpos[3:7]  # MuJoCo uses [w,x,y,z]
        ang_vel_world = self.data.qvel[3:6]
        R = self.data.xmat[self.base_body_id].reshape(3, 3)  # body-to-world rotation
        ang_vel_body = R.T @ ang_vel_world  # world → body frame

        # Projected gravity
        grav = projected_gravity(quat_wxyz)

        # Joint positions (relative to default)
        joint_pos = self.data.qpos[self.qpos_addr] - DEFAULT_JOINT_POS
        joint_vel = self.data.qvel[self.qvel_addr]

        obs = np.concatenate([
            ang_vel_body,               # 3
            grav,                        # 3
            self.cmd,                    # 3
            joint_pos,                   # 12
            joint_vel,                   # 12
            self.last_action,            # 12
        ]).astype(np.float32)

        return obs

    # ------------------------------------------------------------------ #

    def step_policy(self) -> np.ndarray:
        """Run one policy inference step, return target joint positions."""
        obs = self.get_obs()
        raw_action = self.policy.step(obs)  # [12] joint offsets

        # Apply EMA filter to smooth actions (reduces PhysX↔MuJoCo sim-gap)
        if ACTION_EMA_ALPHA < 1.0:
            self._filtered_action = (
                ACTION_EMA_ALPHA * raw_action
                + (1.0 - ACTION_EMA_ALPHA) * self._filtered_action
            )
            action = self._filtered_action
        else:
            action = raw_action

        # Store filtered action as last_action (consistent with what physics sees)
        self.last_action = action.copy()

        target_pos = DEFAULT_JOINT_POS + action * ACTION_SCALE
        # Clip to soft joint limits (matching IsaacLab training behavior)
        target_pos = np.clip(target_pos, self.joint_pos_lower, self.joint_pos_upper)
        return target_pos

    # ------------------------------------------------------------------ #

    def step_sim(self, target_pos: np.ndarray) -> None:
        """Run DECIMATION physics steps with explicit PD + DCMotor torque clipping."""
        for _ in range(DECIMATION):
            current_pos = self.data.qpos[self.qpos_addr]
            current_vel = self.data.qvel[self.qvel_addr]

            # Explicit PD → torque
            torques = pd_control(target_pos, current_pos, current_vel)

            # T-N curve clipping
            torques = clip_effort_dcmotor(torques, current_vel)

            # Apply to actuators
            self.data.ctrl[self.act_ids] = torques

            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

    # ------------------------------------------------------------------ #

    def is_fallen(self) -> bool:
        """Check if base height is below threshold."""
        return self.data.qpos[2] < FALL_HEIGHT

    # ------------------------------------------------------------------ #

    def base_height(self) -> float:
        return float(self.data.qpos[2])


# ─────────────────────────────────────────────────────────────────────────────
# Keyboard callback
# ─────────────────────────────────────────────────────────────────────────────

def make_key_callback(sim: Go2Sim):
    """Create a keyboard callback for mujoco.viewer."""

    def key_callback(keycode: int) -> None:
        # W=87 S=83 A=65 D=68 Q=81 E=69 Space=32 R=82 Esc=256
        if keycode == 87:  # W
            sim.cmd[0] = min(sim.cmd[0] + CMD_STEP, VX_RANGE[1])
        elif keycode == 83:  # S
            sim.cmd[0] = max(sim.cmd[0] - CMD_STEP, VX_RANGE[0])
        elif keycode == 81:  # Q
            sim.cmd[1] = min(sim.cmd[1] + CMD_STEP, VY_RANGE[1])
        elif keycode == 69:  # E
            sim.cmd[1] = max(sim.cmd[1] - CMD_STEP, VY_RANGE[0])
        elif keycode == 65:  # A
            sim.cmd[2] = min(sim.cmd[2] + CMD_STEP, WZ_RANGE[1])
        elif keycode == 68:  # D
            sim.cmd[2] = max(sim.cmd[2] - CMD_STEP, WZ_RANGE[0])
        elif keycode == 32:  # Space
            sim.cmd[:] = 0.0
        elif keycode == 82:  # R
            sim.reset()
            print("[Reset] Episode reset.")

    return key_callback


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TARLoco MuJoCo Sim2Sim Verification")
    parser.add_argument("--load_run", type=str, required=True,
                        help="Path to RSL-RL training log directory")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint filename (default: latest model_*.pt)")
    parser.add_argument("--mjcf", type=str, default=None,
                        help="Path to go2.xml MJCF file (default: auto-search)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Inference device (default: cpu)")
    parser.add_argument("--ema", type=float, default=None,
                        help="Action EMA filter alpha (0-1, lower=smoother). "
                             "Overrides ACTION_EMA_ALPHA constant.")
    parser.add_argument("--lstm_warmup", type=int, default=None,
                        help="LSTM warmup steps (default: 100). "
                             "Overrides LSTM_WARMUP_STEPS constant.")
    args = parser.parse_args()

    # Override global constants from CLI args
    global ACTION_EMA_ALPHA, LSTM_WARMUP_STEPS
    if args.ema is not None:
        ACTION_EMA_ALPHA = args.ema
    if args.lstm_warmup is not None:
        LSTM_WARMUP_STEPS = args.lstm_warmup

    # Locate files
    mjcf_path = args.mjcf if args.mjcf else find_mjcf()
    ckpt_path = find_checkpoint(args.load_run, args.checkpoint)
    print(f"[Sim2Sim] MJCF  : {mjcf_path}")
    print(f"[Sim2Sim] Ckpt  : {ckpt_path}")

    # Load policy
    policy = TarRnnPolicy(ckpt_path, action_scale=ACTION_SCALE, device=args.device)

    # Create simulation
    sim = Go2Sim(mjcf_path, policy)
    sim.reset()

    print("[Sim2Sim] Starting simulation...")
    print(f"  LSTM warmup: {LSTM_WARMUP_STEPS} steps | Action EMA α: {ACTION_EMA_ALPHA}")
    print("  W/S: forward/backward  |  A/D: turn left/right")
    print("  Q/E: strafe left/right  |  Space: stop  |  R: reset")

    episode = 0
    episode_steps = 0

    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=make_key_callback(sim)) as viewer:
        while viewer.is_running():
            t_start = time.time()

            # Policy step
            target_pos = sim.step_policy()

            # Simulation step (DECIMATION sub-steps)
            sim.step_sim(target_pos)
            episode_steps += 1

            # Fall detection
            if sim.is_fallen():
                episode += 1
                print(f"[Sim2Sim] Episode {episode} ended at step {episode_steps} "
                      f"(height={sim.base_height():.3f}m). Resetting...")
                sim.reset()
                episode_steps = 0

            # Sync viewer
            viewer.sync()

            # Real-time pacing
            elapsed = time.time() - t_start
            policy_dt = SIM_DT * DECIMATION
            if elapsed < policy_dt:
                time.sleep(policy_dt - elapsed)

    print("[Sim2Sim] Viewer closed.")


if __name__ == "__main__":
    main()
