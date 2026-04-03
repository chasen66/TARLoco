#  Copyright 2025 University of Manchester, Amr Mousa
#  SPDX-License-Identifier: CC-BY-SA-4.0


from dataclasses import dataclass, field
from typing import List, Type

import gymnasium as gym

from exts.tarloco.envs import wrappers
from exts.tarloco.learning import runners

from . import agents, algorithms, envs

# Define a dataclass to hold the configuration for each task


@dataclass
class TaskConfig:
    env_cfg_entry_point: Type
    rsl_rl_cfg_entry_point: Type
    agent_cfg: Type = algorithms.RslRlOnPolicyRunnerCfg
    runner: Type = runners.OnPolicyRunner
    env_wrappers: List[Type] = field(default_factory=lambda: [wrappers.RslRlVecEnvWrapper])


# Define the registry

registry = {
    # ------
    # TAR
    # ------
    "go2-train-tar-rnn-rough": TaskConfig(
        env_cfg_entry_point=envs.TarGo2LocomotionVelocityRoughEnvCfg,
        rsl_rl_cfg_entry_point=agents.Go2RoughRnnTarRunnerCfg,
    ),
    "go2-eval-tar-rnn-rough": TaskConfig(
        env_cfg_entry_point=envs.TarGo2LocomotionVelocityRoughEnvEvalCfg,
        rsl_rl_cfg_entry_point=agents.Go2RoughRnnTarRunnerCfg,
    ),
    # ------
    # SLR
    # ------
    "go2-train-slr-rough": TaskConfig(
        env_cfg_entry_point=envs.SlrGo2LocomotionVelocityRoughEnvCfg,
        rsl_rl_cfg_entry_point=agents.Go2RoughPpoSlrRunnerCfg,
    ),
    "go2-eval-slr-rough": TaskConfig(
        env_cfg_entry_point=envs.SlrGo2LocomotionVelocityRoughEnvEvalCfg,
        rsl_rl_cfg_entry_point=agents.Go2RoughPpoSlrRunnerCfg,
    ),
    # --------
    # HIM
    # --------
    "go2-train-him-rough": TaskConfig(
        env_cfg_entry_point=envs.HimGo2LocomotionVelocityRoughEnvCfg,
        rsl_rl_cfg_entry_point=agents.Go2PpoHimRunnerCfg,
    ),
    "go2-eval-him-rough": TaskConfig(
        env_cfg_entry_point=envs.HimGo2LocomotionVelocityRoughEnvEvalCfg,
        rsl_rl_cfg_entry_point=agents.Go2PpoHimRunnerCfg,
    ),
    # ------
    # Teacher
    # ------
    # Plain teacher configuration: Direct feeding to the actor and critic without using an encoder
    "go2-train-teacher-rough": TaskConfig(
        env_cfg_entry_point=envs.TeacherGo2LocomotionVelocityRoughEnvCfg,
        rsl_rl_cfg_entry_point=agents.Go2RoughPpoRunnerCfg,
    ),
    "go2-eval-teacher-rough": TaskConfig(
        env_cfg_entry_point=envs.TeacherGo2LocomotionVelocityRoughEnvEvalCfg,
        rsl_rl_cfg_entry_point=agents.Go2RoughPpoRunnerCfg,
    ),
    # Teacher with MLP privileged encoder that concatenates the latents to one-step proprioceptive observations
    "go2-train-teacher-encoder-rough": TaskConfig(
        env_cfg_entry_point=envs.TeacherGo2LocomotionVelocityRoughEnvCfg,
        rsl_rl_cfg_entry_point=agents.Go2RoughPpoExpertRunnerCfg,
    ),
    "go2-eval-teacher-encoder-rough": TaskConfig(
        env_cfg_entry_point=envs.TeacherGo2LocomotionVelocityRoughEnvEvalCfg,
        rsl_rl_cfg_entry_point=agents.Go2RoughPpoExpertRunnerCfg,
    ),
    # Teacher with RNN privileged encoder that concatenates the latents to one-step proprioceptive observations
    "go2-train-teacher-rnn-rough": TaskConfig(
        env_cfg_entry_point=envs.TeacherGo2LocomotionVelocityRoughEnvCfg,
        rsl_rl_cfg_entry_point=agents.Go2RoughRnnRunnerCfg,
    ),
    "go2-eval-teacher-rnn-rough": TaskConfig(
        env_cfg_entry_point=envs.TeacherGo2LocomotionVelocityRoughEnvEvalCfg,
        rsl_rl_cfg_entry_point=agents.Go2RoughRnnRunnerCfg,
    ),

    # ------------------------------ Ablation Studies ------------------------------
    # TAR replacing RNN encoder with 10-steps MLP
    "go2-train-tar-mlp-rough": TaskConfig(
        env_cfg_entry_point=envs.TarMlpGo2LocomotionVelocityRoughEnvCfg,
        rsl_rl_cfg_entry_point=agents.Go2RoughPpoTarRunnerCfg,
    ),
    "go2-eval-tar-mlp-rough": TaskConfig(
        env_cfg_entry_point=envs.TarMlpGo2LocomotionVelocityRoughEnvEvalCfg,
        rsl_rl_cfg_entry_point=agents.Go2RoughPpoTarRunnerCfg,
    ),
    # TAR replacing RNN encoder with TCN
    "go2-train-tar-tcn-rough": TaskConfig(
        env_cfg_entry_point=envs.TarTcnGo2LocomotionVelocityRoughEnvCfg,
        rsl_rl_cfg_entry_point=agents.Go2RoughTcnTarRunnerCfg,
    ),
    "go2-eval-tar-tcn-rough": TaskConfig(
        env_cfg_entry_point=envs.TarTcnGo2LocomotionVelocityRoughEnvEvalCfg,
        rsl_rl_cfg_entry_point=agents.Go2RoughTcnTarRunnerCfg,
    ),
    # TAR without privileged information
    "go2-train-tar-rnn-no-priv-rough": TaskConfig(
        env_cfg_entry_point=envs.TarRnnNoPrivGo2LocomotionVelocityRoughEnvCfg,
        rsl_rl_cfg_entry_point=agents.Go2RoughRnnTarNoPrivRunnerCfg,
    ),
    "go2-eval-tar-rnn-no-priv-rough": TaskConfig(
        env_cfg_entry_point=envs.TarRnnNoPrivGo2LocomotionVelocityRoughEnvEvalCfg,
        rsl_rl_cfg_entry_point=agents.Go2RoughRnnTarNoPrivRunnerCfg,
    ),
    # TAR without privileged information and velocity estimation
    "go2-train-tar-rnn-no-priv-no-vel-rough": TaskConfig(
        env_cfg_entry_point=envs.TarRnnNoPrivGo2LocomotionVelocityRoughEnvCfg,
        rsl_rl_cfg_entry_point=agents.Go2RoughRnnTarNoPrivNoVelRunnerCfg,
    ),
    "go2-eval-tar-rnn-no-priv-no-vel-rough": TaskConfig(
        env_cfg_entry_point=envs.TarRnnNoPrivGo2LocomotionVelocityRoughEnvEvalCfg,
        rsl_rl_cfg_entry_point=agents.Go2RoughRnnTarNoPrivNoVelRunnerCfg,
    ),
    # TAR replacing RNN encoder with 10-steps MLP, without privileged information
    "go2-train-tar-mlp-no-priv-rough": TaskConfig(
        env_cfg_entry_point=envs.TarMlpNoPrivGo2LocomotionVelocityRoughEnvCfg,
        rsl_rl_cfg_entry_point=agents.Go2RoughPpoTarNoPrivRunnerCfg,
    ),
    "go2-eval-tar-mlp-no-priv-rough": TaskConfig(
        env_cfg_entry_point=envs.TarMlpNoPrivGo2LocomotionVelocityRoughEnvEvalCfg,
        rsl_rl_cfg_entry_point=agents.Go2RoughPpoTarNoPrivRunnerCfg,
    ),
    # TAR replacing RNN encoder with 10-steps MLP, without privileged information and velocity estimation
    "go2-train-tar-mlp-no-priv-no-vel-rough": TaskConfig(
        env_cfg_entry_point=envs.TarMlpNoPrivGo2LocomotionVelocityRoughEnvCfg,
        rsl_rl_cfg_entry_point=agents.Go2RoughPpoTarNoPrivNoVelRunnerCfg,
    ),
    "go2-eval-tar-mlp-no-priv-no-vel-rough": TaskConfig(
        env_cfg_entry_point=envs.TarMlpNoPrivGo2LocomotionVelocityRoughEnvEvalCfg,
        rsl_rl_cfg_entry_point=agents.Go2RoughPpoTarNoPrivNoVelRunnerCfg,
    ),
}


# Register each environment
for env_id, config in registry.items():
    gym.register(
        id=env_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": config.env_cfg_entry_point,
            "rsl_rl_cfg_entry_point": config.rsl_rl_cfg_entry_point,
        },
    )

__all__ = ["registry"]
