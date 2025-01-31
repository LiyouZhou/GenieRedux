#!/usr/bin/env python3

from rl.reward_from_latents import RewardFromLatents
from matplotlib import pyplot as plt
from hydra import compose, initialize
from omegaconf import DictConfig
from matplotlib import pyplot as plt
from models import construct_model
from pathlib import Path
import numpy as np
import torch
from rl.gen_push_t_env import GenerativePushT
from stable_baselines3 import PPO

base_dir = "/home/liyouzhou/study/GenieRedux"

with initialize(version_base=None, config_path=f"configs", job_name="test_app"):
    cfg = compose(config_name="config/genie_redux_guided.yaml")

cfg = DictConfig(cfg.config)
cfg.tokenizer_fpath = (
    f"{base_dir}/checkpoints/tokenizer/tokenizer_push_t_fp32/model-50000.pt"
)

pt_dataset_path = f"{base_dir}/datasets/pusht_real/real_pusht_20230105"
image_size = (64, 64)
num_frames = 16

model = construct_model(
    config=cfg,
)
model.load_state_dict(
    torch.load(
        "checkpoints/genie_redux_guided/genie_redux_guided_push_t_fp32/model-100000.pt"
    )["model"]
)
goal_model = torch.load("pusht_goal_reward_from_latents.pth")
env = GenerativePushT(
    model,
    goal_model,
    initial_state_path=f"{base_dir}/datasets/pusht_goal/pusht_goal/videos/initial_frame_latents.npy",
)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000, progress_bar=True)
model.save("ppo_pusht")
