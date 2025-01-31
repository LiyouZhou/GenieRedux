from pathlib import Path
import random
import sys
from random import randint

import numpy as np
import torch
from einops import rearrange
from gymnasium import Env, spaces
from torch import nn

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.genie_redux import GenieReduxGuided


class GenerativePushT(Env):
    def __init__(self, model: GenieReduxGuided, goal_model, initial_state_path):
        super().__init__()
        self.action_space = spaces.Box(-1, 1, shape=(15, 2))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(256, 32))
        self.model = model
        self.model = self.model.to("cuda")
        self.model.eval()
        self.action_min = torch.Tensor(np.array([0.2277, -0.5086])).to("cuda")
        self.action_max = torch.Tensor(np.array([0.8364, 0.5450])).to("cuda")
        self.action_range = self.action_max - self.action_min

        self.goal_model = goal_model
        self.goal_model = self.goal_model.to("cuda")
        self.goal_model.eval()

        self.initial_states = np.load(initial_state_path)
        self.initial_poses = np.load(initial_state_path.replace("latents", "poses"))

        self.reset()

    def reset(self, seed=None):
        # select random index for initial state and pose
        random.seed(seed)
        idx = randint(0, len(self.initial_states))
        self.initial_state = self.initial_states[idx]
        self.initial_pose = self.initial_poses[idx]
        self.state = torch.tensor(self.initial_state).to("cuda")
        self.pose = self.initial_pose

        self.step_count = 0
        info = {"step": self.step_count, "scores": None}

        return self.obs, info

    def reached_goal(self):
        with torch.no_grad():
            latent_codes = self.model.tokenizer.vq.codebook[self.state]
            latent_codes = latent_codes.unsqueeze(0)
            latent_codes = rearrange(latent_codes, "b ... -> b (...)")
            reward = self.goal_model(latent_codes)
            reward = nn.Softmax(dim=1)(reward)
            pred = reward.cpu().numpy()
            reward = pred.argmax()

        return reward == 1

    def step(self, actions):
        poses = []
        initial_pose = self.pose
        for action in actions:
            pose = np.add(initial_pose, action)
            poses.append(pose)
            initial_pose = pose

        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                poses = np.stack(poses)
                poses = torch.tensor(poses).float().to("cuda")
                poses = poses.unsqueeze(0)

                prime_token_ids = torch.unsqueeze(self.state, 0)
                prime_token_ids = rearrange(prime_token_ids, "b ... -> b (...)")

                output, scores = self.model.dynamics.sample(
                    prime_token_ids=prime_token_ids,
                    num_tokens=self.model.num_tokens_per_frames(15, 1),
                    actions=poses,
                    patch_shape=self.model.get_video_patch_shape(16, 1),
                    return_confidence=True,
                )

        output = output.reshape(15, 256)
        self.state = output[0].squeeze()
        info = {"step": self.step_count, "scores": scores}
        score = torch.mean(info["scores"])

        reached_goal = self.reached_goal()
        done = False
        terminated = False
        reward = -0.1

        if reached_goal:
            reward = 1
            done = True
        elif score.cpu().numpy() < 0.05:
            terminated = True
        elif self.step_count >= 300:
            reward = -1
            terminated = True

        self.step_count += 1

        return self.obs, float(reward), terminated, done, info

    @property
    def obs(self):
        obs_idx = self.state
        obs = self.model.tokenizer.vq.codebook[obs_idx]

        return obs.cpu().numpy()

    def render(self):
        indices = self.state
        indices = indices.reshape(1, 1, 16, 16)
        dummy_first_frame = indices.clone()
        indices = torch.cat([dummy_first_frame, indices], dim=1)
        image = self.model.decode_from_codebook_indices(indices)
        image = image.squeeze()
        image = image[:, 1]
        image = rearrange(image, "c h w -> h w c")

        return image.cpu().numpy()


if __name__ == "__main__":
    from pathlib import Path

    from hydra import compose, initialize
    from matplotlib import pyplot as plt
    from omegaconf import DictConfig
    from reward_from_latents import RewardFromLatents

    from models import construct_model

    base_dir = Path(__file__).resolve().parent.parent

    print(base_dir)

    with initialize(version_base=None, config_path=f"../configs", job_name="test_app"):
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
            f"{base_dir}/checkpoints/genie_redux_guided/genie_redux_guided_push_t_fp32/model-100000.pt"
        )["model"]
    )

    goal_model = torch.load("pusht_goal_reward_from_latents.pth")
    env = GenerativePushT(
        model,
        goal_model,
        initial_state_path=f"{base_dir}/datasets/pusht_goal/pusht_goal/videos/initial_frame_latents.npy",
    )

    state = env.reset()

    num_frames = 10
    fig, axs = plt.subplots(1, num_frames, figsize=(20, 5))
    for i in range(num_frames):
        img = env.render()
        img = img.clip(0, 1)
        axs[i].imshow(img)
        env.step(np.full([15, 2], 0.01))
