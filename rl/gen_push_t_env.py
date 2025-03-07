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
    def __init__(self, model: GenieReduxGuided, goal_model, ds):
        super().__init__()
        self.action_space = spaces.Box(-1, 1, shape=(2,))
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

        self.ds = ds
        episode_ends = list(ds.replay_buffer.meta.episode_ends)
        self.episode_starts = [0] + episode_ends[:-1]

        self.reset()

    def observe(self):
        self.state[-1]
        return

    def reset(self, seed=None):
        # select random index for initial state and pose
        random.seed(seed)
        num_episodes = len(self.episode_starts)
        idx = randint(0, num_episodes - 1)
        idx = 0
        start_idx = self.episode_starts[idx]
        episode_actions = self.ds.replay_buffer.data.action[start_idx : start_idx + 15]
        episode_obs = self.ds.replay_buffer.data.camera_3[start_idx : start_idx + 15]

        episode_obs = torch.tensor(episode_obs).to("cuda")
        episode_obs = episode_obs / 255.0
        episode_obs = rearrange(episode_obs, "(b f) h w c -> b c f h w", b=1)

        with torch.no_grad():
            frame_latent = self.model.tokenizer(episode_obs, return_only_codebook_ids=True)

        self.state = frame_latent
        self.actions = torch.tensor(episode_actions).to("cuda")
        self.actions = rearrange(self.actions, "(b f) ... -> b f ...", b=1)

        print(f"state: {self.state.shape}")
        print(f"actions: {self.actions.shape}")

        self.step_count = 0
        info = {"step": self.step_count, "scores": None, "frames": episode_obs.clone()}

        return self.state, info

    def reached_goal(self):
        with torch.no_grad():
            latent_codes = self.model.tokenizer.vq.codebook[self.state[0, -1]]
            latent_codes = rearrange(latent_codes, "... -> 1 (...)")
            reward = self.goal_model(latent_codes)
            reward = nn.Softmax(dim=1)(reward)
            pred = reward.cpu().numpy()
            reward = pred.argmax()

        return reward == 1

    def step(self, action):
        action = torch.tensor(action).to("cuda")
        action = rearrange(action, "(b f d) -> b f d", b=1, f=1)
        action += self.actions[:, -1]
        self.actions = torch.cat([self.actions[:, 1:], action], dim=1)
        print("self.actions", self.actions.shape, self.actions.dtype)
        print("self.state", self.state.shape, self.actions.dtype)

        with torch.no_grad():
            output, scores = self.model.dynamics.sample(
                prime_token_ids=rearrange(self.state, "b ... -> b (...)"),
                num_tokens=self.model.num_tokens_per_frames(15, 1),
                actions=self.actions.float(),
                patch_shape=self.model.get_video_patch_shape(16, 1),
                return_confidence=True,
            )

        output = rearrange(output, "b (f w h) -> b f w h", b=1, w=16, h=16)
        self.state = torch.cat([self.state[:, 1:], output[:, -1:]], dim=1)
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
        image = self.model.decode_from_codebook_indices(self.state)
        image = rearrange(image, "b c f h w -> (b f) h w c")
        image = image[-1]
        return image.cpu().numpy()


if __name__ == "__main__":
    from pathlib import Path
    from matplotlib import pyplot as plt
    from reward_from_latents import RewardFromLatents
    from model_utils import construct_model_from_checkpoint, create_push_t_image_dataset

    base_dir = Path(__file__).resolve().parent.parent
    model = construct_model_from_checkpoint(
        "checkpoints/tokenizer/tokenizer_250213112859/model-50000.pt",
        "checkpoints/genie_redux_guided/genie_redux_guided_250217235935/model-100000.pt",
    )
    torch.serialization.add_safe_globals([RewardFromLatents])
    goal_model = torch.load("pusht_goal_reward_from_latents.pth", weights_only=False)
    ds = create_push_t_image_dataset(
        pt_dataset_path="datasets/pusht_real/pusht_real",
    )
    env = GenerativePushT(
        model,
        goal_model,
        ds,
    )

    state, info = env.reset()

    seed_imgs = info["frames"]
    print("seed_imgs", seed_imgs.shape)
    # seed_imgs = rearrange(seed_imgs, "b c f h w -> (b f) h w c")
    for i in range(seed_imgs.shape[0]):
        img = ds.replay_buffer.data.camera_3[0 : i + 1]
        print("img", img.shape)

        plt.imsave(f"seed_img_{i}.png", seed_imgs[i].cpu())


    num_frames = 10
    fig, axs = plt.subplots(1, num_frames, figsize=(20, 5))
    for i in range(num_frames):
        img = env.render()
        img = img.clip(0, 1)
        axs[i].imshow(img)
        plt.imsave(f"img_{i}.png", img)
        env.step(
            np.full(
                [
                    2,
                ],
                0.01,
            )
        )
