from gymnasium import Env
from gymnasium import spaces
from random import randint
import torch
import numpy as np
from torch import nn
from einops import rearrange

import sys

sys.path.append(".")
from models.genie_redux import GenieReduxGuided


class GenerativePushT(Env):
    def __init__(self, model: GenieReduxGuided, goal_model, initial_state_path):
        super().__init__()
        self.action_space = spaces.Box(-1, 1, shape=(15, 2))
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

    def reset(self):
        # select random index for initial state and pose
        idx = randint(0, len(self.initial_states))
        self.initial_state = self.initial_states[idx]
        self.initial_pose = self.initial_poses[idx]
        self.state = torch.tensor(self.initial_state).to("cuda")
        self.pose = self.initial_pose

        self.step_count = 0

        return self.state

    def reward(self):
        with torch.no_grad():
            latent_codes = self.model.tokenizer.vq.codebook[self.state]
            latent_codes = latent_codes.unsqueeze(0)
            latent_codes = rearrange(latent_codes, "b ... -> b (...)")
            print(latent_codes.shape)
            reward = self.goal_model(latent_codes)
            reward = nn.Softmax(dim=1)(reward)
            pred = reward.cpu().numpy()
            print("pred", pred)
            reward = pred.argmax()

        return reward

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
                print("aciton", poses, poses.shape)

                prime_token_ids = torch.unsqueeze(self.state, 0)
                print("prime_token_ids", prime_token_ids.shape)

                prime_frames = self.render()
                prime_frames = torch.tensor(prime_frames).float().to("cuda")
                prime_frames = rearrange(prime_frames, "h w c -> c h w")
                prime_frames = prime_frames.unsqueeze(0)
                print("prime_frames", prime_frames.shape)

                p_frames = prime_frames[:1]
                p_frames = rearrange(p_frames, "f c h w -> c f h w")
                print(p_frames.shape)
                p_frames = p_frames.unsqueeze(0)

                prime_token_ids = self.model.get_tokenizer_codebook_ids(p_frames)
                prime_token_ids = rearrange(prime_token_ids, "b ... -> b (...)")

                output = self.model.dynamics.sample(
                    prime_token_ids=prime_token_ids,
                    num_tokens=self.model.num_tokens_per_frames(15, 1),
                    actions=poses,
                    patch_shape=self.model.get_video_patch_shape(16, 1),
                    return_confidence=False,
                )

        output = output.reshape(15, 256)
        self.state = output[0].squeeze()
        reward = self.reward()

        done = False
        terminated = False

        if reward >= 1:
            done = True
            terminated = False

        if self.step_count >= 16:
            done = False
            terminated = True

        self.step_count += 1
        info = {"step", self.step_count}

        indices = output.reshape(1, 15, 16, 16)
        images = self.model.decode_from_codebook_indices(indices)
        images = images.squeeze()
        images = rearrange(images, "c f h w -> f h w c")
        images = images.cpu().numpy()

        return self.state, reward, terminated, done, info

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
    from reward_from_latents import RewardFromLatents
    from matplotlib import pyplot as plt
    from data.push_t_wrapper import PushTDataset
    from hydra import compose, initialize
    from omegaconf import DictConfig
    from matplotlib import pyplot as plt
    from models import construct_model
    from pathlib import Path

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

    pt_ds = PushTDataset(
        dataset_path=pt_dataset_path,
        image_size=image_size,
        num_frames=num_frames,
    )

    model = construct_model(
        config=cfg,
    )

    goal_model = torch.load("pusht_goal_reward_from_latents.pth")
    prime_frames = pt_ds[10]["input_frames"][:1]
    print(prime_frames.shape)
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
