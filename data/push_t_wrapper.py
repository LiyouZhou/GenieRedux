import sys
import os
from pathlib import Path


from omegaconf import OmegaConf
import numpy as np
from pathlib import Path

from data.data import EnvironmentDataset


p = Path(__file__).parent.parent
sys.path.append(str(p / "diffusion_policy"))

from diffusion_policy.dataset.real_pusht_image_dataset import RealPushTImageDataset


class PushTDataset(EnvironmentDataset):
    def __init__(
        self, dataset_path, image_size, num_frames, transforms=None, split=None
    ):
        if len(image_size) == 2:
            image_size = [3] + list(image_size)

        shape_meta = {
            "obs": {
                "camera_1": {"shape": image_size, "type": "rgb"},
                "camera_3": {"shape": image_size, "type": "rgb"},
                "robot_eef_pose": {"shape": [2], "type": "low_dim"},
            },
            "action": {"shape": [2]},
        }
        shape_meta = OmegaConf.create(shape_meta)

        self.ds = RealPushTImageDataset(
            shape_meta=shape_meta,
            dataset_path=dataset_path,
            horizon=num_frames,
            use_cache=True,
        )
        self.transforms = transforms

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        seq_len = len(sample["obs"]["camera_3"])
        is_first = np.zeros((seq_len, 1), dtype=int)
        is_first[0, :] = np.ones_like(is_first[0, :])

        input_frames = sample["obs"]["camera_3"]
        output_frame = sample["obs"]["camera_3"][-1]
        if self.transforms is not None:
            input_frames = self.transforms(input_frames)
            output_frame = self.transforms(output_frame)
        actions = sample["action"][:-1]

        return {
            "input_frames": input_frames,
            "output_frame": output_frame,
            "actions": actions,
            "is_first": is_first,
        }
