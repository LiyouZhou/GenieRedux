import functools
import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat

from models.dynamics import Dynamics
from models.tokenizer import Tokenizer
from models.lam import LatentActionModel

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else (val,) * length


def reduce_mult(arr):
    return functools.reduce(lambda x, y: x * y, arr)


def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


class GenieReduxGuided(nn.Module):
    def __init__(
        self,
        tokenizer: Tokenizer,
        dynamics: Dynamics,
        *args,
        **kwargs,
    ):
        if "assert_unguided" not in kwargs or not kwargs["assert_unguided"]:
            assert dynamics.maskgit.is_guided, "Dynamics must be guided"
        super().__init__()

        self.tokenizer = tokenizer
        self.tokenizer.eval()
        for param in self.tokenizer.parameters():
            param.requires_grad = False

        self.dynamics = dynamics

        self.image_size = tokenizer.image_size

        self.dim_actions = dynamics.maskgit.action_dim

        config = {}
        arguments = locals()
        for key in arguments.keys():
            if key not in [
                "self",
                "config",
                "__class__",
                "tokenizer",
                "dynamics",
                "param",
            ]:
                config[key] = arguments[key]
        self.config = config

        self.device = next(self.dynamics.parameters()).device

    @property
    def patch_height_width(self):
        return self.tokenizer.patch_height_width

    def trainable_parameters(self):
        trainable_parameters = list(self.dynamics.parameters())
        return trainable_parameters

    def state_dict(self, *args, **kwargs):
        state_dict = {"dynamics": self.dynamics.state_dict(*args, **kwargs)}
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.dynamics.load_state_dict(state_dict["dynamics"], *args, **kwargs)

    def accelator_log(self, accelerator_tracker_dict, loss):
        if exists(accelerator_tracker_dict):
            train = accelerator_tracker_dict["train"]
            tracker = accelerator_tracker_dict["tracker"]
            step = accelerator_tracker_dict["step"]

            mode = "Train" if train else "Validation"
            tracker.log(
                {
                    f"{mode} cross entropy loss": loss.item(),
                },
                step=step,
            )

    def get_tokenizer_codebook_ids(self, videos):
        return self.tokenizer(videos, return_only_codebook_ids=True)

    def num_tokens_per_frames(self, num_frames, num_first_frames):
        return self.tokenizer.num_tokens_per_frames(num_frames, num_first_frames)

    def get_video_patch_shape(self, num_frames, num_first_frames):
        return self.tokenizer.get_video_patch_shape(num_frames, num_first_frames)

    def decode_from_codebook_indices(self, codebook_indices):
        return self.tokenizer.decode_from_codebook_indices(codebook_indices)

    def get_codes_from_indices(self, indices):
        # convert the indices to action one hot vectors
        return F.one_hot(indices, num_classes=self.dim_actions).float()

    @eval_decorator
    @torch.no_grad()
    def sample(
        self,
        prime_frames=None,
        actions=None,
        num_frames=None,
        return_token_ids=False,
        return_confidence=False,
        inference_steps=1,
        mask_schedule="cosine",
        sample_temperature=1.0,
        return_logits=False,
        return_tokens=False,
        *args,
        **kwargs,
    ):
        assert prime_frames is not None, "Prime frames should be provided"
        assert actions is not None, "Actions should be provided"
        assert num_frames is not None, "Number of frames to generate should be provided"

        # derive the priming token ids, to be prepended to the input being demasked by mask-git at each round
        prime_token_ids = self.get_tokenizer_codebook_ids(prime_frames)
        prime_token_ids = rearrange(prime_token_ids, "b ... -> b (...)")

        prime_num_frames = prime_frames.shape[2]
        num_tokens = self.num_tokens_per_frames(
            num_frames, num_first_frames=prime_num_frames
        )
        patch_shape = self.get_video_patch_shape(
            num_frames + prime_num_frames, num_first_frames=prime_num_frames
        )

        # derive the latent actions
        actions = self.get_codes_from_indices(actions)

        output = self.dynamics.sample(
            prime_token_ids=prime_token_ids,
            actions=actions,
            num_tokens=num_tokens,
            patch_shape=patch_shape,
            return_confidence=return_confidence,
            inference_steps=inference_steps,
            mask_schedule=mask_schedule,
            sample_temperature=sample_temperature,
            return_logits=return_logits,
        )

        if return_logits:
            logits = output
            return logits

        if return_confidence:
            video_token_ids, confidence = output
        else:
            video_token_ids = output

        video_token_ids = torch.cat((prime_token_ids, video_token_ids), dim=-1)

        if return_token_ids:
            if return_confidence:
                return video_token_ids, confidence
            else:
                return video_token_ids

        video = self.decode_from_codebook_indices(video_token_ids)

        video = video[:, :, prime_num_frames:]

        if return_tokens:
            return video, video_token_ids

        if return_confidence:
            return video, confidence
        else:
            return video

    def generate_interactive_video(
        self,
        prime_frames=None,
        actions=None,
        num_frames=None,
        dream_length=2,
        inference_steps=25,
        mask_schedule="cosine",
        sample_temperature=1.0,
        return_confidence=False,
        return_logits=False,
        return_tokens=False,
        *args,
        **kwargs,
    ):
        assert prime_frames is not None, "Prime frames should be provided"
        assert actions is not None, "Actions should be provided"
        assert num_frames is not None, "Number of frames to generate should be provided"

        prime_token_ids = self.get_tokenizer_codebook_ids(prime_frames)

        # Extend actions to cover dream length
        actions = torch.cat([actions] + [actions[:, -1:]] * dream_length, dim=1)

        actions = self.get_codes_from_indices(actions)

        initial_prime_num_frames = prime_frames.shape[2]

        video_tokens = prime_token_ids
        full_confidence = None

        for i in range(0, num_frames, dream_length):
            # Calculate window start and end frames
            start_frame = 0  # max(0, i + initial_prime_num_frames - window_size)
            end_frame = i + initial_prime_num_frames

            # Extract tokens for the current window
            window_tokens = video_tokens[:, start_frame:end_frame]
            window_num_frames = end_frame - start_frame

            # Prepare tokens and actions for generation
            patch_shape = window_tokens.shape[1:]
            window_tokens_flat = rearrange(window_tokens, "b ... -> b (...)")

            num_tokens = self.tokenizer.num_tokens_per_frames(
                dream_length, num_first_frames=window_num_frames
            )

            patch_shape = self.tokenizer.get_video_patch_shape(
                dream_length + window_num_frames, num_first_frames=window_num_frames
            )

            window_actions = actions[:, start_frame : end_frame + dream_length - 1]

            # Generate new tokens for the next frame
            output = self.dynamics.sample(
                prime_token_ids=window_tokens_flat,
                actions=window_actions,
                prompt_tokens=None,
                num_tokens=num_tokens,
                patch_shape=patch_shape,
                inference_steps=inference_steps,
                mask_schedule=mask_schedule,
                sample_temperature=sample_temperature,
                return_confidence=return_confidence,
                return_logits=return_logits,
            )

            if return_logits:
                new_tokens, logits = output
                if full_logits is None:
                    full_logits = logits
                else:
                    for ind, item in enumerate(logits):
                        item = rearrange(
                            item,
                            "b (t h w) -> b t h w",
                            h=patch_shape[1],
                            w=patch_shape[2],
                        )
                        full_item = rearrange(
                            full_logits[ind],
                            "b (t h w) -> b t h w",
                            h=patch_shape[1],
                            w=patch_shape[2],
                        )
                        full_item = torch.cat([full_item, item], dim=1)
                        full_logits[ind] = rearrange(full_item, "b t h w -> b (t h w)")
            elif return_confidence:
                new_tokens, confidence = output
                # full_confidence.append(confidence)
                if full_confidence is None:
                    full_confidence = confidence
                else:
                    for ind, item in enumerate(confidence):
                        item = rearrange(
                            item,
                            "b (t h w) -> b t h w",
                            h=patch_shape[1],
                            w=patch_shape[2],
                        )
                        full_item = rearrange(
                            full_confidence[ind],
                            "b (t h w) -> b t h w",
                            h=patch_shape[1],
                            w=patch_shape[2],
                        )
                        full_item = torch.cat([full_item, item], dim=1)
                        full_confidence[ind] = rearrange(
                            full_item, "b t h w -> b (t h w)"
                        )
            else:
                new_tokens = output

            # Extract the last frame's tokens
            _, h, w = patch_shape
            new_frame_tokens = rearrange(new_tokens, "b (t h w) -> b t h w", h=h, w=w)[
                :, :
            ]

            # Append new frame tokens to the video tokens
            video_tokens = torch.cat([video_tokens, new_frame_tokens], dim=1)

            print("Generated frame", i + 1)

        # Decode all generated frames at once at the end
        video = self.decode_from_codebook_indices(
            video_tokens  # [:, initial_prime_num_frames:]
        )

        video = video[:, :, initial_prime_num_frames:]

        if return_tokens:
            return video, video_tokens[:, initial_prime_num_frames:]

        if return_logits:
            return video, full_logits

        if return_confidence:
            return video, full_confidence

        return video

    def forward(
        self,
        videos=None,
        actions=None,
        video_mask=None,
        accelerator_tracker_dict=None,
        return_token_ids=False,
        *args,
        **kwargs,
    ):

        assert exists(videos) and exists(actions), "videos and actions must be provided"

        with torch.no_grad():
            video_codebook_ids = self.tokenizer(videos, return_only_codebook_ids=True)

        video_codebook_ids = video_codebook_ids.detach()

        return self.dynamics(
            video_codebook_ids=video_codebook_ids,
            actions=actions,
            video_mask=video_mask,
            accelerator_tracker_dict=accelerator_tracker_dict,
            return_token_ids=return_token_ids,
        )


class GenieRedux(GenieReduxGuided):
    def __init__(
        self,
        tokenizer: Tokenizer,
        latent_action_model: LatentActionModel,
        dynamics: Dynamics,
    ):
        assert not dynamics.maskgit.is_guided, "Dynamics must be unguided"

        super().__init__(
            tokenizer=tokenizer,
            dynamics=dynamics,
            assert_unguided=True,
        )

        self.latent_action_model = latent_action_model

        if "latent_action_model" in self.config:
            del self.config["latent_action_model"]

    def trainable_parameters(self):
        trainable_parameters = list(self.dynamics.parameters())
        trainable_parameters += list(self.latent_action_model.parameters())
        return trainable_parameters

    def state_dict(self, *args, **kwargs):
        state_dict = {}
        state_dict["dynamics"] = self.dynamics.state_dict()
        state_dict["latent_action_model"] = self.latent_action_model.state_dict()
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.dynamics.load_state_dict(state_dict["dynamics"], *args, **kwargs)
        self.latent_action_model.load_state_dict(
            state_dict["latent_action_model"], *args, **kwargs
        )

    def get_codes_from_indices(self, indices):
        h, w = self.patch_height_width
        actions = self.latent_action_model.get_codes_from_indices(indices)
        actions = repeat(actions, "b t d -> b t h w d", h=h, w=w)
        return actions

    def forward(
        self,
        videos,
        video_mask=None,
        accelerator_tracker_dict=None,
        *args,
        **kwargs,
    ):

        with torch.no_grad():
            video_codebook_ids = self.tokenizer(videos, return_only_codebook_ids=True)
        lam_loss, latent_actions = self.latent_action_model(
            videos,
            return_tokens=True,
            accelerator_tracker_dict=accelerator_tracker_dict,
        )

        video_codebook_ids = video_codebook_ids.detach()

        dyn_loss = self.dynamics(
            video_codebook_ids,
            latent_actions,
            video_mask=video_mask,
            accelerator_tracker_dict=accelerator_tracker_dict,
            return_token_ids=False,
        )

        loss = lam_loss + dyn_loss

        return loss
