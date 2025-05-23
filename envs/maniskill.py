import gymnasium as gym
import numpy as np
from functools import partial
import torch

class Maniskill:
    metadata = {}

    def __init__(self, task, size=(128, 128), seed=0):
        assert task in ("PickCube-v1")
        import mani_skill

        self._env = gym.make(
            task, # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
            num_envs=1,
            obs_mode="rgb", # there is also "state_dict", "rgbd", ...
            control_mode="pd_ee_delta_pos", # there is also "pd_joint_delta_pos", ...
            # render_mode="human"
        ) # TODO: add seed

        self.seed = seed

        self.obs_process_fn = partial( # from maniskill
            self.convert_obs,
            concat_fn=partial(torch.concatenate, dim=-1),
            # transpose_fn=partial(
            #     torch.permute, dims=(0, 3, 1, 2)
            # ),  # (B, H, W, C) -> (B, C, H, W)
            transpose_fn=lambda x: x,
            state_obs_extractor=lambda obs: list(obs["agent"].values()) + list(obs["extra"].values()),
            depth = False
        )

    def convert_obs(self, obs, concat_fn, transpose_fn, state_obs_extractor, depth = True): # from maniskill
        img_dict = obs["sensor_data"]
        ls = ["rgb"]
        if depth:
            ls = ["rgb", "depth"]

        new_img_dict = {
            key: transpose_fn(
                concat_fn([v[key] for v in img_dict.values()])
            )  # (C, H, W) or (B, C, H, W)
            for key in ls
        }
        if "depth" in new_img_dict and isinstance(new_img_dict['depth'], torch.Tensor): # MS2 vec env uses float16, but gym AsyncVecEnv uses float32
            new_img_dict['depth'] = new_img_dict['depth'].to(torch.float16)

        # Unified version
        states_to_stack = state_obs_extractor(obs)
        for j in range(len(states_to_stack)):
            if states_to_stack[j].dtype == np.float64:
                states_to_stack[j] = states_to_stack[j].astype(np.float32)
        try:
            state = np.hstack(states_to_stack)
        except:  # dirty fix for concat trajectory of states
            state = np.column_stack(states_to_stack)
        if state.dtype == np.float64:
            for x in states_to_stack:
                print(x.shape, x.dtype)
            import pdb

            pdb.set_trace()

        out_dict = {
            "state": state,
            "rgb": new_img_dict["rgb"],
        }

        if "depth" in new_img_dict:
            out_dict["depth"] = new_img_dict["depth"]


        return out_dict

    @property
    def observation_space(self):
        # return self._env.observation_space
        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8), #TODO: is channel first faster?
                "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32),
                "is_first": gym.spaces.Box(0, 1, (), dtype=bool), # alt: could represent as two-value discrete
                "is_last": gym.spaces.Box(0, 1, (), dtype=bool), # alt: could represent as two-value discrete
                "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool), # alt: could represent as two-value discrete
                "reward": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            })

    @property
    def action_space(self):
        return self._env.action_space

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        done = done.squeeze()

        reward = np.float32(reward)[0]
        
        processed_obs = self.obs_process_fn(obs)
        obs = {
            "image": processed_obs['rgb'].cpu(),
            "state": processed_obs['state'],
            "is_first": False,
            "is_last": done,
            "is_terminal": done,
            "reward": reward,
        }
        print(processed_obs['rgb'].shape)
        return obs, reward, done, truncated, info
    

    def render(self):
        return self._env.render()

    def reset(self):
        obs, _ = self._env.reset(seed=self.seed)
        processed_obs = self.obs_process_fn(obs)
        print(processed_obs['rgb'].shape)
        obs = {
            "image": processed_obs['rgb'].cpu(),
            "state": processed_obs['state'],
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }
        return obs
