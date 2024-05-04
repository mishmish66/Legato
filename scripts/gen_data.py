import gymnasium as gym
import mujoco
import numpy as np

import concurrent.futures

from torch import nn

import tempfile

from tqdm import tqdm


def work_fn():
    env = gym.make("PointMaze_Medium-v3", max_episode_steps=1024)
    obs = env.reset()
    actions = []
    done = False
    truncated = False

    obs_acts = []

    while not done and not truncated:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        obs_acts.append((obs["observation"], action))

    return obs_acts


if __name__ == "__main__":

    trajectories = []

    num_trajectories = 65536
    pbar = tqdm(total=num_trajectories)
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        # Generate 1024 trajectories
        futures = [executor.submit(work_fn) for _ in range(num_trajectories)]
        for future in concurrent.futures.as_completed(futures):
            trajectories.append(future.result())
            pbar.update(1)

    # Consume the trajectories to create a tensor of observations and actions
    observations = []
    actions = []

    pbar = tqdm(total=len(trajectories))
    for trajectory in trajectories:
        trajectory_observations = []
        trajectory_actions = []
        for obs, act in trajectory:
            trajectory_observations.append(obs)
            trajectory_actions.append(act)
            del obs, act

        observations.append(trajectory_observations)
        actions.append(trajectory_actions)

        pbar.update(1)

    observations = np.array(observations)
    actions = np.array(actions)

    # Save the data
    np.savez_compressed("data.npz", observations=observations, actions=actions)
