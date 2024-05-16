import argparse
import concurrent.futures

import gymnasium as gym
import humanize
import numpy as np
from torch import nn
from tqdm import tqdm


def gen_trajectory(env, big_change_prob=0.01):
    obs = env.reset()
    actions = []
    done = False
    truncated = False

    obs_acts = []

    np_rng = np.random.default_rng()
    big_act_sampler = lambda: env.action_space.sample()

    def small_act_sampler(act):
        noised_act = act + np_rng.normal(0, 0.5, size=2)
        return np.clip(noised_act, env.action_space.low, env.action_space.high)

    action = big_act_sampler()

    while not done and not truncated:
        if np_rng.random() < big_change_prob:
            action = big_act_sampler()
        else:
            action = small_act_sampler(action)

        obs, reward, done, truncated, info = env.step(action)
        obs_acts.append((obs["observation"], action))

    return obs_acts


def work_fn(batch_size: int):

    env = gym.make("PointMaze_Medium-v3", max_episode_steps=1024, continuing_task=True)

    traj_results = [gen_trajectory(env) for _ in range(batch_size)]

    return traj_results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trajectories", type=int, default=65536)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    num_trajectories = args.num_trajectories
    batch_size = args.batch_size

    trajectories = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=24) as executor:
        futures = [
            executor.submit(work_fn, batch_size)
            for _ in range(num_trajectories // batch_size)
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            trajectories_to_add = future.result()
            trajectories.extend(trajectories_to_add)
            # Clean up the future
            del future

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

    pbar.close()

    observations = np.array(observations)
    actions = np.array(actions)

    # Save the data
    total_size_bytes = observations.nbytes + actions.nbytes
    total_size_formatted = humanize.naturalsize(total_size_bytes)

    print(f"Total size of generated data: {total_size_formatted}")
    np.savez_compressed("data.npz", observations=observations, actions=actions)
