import time
import argparse
import pickle
import ray

from fspec.agents import agents as agent_types

import os
import time

import gym
import numpy as np
import torch
from fspec.infrastructure import pytorch_util as ptu
import tqdm

from fspec.infrastructure import utils
from fspec.infrastructure.logger import Logger
from fspec.infrastructure.replay_buffer import ReplayBuffer
import fspec.envs as _

from scripting_utils import make_logger, make_config

def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    dataset_name = f"{config['dataset_name']}_{int(time.time())}"
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = config["make_env"]()
    exploration_schedule = config.get("exploration_schedule", None)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    assert discrete, "DQN only supports discrete action spaces"

    agent_cls = agent_types[config["agent"]]
    agent = agent_cls(
        env.observation_space.shape,
        env.action_space.n,
        **config["agent_kwargs"],
    )

    ep_len = env.spec.max_episode_steps or env.max_episode_steps

    observation = None

    # Replay buffer
    replay_buffer = ReplayBuffer(capacity=config["total_steps"])

    observation = env.reset()

    recent_observations = []

    for step in tqdm.trange(config["total_steps"], dynamic_ncols=True):
        if exploration_schedule is not None:
            epsilon = exploration_schedule.value(step)
            action = agent.get_action(observation, epsilon)
        else:
            epsilon = None
            action = agent.get_action(observation)

        next_observation, reward, done, info = env.step(action)
        next_observation = np.asarray(next_observation)

        truncated = info.get("TimeLimit.truncated", False)

        replay_buffer.insert(
            observation=observation,
            action=action,
            reward=reward,
            done=done and not truncated,
            next_observation=next_observation,
        )
        recent_observations.append(observation)

        # Handle episode termination
        if done:
            observation = env.reset()

            logger.log_scalar(info["episode"]["r"], "train_return", step)
            logger.log_scalar(info["episode"]["l"], "train_ep_len", step)
        else:
            observation = next_observation

        # Main training loop
        batch = replay_buffer.sample(config["batch_size"])

        # Convert to PyTorch tensors
        batch = ptu.from_numpy(batch)

        update_info = agent.update(
            batch["observations"],
            batch["actions"],
            batch["rewards"],
            batch["next_observations"],
            batch["dones"],
            step,
        )

        # Logging code
        if epsilon is not None:
            update_info["epsilon"] = epsilon

        if step % args.log_interval == 0:
            for k, v in update_info.items():
                logger.log_scalar(v, k, step)
            logger.flush()

        if step % args.eval_interval == 0:
            # Evaluate
            trajectories = utils.sample_n_trajectories(
                env,
                agent,
                args.num_eval_trajectories,
                ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

            logger.log_scalar(np.mean(returns), "eval_return", step)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)

            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", step)
                logger.log_scalar(np.max(returns), "eval/return_max", step)
                logger.log_scalar(np.min(returns), "eval/return_min", step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)

            actions = np.concatenate([t["action"] for t in trajectories], 0)
            logger.log_histogram(actions, "eval/actions", step)


            if not args.no_save:
                dataset_file = os.path.join(args.dataset_dir, f"{dataset_name}.pkl")
                with open(dataset_file, "wb") as f:
                    pickle.dump(replay_buffer, f)
                    print("Saved dataset to", dataset_file)

    if not args.no_save:   
        dataset_file = os.path.join(args.dataset_dir, f"{dataset_name}.pkl")
        with open(dataset_file, "wb") as f:
            pickle.dump(replay_buffer, f)
            print("Saved dataset to", dataset_file)


banner = """
======================================================================
Draftsman

Training an agent for the {env} environment using algorithm {alg}.
======================================================================
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)

    parser.add_argument("--eval_interval", "-ei", type=int, default=10000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--no_save", type=bool)

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "fspec_"  # keep for autograder

    config = make_config(args.config_file)
    logger = make_logger(logdir_prefix, config)

    print(banner.format(env=config["env_name"], alg=config["agent"]))

    run_training_loop(config, logger, args)

if __name__ == "__main__":
    main()
