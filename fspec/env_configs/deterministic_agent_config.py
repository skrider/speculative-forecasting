import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gym.wrappers.time_limit import TimeLimit

from typing import Optional


def deterministic_agent_config(env_name: str, exp_name: Optional[str] = None, action = 0, **kwargs):
    log_string = "{env_name}_deterministic".format(env_name=env_name)

    return {
        "agent": "deterministic",
        "agent_kwargs": {
            "action": action,
        },
        "make_env": lambda: RecordEpisodeStatistics(TimeLimit(gym.make(env_name), 100)),
        "env_name": env_name,
        "log_name": log_string,
        "batch_size": 0,
        **kwargs,
    }
