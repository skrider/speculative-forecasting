import numpy as np
import math
import gym
import torch
from typing import List, Tuple
import ray
import pandas as pd
from draftsman.infrastructure import pytorch_util as ptu

class SpeculativeDecoding(gym.Env):
    """
    Class for speculative decoding gym environments
    """

    def __init__(
        self,
        conversations_path: str,
        n_conversations: int,
        max_tokens_guess: int,
        max_tokens: int,
        conversation_offset: int = 0,
        accepted_tokens_weight: float = 1.0,
        rejected_tokens_weight: float = 1.0,
        logarithmic: bool = False,
        use_main_hidden_states: bool = False,
        use_draft_hidden_states: bool = False,
        one_hot_encode_prev: bool = False,
    ):
        self.logarithmic = logarithmic
        self.num_actions = max_tokens_guess

        # TODO add cache
        # TODO enable flash attention
        self.n_conversations = n_conversations

        self.max_tokens_guess = max_tokens_guess
        self.accepted_tokens_weight = accepted_tokens_weight
        self.rejected_tokens_weight = rejected_tokens_weight

        if logarithmic:
            self.action_space = gym.spaces.Discrete(int(math.log2(self.num_actions)) + 1)
        else:
            self.action_space = gym.spaces.Discrete(self.num_actions)

        # conversations from ShareGPT
        self.conversations_df = pd.read_parquet(conversations_path, engine="pyarrow")
        self.conversation_offset = conversation_offset

        __import__('pdb').set_trace()
        c0 = self.conversations_df.loc()[self.conversation_offset]
        self.obs_size = 0
        if use_main_hidden_states:
            self.main_hidden_dim = len(c0.main_hidden_states[0])
            self.obs_size += self.main_hidden_dim
        if use_draft_hidden_states:
            self.draft_hidden_dim = len(c0.draft_hidden_states[0])
            self.obs_size += self.draft_hidden_dim
        if one_hot_encode_prev:
            self.obs_size += self.max_tokens_guess * 2
        else:
            self.obs_size += 2

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32
        )
        self.action_dim = self.ac_dim = 1
        self.observation_dim = self.obs_dim = 1

    def _get_hiddens(self, token_index):
        obs = []
        if self.use_main_hidden_states:
            obs += self.conversations_df.loc()[self.conversation_index].main_hidden_states[token_index]
        if self.use_draft_hidden_states:
            obs += self.conversations_df.loc()[self.conversation_index].draft_hidden_states[token_index]
        # convert to numpy
        return np.concatenate(obs)
    
    def _encode_prev_accept_reject(self, prev_accept, prev_reject):
        if self.one_hot_encode_prev:
            return np.concatenate((
                np.eye(self.max_tokens_guess)[prev_accept].astype(np.float32),
                np.eye(self.max_tokens_guess)[prev_reject].astype(np.float32)
            ))
        else:
            return np.array([prev_accept, prev_reject]).astype(np.float32)

    def reset(self):
        self.conversation_index = np.random.randint(self.n_conversations) + self.conversation_offset
        indexer = self.conversations_df.loc()
        self.token_index = 0

        obs = np.concatenate((
            self._get_hiddens(self.token_index),
            self._encode_prev_accept_reject(0, 0)
        ))
        
        return obs

    def step(self, action):
        # generate draft
        if self.logarithmic:
            num_tokens = 2 ** action
        else:
            num_tokens = action

        conversation = self.conversations_df.loc()[self.conversation_index]
        accept_mask = conversation.accept_mask[self.token_index:min(self.token_index + num_tokens, self.max_tokens)]
        if len(accept_mask == 0):
            n_accepted = n_rejected = 0
        else:
            n_accepted = np.argmax(accept_mask)
            if accept_mask[n_accepted] == 0:
                # no tokens rejected
                n_accepted = num_tokens
            n_rejected = num_tokens - n_accepted

        reward = self.accepted_tokens_weight * n_accepted - self.rejected_tokens_weight * n_rejected

        # we always decode at least one token
        self.token_index += n_accepted + 1

        obs = np.concatenate((
            self._get_hiddens(),
            self._encode_prev_accept_reject(n_accepted, n_rejected)
        ))
        
        done = self.token_index >= self.max_tokens

        return obs, reward, done, {}
