import numpy as np
import math
import gym
import torch
from typing import List, Tuple
import ray
import pandas as pd
from draftsman.infrastructure import pytorch_util as ptu
from dataclasses import dataclass

@dataclass
class SpeculativeDecodingRun:
    """
    Dataclass for speculative decoding run
    """
    input_ids: np.ndarray
    main_hidden_states: np.ndarray
    draft_hidden_states: np.ndarray
    accept_mask: np.ndarray
    dataset_index: int

class LogicalSpeculativeDecodingDataset():
    def __init__(self, datasets, offset, size):
        self.dfs = []
        self.sizes = []
        self.size = 0
        for d in datasets:
            df = pd.read_parquet(d, engine="fastparquet")
            self.sizes += [len(df)]
            self.size += len(df)
            self.dfs.append(df)
        assert offset + size <= self.size

        self.offset = offset
        self.logical_size = size

    def get(self, index):
        index = index % self.logical_size
        physical_index = self.offset + index
        for i, size in enumerate(self.sizes):
            if physical_index < size:
                return self.dfs[i].loc()[physical_index]
            physical_index -= size
        raise IndexError()

class SpeculativeDecoding(gym.Env):
    """
    Class for speculative decoding gym environments
    """

    def __init__(
        self,
        conversations_paths: str,
        n_conversations: int,
        max_tokens_guess: int,
        max_tokens: int,
        conversation_offset: int = 0,
        accepted_tokens_weight: float = 1.0,
        rejected_tokens_weight: float = 1.0,
        missed_tokens_weight: float = 0.0,
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
        self.max_tokens = max_tokens

        self.accepted_tokens_weight = accepted_tokens_weight
        self.rejected_tokens_weight = rejected_tokens_weight
        self.missed_tokens_weight = missed_tokens_weight

        if logarithmic:
            self.action_space = gym.spaces.Discrete(int(math.log2(self.num_actions)) + 1)
        else:
            self.action_space = gym.spaces.Discrete(self.num_actions)

        # conversations from ShareGPT
        self.conversations = LogicalSpeculativeDecodingDataset(
            conversations_paths, 
            conversation_offset, 
            n_conversations
        )
        self.conversation_offset = conversation_offset

        c0_raw = self.conversations.get(0)
        c0 = self._parse_item(c0_raw)

        self.use_main_hidden_states = use_main_hidden_states
        self.use_draft_hidden_states = use_draft_hidden_states
        self.one_hot_encode_prev = one_hot_encode_prev

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
        self.max_episode_steps = self.max_tokens

    def _parse_item(self, item):
        input_ids = np.frombuffer(item.input_ids, dtype=np.int64)
        main_hidden_states = np.frombuffer(item.main_hidden_states, dtype=np.float32).reshape(self.max_tokens, -1)
        draft_hidden_states = np.frombuffer(item.draft_hidden_states, dtype=np.float32).reshape(self.max_tokens, -1)
        accept_mask = np.frombuffer(item.accept_mask, dtype=np.float32)
        # cast to int
        accept_mask = accept_mask.astype(np.int32)
        dataset_index = item.dataset_index
        return SpeculativeDecodingRun(
            input_ids=input_ids,
            main_hidden_states=main_hidden_states,
            draft_hidden_states=draft_hidden_states,
            accept_mask=accept_mask,
            dataset_index=dataset_index,
        )

    def _get_hiddens(self, token_index):
        obs = []
        if token_index >= self.max_tokens:
            token_index = self.max_tokens - 1
        if self.use_main_hidden_states:
            obs += [self.conversation.main_hidden_states[token_index]]
        if self.use_draft_hidden_states:
            obs += [self.conversation.draft_hidden_states[token_index]]
        # convert to numpy
        return np.concatenate(obs)
    
    def _encode_prev_accept_reject(self, prev_accept, prev_reject):
        if self.one_hot_encode_prev:
            return np.concatenate((
                np.eye(self.max_tokens_guess)[prev_accept - 1].astype(np.float32),
                np.eye(self.max_tokens_guess)[prev_reject - 1].astype(np.float32)
            ))
        else:
            return np.array([prev_accept, prev_reject]).astype(np.float32)

    def reset(self):
        conversation_index = np.random.randint(self.n_conversations)
        self.conversation = self._parse_item(self.conversations.get(conversation_index))
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

        accept_mask = self.conversation.accept_mask[self.token_index:min(self.token_index + num_tokens, self.max_tokens)]
        missed_mask = self.conversation.accept_mask[min(self.token_index + num_tokens, self.max_tokens):min(self.token_index + self.max_tokens_guess, self.max_tokens)]
        if len(accept_mask) == 0:
            n_accepted = n_rejected = 0
        else:
            n_accepted = np.argmax(accept_mask)
            if accept_mask[n_accepted] == 0:
                # no tokens rejected
                n_accepted = num_tokens
            n_rejected = num_tokens - n_accepted

        if len(missed_mask) == 0:
            n_missed = 0
        else:
            n_missed = np.argmax(missed_mask)
            if missed_mask[n_missed] == 0:
                n_missed = len(missed_mask)

        reward = self.accepted_tokens_weight * n_accepted
        reward -= self.rejected_tokens_weight * n_rejected
        reward -= self.missed_tokens_weight * n_missed

        # we always decode at least one token, but in an actual online setting we 
        # would not have access to its hidden state. So we need to take the second-
        # to-last token as our observation.
        self.token_index += n_accepted

        obs = np.concatenate((
            self._get_hiddens(self.token_index),
            self._encode_prev_accept_reject(n_accepted, n_rejected)
        ))

        # we always decode at least one token
        self.token_index += 1
        
        done = self.token_index >= self.max_tokens

        return obs, reward, done, {}
