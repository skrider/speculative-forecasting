import numpy as np
import gym
import pickle
from vllm import LLM, SamplingParams
import torch
from typing import List, Tuple, Dict
import ray
import pandas as pd
from transformers import LlamaForCausalLM, LlamaTokenizerFast
from draftsman.infrastructure import pytorch_util as ptu

class GPUActor:
    def __init__(self, model_path: str):
        self.model = LlamaForCausalLM.from_pretrained(model_path, use_cache=False)
        self.model.eval()
        self.model.cuda()
        self.model.half()
    def spec_step_deterministic(self, draft):
        main_prompts = [prefix]
        for i in range(len(draft)):
            main_prompts += [prefix + draft[: i + 1]]
        
        main_run = self.main_llm.generate.remote(
            prompt_token_ids=main_prompts, 
            sampling_params=self.main_sp, 
            use_tqdm=False
        )
        main_run = ray.get(main_run)

        n = 0
        sampled_token = main_run[0].outputs[0].token_ids[0]
        for i, x in enumerate(draft):
            sampled_token = main_run[i].outputs[0].token_ids[0]
            if x == sampled_token:
                n += 1
            else:
                break

        done = main_run[n].outputs[0].finish_reason != "length"

        return draft[:n] + [sampled_token], done
        

class SpeculativeDecoding(gym.Env):
    """
    Class for speculative decoding gym environments
    """

    def __init__(
        self,
        draft_model_path: str,
        main_model_path: str,
        tokenizer: str,
        conversations_path: str,
        n_conversations: int,
        max_tokens_guess: int,
        accepted_tokens_weight: float = 1.0,
        rejected_tokens_weight: float = 1.0,
        tensor_parallel_size: int = 1,
        max_tokens: int = 100,
    ):
        self.main_llm = ray.remote(num_gpus=(1 if tensor_parallel_size == 1 else 0))(LLM).remote(
            model=main_model_path,
            tokenizer=tokenizer,
            max_num_seqs=max_tokens_guess,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            dtype="half",
            gpu_memory_utilization=0.9,
        )
        self.main_sp = SamplingParams(n=1, temperature=0., max_tokens=1)
        self.num_actions = max_tokens_guess
        self.max_tokens = max_tokens
        self.max_episode_steps = max_tokens

        # TODO add cache
        # TODO enable flash attention
        self.draft_llm = LlamaForCausalLM.from_pretrained(draft_model_path, use_cache=True)
        self.draft_llm.eval()
        self.draft_llm.to(ptu.device)
        self.draft_hidden_dim = self.draft_llm.config.hidden_size
        self.tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer)
        self.n_conversations = n_conversations

        self.max_tokens_guess = max_tokens_guess
        self.accepted_tokens_weight = accepted_tokens_weight
        self.rejected_tokens_weight = rejected_tokens_weight

        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.draft_hidden_dim + 2,), dtype=np.float32
        )
        self.action_dim = self.ac_dim = 1
        self.observation_dim = self.obs_dim = 1
        
        # conversations from ShareGPT
        self.conversations_df = pd.read_parquet(conversations_path, engine="pyarrow")

    def seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _get_draft_last_hidden(self):
        with torch.no_grad():
            draft_out = self.draft_llm(
                input_ids=self.tokens.unsqueeze(0), 
                output_hidden_states=True, 
                use_cache=False)

        # get last hidden state
        return draft_out.hidden_states[-1][0, -1, :].detach().cpu().numpy()

    def reset(self):
        conversation_index = np.random.randint(self.n_conversations)
        indexer = self.conversations_df.loc()
        self.prompt = indexer[conversation_index].text
        self.tokens = torch.as_tensor(self.tokenizer.encode(self.prompt)).to(ptu.device)

        last_hidden = self._get_draft_last_hidden()
        obs = np.concatenate((
            last_hidden,
            np.array([0., 0.])
        ))
        return obs

    def _tokens_list(self):
        return self.tokens.cpu().numpy().tolist()

    def step(self, action):
        # generate draft
        num_tokens = action
        if num_tokens > 0:
            draft_out = self.draft_llm.generate(
                input_ids=self.tokens.unsqueeze(0),
                max_new_tokens=num_tokens,
                num_return_sequences=1,
                use_cache=True,
                do_sample=False,
            )
            draft = draft_out.squeeze(0)[self.tokens.shape[0]:].cpu().numpy().tolist()
        else:
            draft = []
        
        accepted_draft, done = self._main_model_step_deterministic(
            self._tokens_list(), draft
        )

        # update tokens
        self.tokens = torch.cat((
            self.tokens, 
            torch.as_tensor(accepted_draft).to(ptu.device)
        ))

        draft_accepted_tokens = len(accepted_draft) - 1
        draft_wasted_tokens = len(draft) - draft_accepted_tokens

        # compute reward
        reward = self.accepted_tokens_weight * draft_accepted_tokens - self.rejected_tokens_weight * draft_wasted_tokens

        last_hidden_state = self._get_draft_last_hidden()
        # append accepted and wasted tokens
        obs = np.concatenate((
            last_hidden_state,
            np.array([draft_accepted_tokens, draft_wasted_tokens]).astype(np.float32)
        ))
        
        done = done or len(self.tokens) >= self.max_tokens

        return obs, reward, done, {}

    def _main_model_step_deterministic(
            self,
            prefix: List[int], 
            draft: List[int]) -> Tuple[List[int], bool]:
        main_prompts = [prefix]
        for i in range(len(draft)):
            main_prompts += [prefix + draft[: i + 1]]
        
        main_run = self.main_llm.generate.remote(
            prompt_token_ids=main_prompts, 
            sampling_params=self.main_sp, 
            use_tqdm=False
        )
        main_run = ray.get(main_run)

        n = 0
        sampled_token = main_run[0].outputs[0].token_ids[0]
        for i, x in enumerate(draft):
            sampled_token = main_run[i].outputs[0].token_ids[0]
            if x == sampled_token:
                n += 1
            else:
                break

        done = main_run[n].outputs[0].finish_reason != "length"

        return draft[:n] + [sampled_token], done
