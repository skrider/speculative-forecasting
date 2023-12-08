import gym

from .spec import Spec

gym.register(
    id="SpeculativeDecoding-v0",
    entry_point=Spec,
    kwargs={
        "draft_model_path": "JackFram/llama-160m",
        "main_model_path": "JackFram/llama-160m",
        "tokenizer": "meta-llama/Llama-2-7b",
        "conversations_path": "datasets/chatbot-arena.parquet",
        "n_conversations": 1,
        "max_tokens_guess": 10,
        "max_tokens": 256,
        "accepted_tokens_weight": 1.0,
        "rejected_tokens_weight": 1.0,
    },
)
