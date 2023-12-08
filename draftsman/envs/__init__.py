import gym

from .spec import SpeculativeDecoding

gym.register(
    id="SpeculativeDecoding-v0",
    entry_point=SpeculativeDecoding,
    kwargs={
        "draft_model_path": "JackFram/llama-160m",
        "main_model_path": "JackFram/llama-160m",
        "tokenizer": "JackFram/llama-160m",
        "conversations_path": "datasets/chatbot-arena.parquet",
        "n_conversations": 1000,
        "max_tokens_guess": 16,
        "max_tokens": 256,
        "accepted_tokens_weight": 1.0,
        "rejected_tokens_weight": 1.0,
    },
)
