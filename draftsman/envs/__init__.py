import gym

from .spec import SpeculativeDecoding

gym.register(
    id="SpeculativeDecoding-v0",
    entry_point=SpeculativeDecoding,
    kwargs={
        "draft_model_path": "JackFram/llama-160m",
        "main_model_path": "meta-llama/Llama-2-7b-hf",
        "tokenizer": "JackFram/llama-160m",
        "conversations_path": "datasets/chatbot_arena_2.parquet",
        "n_conversations": 1000,
        "max_tokens_guess": 16,
        "max_tokens": 256,
        "tensor_parallel_size": 1,
        "accepted_tokens_weight": 1.0,
        "rejected_tokens_weight": 1.0,
    },
)

# speculative decoding for running on an 8x2080 Ti machine
gym.register(
    id="SpeculativeDecodingHoneydew-v0",
    entry_point=SpeculativeDecoding,
    kwargs={
        "draft_model_path": "JackFram/llama-160m",
        "main_model_path": "meta-llama/Llama-2-7b-hf",
        "tokenizer": "JackFram/llama-160m",
        "conversations_path": "datasets/chatbot_arena_2.parquet",
        "n_conversations": 1000,
        "max_tokens_guess": 16,
        "max_tokens": 256,
        "tensor_parallel_size": 2,
        "accepted_tokens_weight": 1.0,
        "rejected_tokens_weight": 1.0,
    },
)

# speculative decoding covering the entire dataset
gym.register(
    id="SpeculativeDecodingMany-v0",
    entry_point=SpeculativeDecoding,
    kwargs={
        "draft_model_path": "JackFram/llama-160m",
        "main_model_path": "meta-llama/Llama-2-7b-hf",
        "tokenizer": "JackFram/llama-160m",
        "conversations_path": "datasets/chatbot_arena_2.parquet",
        "n_conversations": 20000,
        "max_tokens_guess": 16,
        "max_tokens": 256,
        "tensor_parallel_size": 1,
        "accepted_tokens_weight": 1.0,
        "rejected_tokens_weight": 1.0,
    },
)
