import gym

from .spec import SpeculativeDecoding

gym.register(
    id="SpeculativeDecodingLog-v1",
    entry_point=SpeculativeDecoding,
    kwargs={
        "conversations_paths": ["datasets/out_0.parquet", "datasets/out_1.parquet"],
        "n_conversations": 5000,
        "conversation_offset": 0,
        "max_tokens_guess": 16,
        "max_tokens": 192,
        "accepted_tokens_weight": 1.0,
        "rejected_tokens_weight": 1.0,
        "logarithmic": False,
        "use_main_hidden_states": False,
        "use_draft_hidden_states": True,
        "one_hot_encode_prev": False
    },
)

gym.register(
    id="SpeculativeDecodingLogTest-v1",
    entry_point=SpeculativeDecoding,
    kwargs={
        "conversations_paths": ["datasets/out_2.parquet", "datasets/out_3.parquet"],
        "n_conversations": 4000,
        "conversation_offset": 0,
        "max_tokens_guess": 16,
        "max_tokens": 192,
        "accepted_tokens_weight": 1.0,
        "rejected_tokens_weight": 1.0,
        "logarithmic": False,
        "use_main_hidden_states": False,
        "use_draft_hidden_states": True,
        "one_hot_encode_prev": False
    },
)

gym.register(
    id="SpeculativeDecodingLog-v2",
    entry_point=SpeculativeDecoding,
    kwargs={
        "conversations_paths": ["datasets/out_0.parquet", "datasets/out_1.parquet"],
        "n_conversations": 5000,
        "conversation_offset": 0,
        "max_tokens_guess": 16,
        "max_tokens": 192,
        "accepted_tokens_weight": 1.0,
        "rejected_tokens_weight": 1.0,
        "logarithmic": True,
        "use_main_hidden_states": True,
        "use_draft_hidden_states": True,
        "one_hot_encode_prev": False
    },
)

gym.register(
    id="SpeculativeDecodingLogTest-v2",
    entry_point=SpeculativeDecoding,
    kwargs={
        "conversations_paths": ["datasets/out_2.parquet", "datasets/out_3.parquet"],
        "n_conversations": 5000,
        "conversation_offset": 0,
        "max_tokens_guess": 16,
        "max_tokens": 192,
        "accepted_tokens_weight": 1.0,
        "rejected_tokens_weight": 1.0,
        "logarithmic": True,
        "use_main_hidden_states": True,
        "use_draft_hidden_states": True,
        "one_hot_encode_prev": False
    },
)

gym.register(
    id="SpeculativeDecodingLog-v3",
    entry_point=SpeculativeDecoding,
    kwargs={
        "conversations_paths": ["datasets/out_0.parquet", "datasets/out_1.parquet"],
        "n_conversations": 5000,
        "conversation_offset": 0,
        "max_tokens_guess": 16,
        "max_tokens": 192,
        "accepted_tokens_weight": 1.0,
        "rejected_tokens_weight": 1.0,
        "logarithmic": True,
        "use_main_hidden_states": True,
        "use_draft_hidden_states": True,
        "one_hot_encode_prev": True
    },
)

gym.register(
    id="SpeculativeDecodingLogTest-v3",
    entry_point=SpeculativeDecoding,
    kwargs={
        "conversations_paths": ["datasets/out_2.parquet", "datasets/out_3.parquet"],
        "n_conversations": 5000,
        "conversation_offset": 0,
        "max_tokens_guess": 16,
        "max_tokens": 192,
        "accepted_tokens_weight": 1.0,
        "rejected_tokens_weight": 1.0,
        "logarithmic": True,
        "use_main_hidden_states": True,
        "use_draft_hidden_states": True,
        "one_hot_encode_prev": True
    },
)
