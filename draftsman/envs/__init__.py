import gym

from .spec import SpeculativeDecoding

gym.register(
    id="SpeculativeDecodingLog-v1",
    entry_point=SpeculativeDecoding,
    kwargs={
        "conversations_path": "datasets/temp.parquet",
        "n_conversations": 64,
        "conversation_offset": 3762,
        "max_tokens_guess": 16,
        "max_tokens": 256,
        "accepted_tokens_weight": 1.0,
        "rejected_tokens_weight": 1.0,
        "logarithmic": True,
        "use_main_hidden_states": False,
        "use_draft_hidden_states": True,
        "one_hot_encode_prev": False
    },
)
