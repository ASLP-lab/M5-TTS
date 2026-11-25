"""The boundary between rollout, rewards and the GRPO objective."""
from dataclasses import dataclass


@dataclass
class Episode:
    prefix_token_ids: list[list[int]]
    generated_token_ids: list[list[int]]
    group_token_ids: str
    reward: float | list[float]
    reward_info: dict
    prompt_index: int = 0
    # Diagnostic metadata stays outside numeric reward_info and advantage math.
    asr_alignment: dict | None = None
    generation_metadata: dict | None = None
    # Learner/actor-local derived state, recomputed from raw full groups on RPC.
    # Never add numeric arrays to reward_info or normalize this after selection.
    token_cer_advantage: dict | None = None
