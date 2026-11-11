import torch
from transformers.models.llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding, LlamaRMSNorm
from typing import Optional, Union, List

    
class TransformerDecoder(torch.nn.Module):
    def __init__(
        self,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32.0,
        attention_type: str="sdpa"
        ):
        super().__init__()
        
        self.config = LlamaConfig(
            hidden_size=embed_dim,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            max_position_embeddings=max_seq_len,
            intermediate_size=intermediate_dim,
            attention_dropout=attn_dropout,
            rms_norm_eps=norm_eps,
            rope_theta=rope_base,
            rope_scaling={
                        'rope_type':'llama3',
                        'factor':scale_factor,
                        'high_freq_factor': 4.0,
                        'low_freq_factor': 1.0,
                        'original_max_position_embeddings': max_seq_len
                        }
        )
        self.config._attn_implementation = attention_type
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList([LlamaDecoderLayer(self.config, i) for i in range(num_layers)])
        self.position_embedding = LlamaRotaryEmbedding(config=self.config)
        self.norm = LlamaRMSNorm(self.config.hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        past_key_values = None,
        use_cache: Optional[bool] = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            tokens (torch.Tensor): input tensor with shape ``[B, T, C]``
            mask (Optional[_MaskType]): Used to mask the scores after the query-key multiplication
                and before the softmax. This parameter is required during inference if caches have been setup.
                Either:

                A boolean tensor with shape ``[b x s x s]``, ``[b x s x self.encoder_max_cache_seq_len]``,
                or ``[b x s x self.encoder_max_cache_seq_len]`` if using KV-cacheing with encoder/decoder layers.
                A value of True in row ``i`` and column ``j`` means token ``i`` attends to token ``j``. A value of False means
                token ``i`` does not attend to token ``j``. If no mask is specified, a causal mask
                is used by default.

                A :class:`~torch.nn.attention.flex_attention.BlockMask` for document masking in a packed sequence
                created via `create_block_mask <https://pytorch.org/blog/flexattention/#mask-mods>`_. We  use
                :func:`~torch.nn.attention.flex_attention.flex_attention` when computing attention with block masks.
                Default is None.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape ``[b x s]``.
                During inference, this indicates the position of the current token.
                This parameter is required during inference if caches have been setup. Default is None.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: output tensor with shape ``[b x s x v]`` or a list of layer
                output tensors defined by ``output_hidden_states`` with the
                final output tensor appended to the list.

        Note:
            At the very first step of inference, when the model is provided with a prompt,
            ``input_pos`` should contain the positions of all of the tokens in the prompt.
            For a single-batch prompt, or a batch of prompts with identical lengths, this
            will be ``torch.arange(prompt_length)``. For a batch of varying-length prompts,
            shorter prompts are left-padded and position ids are correspondingly right-shifted,
            thus positional ids should be of shape ``[b, padded_prompt_length]``.
            This is because we will need to retrieve the positional embeddings for each input id.
            In the subsequent steps, if the model has been setup with KV-caches, ``input_pos`` will contain
            the position(s) of the current token(s) ``torch.tensor([padded_prompt_length])``. Otherwise,
            ``input_pos`` will contain all the position ids up to the current token.

        Shape notation:
            - b: batch size
            - s: token sequence length
            - s_e: encoder sequence length
            - v: vocab size
            - d: token embed dim
            - d_e: encoder embed dim
            - m_s: max seq len
        """
        pos_embed = self.position_embedding(hidden_states, position_ids)
        if attention_mask is None and self.config._attn_implementation == "sdpa":
            past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
            keys = torch.arange(past_length + hidden_states.size(1), device=hidden_states.device)
            attention_mask = keys[None, None, None, :] <= position_ids[:, None, :, None]

        for i, layer in enumerate(self.layers):
            layer_out = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_embeddings=pos_embed,
                use_cache=use_cache,
                past_key_value=past_key_values
            )
            # transformers 4.43+ 的 LlamaDecoderLayer 返回 (hidden_states, ...) tuple
            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        # shape: [b, s, d]
        hidden_states = self.norm(hidden_states)
        return hidden_states


def llama3_2_1B(attention_type) -> TransformerDecoder:
    return TransformerDecoder(
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32.,
        attention_type=attention_type
    )


def llama3_2_500M(attention_type) -> TransformerDecoder:
    return TransformerDecoder(
        num_layers=8,
        num_heads=16,
        num_kv_heads=4,
        embed_dim=2048,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32.,
        attention_type=attention_type
    )

def llama3_2_250M(attention_type) -> TransformerDecoder:
    return TransformerDecoder(
        num_layers=6,
        num_heads=12,
        num_kv_heads=3,
        embed_dim=1536,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32.,
        attention_type=attention_type
    )


def llama3_2_100M(attention_type) -> TransformerDecoder:
    return TransformerDecoder(
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        embed_dim=1024,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32.,
        attention_type=attention_type
    )


def llama3_2_60M(attention_type) -> TransformerDecoder:
    return TransformerDecoder(
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        embed_dim=1024,
        max_seq_len=2048,
        intermediate_dim=4096,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32.,
        attention_type=attention_type
    )

def llama3_2_30M(attention_type) -> TransformerDecoder:
    return TransformerDecoder(
        num_layers=8,
        num_heads=8,
        num_kv_heads=2,
        embed_dim=512,
        max_seq_len=2048,
        intermediate_dim=2048,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32.,
        attention_type=attention_type
    )




def llama3_2_20M(attention_type) -> TransformerDecoder:
    return TransformerDecoder(
        num_layers=5,
        num_heads=8,
        num_kv_heads=2,
        embed_dim=512,
        max_seq_len=2048,
        intermediate_dim=2048,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32.,
        attention_type=attention_type
    )



FLAVORS = {
    "llama-1B": llama3_2_1B,
    "llama-500M": llama3_2_500M,
    "llama-250M": llama3_2_250M,
    "llama-100M": llama3_2_100M,
    "llama-60M": llama3_2_60M,
    "llama-30M": llama3_2_30M,
    "llama-20M": llama3_2_20M,
}
