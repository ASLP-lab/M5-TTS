from __future__ import annotations
import math
import torch

from transformers.cache_utils import DynamicCache
from .flavors import FLAVORS
from dataclasses import dataclass
import time

@dataclass
class ModelArgs:
    backbone_flavor: str
    decoder_flavor: str
    text_vocab_size: int
    audio_vocab_size: int
    audio_num_codebooks: int
    use_codebooks: int = 16
    pad_multiple: int = 32
    attention_type: str = "sdpa"
    spk_emb_dim: int = 192
    spk_drop_prob: float = 0.2
    lambda_depth: float = 1.0


"flash_attention_3"
"flash_attention_2"
"flex_attention"
"paged_attention"
"sdpa"
"sdpa_paged"
"eager_paged"
"""
sdpa: attention_mask [B, 1, q_len, kv_len] 
mask: [B, L] -> mask[:, None, :, None] * mask[:, None, None, :] | torch.eye(L)[None, None, :, :]

flash_attention: attention_mask [B, 1, q_len]

"""


def _create_causal_mask(seq_len: int, device: torch.device):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    
def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor):
    """
    Args:
        mask: (max_seq_len, max_seq_len)
        input_pos: (batch_size, seq_len)

    Returns:
        (batch_size, seq_len, max_seq_len)
    """
    r = mask[input_pos, :]
    return r

def _multinomial_sample_one_no_sync(probs):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)

def sample_topk(logits: torch.Tensor, topk: int, temperature: float, need_finish: bool):
    if need_finish:
        pass
    else:
        logits[:,0] = -1e5
    logits = logits / temperature

    filter_value: float = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)

    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token

class CSM(torch.nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        
        self.backbone = FLAVORS[args.backbone_flavor](args.attention_type)
        self.decoder = FLAVORS[args.decoder_flavor](args.attention_type)
        backbone_dim = self.backbone.config.hidden_size
        decoder_dim = self.decoder.config.hidden_size

        self.text_embeddings = torch.nn.Embedding(args.text_vocab_size, backbone_dim)
        torch.nn.init.normal_(self.text_embeddings.weight, mean=0.0, std=1.0 / math.sqrt(self.text_embeddings.embedding_dim))

        self.audio_embeddings = torch.nn.Embedding(args.audio_vocab_size * args.audio_num_codebooks, backbone_dim)
        torch.nn.init.normal_(self.audio_embeddings.weight, mean=0.0, std=1.0 / math.sqrt(self.audio_embeddings.embedding_dim))

        self.projection = torch.nn.Linear(backbone_dim, decoder_dim, bias=False)
        torch.nn.init.normal_(self.projection.weight, mean=0.0, std=1.0 / math.sqrt(self.projection.in_features))

        self.codebook0_head = torch.nn.Linear(backbone_dim, args.audio_vocab_size, bias=False)
        torch.nn.init.zeros_(self.codebook0_head.weight)

        self.audio_head = torch.nn.Parameter(torch.empty(args.audio_num_codebooks - 1, decoder_dim, args.audio_vocab_size))
        torch.nn.init.zeros_(self.audio_head)

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        return self.audio_embeddings(tokens + codebook * self.args.audio_vocab_size)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        text_embeds = self.text_embeddings(tokens[:, :, -1]).unsqueeze(-2)

        audio_tokens = tokens[:, :, :-1] + (self.args.audio_vocab_size * torch.arange(self.args.use_codebooks, device=tokens.device))
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.args.use_codebooks, -1
        )

        return torch.cat([audio_embeds, text_embeds], dim=-2)
        
    def forward(
        self,
        tokens,  # (B,S,C)
        tokens_mask,  # (B,S,C)
        decoder_fraction=1 / 16,
    ):
        dtype, device = next(self.parameters()).dtype, tokens.device
        b, s, c = tokens.shape
        self.args.use_codebooks = c - 1
        n = self.args.audio_num_codebooks
        if s < 2:
            raise ValueError("sequence must contain at least two frames for next‑frame training")
        tokens_in = tokens[:, :-1, :]  # (b, s-1, c)
        tokens_mask_in = tokens_mask[:, :-1, :]  # (b, s-1, c)
        embeds_in = self._embed_tokens(tokens_in)  # (b, s-1, c, dim)
        masked_embeds = (embeds_in * tokens_mask_in.unsqueeze(-1)).sum(dim=2)  # (b, s-1, dim)
        
        input_pos = torch.arange(s - 1, device=device).unsqueeze(0).expand(b, s - 1)  # (b, s-1)
        if self.args.attention_type == "flash_attention_2":
            seq_len = tokens_mask_in[:, :, -2:].sum(dim=-1).sum(dim=-1)
            curr_backbone_mask = torch.arange(0, seq_len.max(), device=seq_len.device)[None, :].expand(b, seq_len.max())
            curr_backbone_mask = curr_backbone_mask < seq_len[:, None] # [b, 1, s]
        else:
            curr_backbone_mask = _create_causal_mask(s - 1, masked_embeds.device).unsqueeze(1) # [b, 1, s, s]
        
        h = self.backbone(
            hidden_states=masked_embeds,
            attention_mask=curr_backbone_mask,
            position_ids=input_pos
        ).to(dtype=dtype)  # (b, s, dim)

        # codebook 0 (semantic) loss
        tokens_tgt = tokens[:, 1:, :]  # (b, s-1, c)
        tokens_mask_tgt = tokens_mask[:, 1:, :]  # (b, s-1, c)

        c0_mask = tokens_mask_tgt[:, :, 0]  # (b, s-1) 
        c0_h = h[c0_mask]  # (m, dim)
        c0_tgt = tokens_tgt[:, :, 0][c0_mask]  # (m,)
        c0_logits = self.codebook0_head(c0_h)  # (m, v_a)
        c0_loss = torch.nn.functional.cross_entropy(c0_logits, c0_tgt, reduction="mean")
        c0_accuracy = (c0_logits.argmax(dim=-1) == c0_tgt).float().mean()

        # decoder
        frame_idx = torch.nonzero(c0_mask, as_tuple=False)  # (m, 2)
        keep = torch.rand(len(frame_idx), device=device) < decoder_fraction
        sel_idx = frame_idx[keep] # (m * decoder_fraction, 2)

        if sel_idx.numel() == 0:
            decoder_loss = torch.zeros((), device=device, dtype=dtype)
            per_level_losses = []
            decoder_h = None

        else:
            sb, ss = sel_idx[:, 0], sel_idx[:, 1]
            last_h = h[sb, ss]  # (msel, dim)
            tgt_row = tokens_tgt[sb, ss]  # (msel, c)
            c0_gt = tgt_row[:, 0]  # (msel,)
            audio_gt = tgt_row[:, 1:-1]  # (msel, n-1)

            # [h_t, c0, c1 … c_{n-2}]
            seq_parts = [
                last_h.unsqueeze(1),
                self._embed_audio(0, c0_gt).unsqueeze(1),
            ]
            for k in range(1, self.args.use_codebooks - 1):
                seq_parts.append(self._embed_audio(k, audio_gt[:, k - 1]).unsqueeze(1))

            dec_in = torch.cat(seq_parts, dim=1)  # (msel, n, dim_backbone)
            proj_h = self.projection(dec_in)  # (msel, n, dim_decoder)

            pos = torch.arange(self.args.use_codebooks, device=device).unsqueeze(0).expand(proj_h.size(0), -1)
            if self.args.attention_type == "flash_attention_2":
                dec_mask = pos < self.args.use_codebooks
            else:
                dec_mask = _create_causal_mask(self.args.use_codebooks, proj_h.device).unsqueeze(1)
            decoder_h = self.decoder(
                hidden_states=proj_h, 
                position_ids=pos, 
                attention_mask=dec_mask
            ).to(dtype=dtype)   # (msel, n, dim_dec)

            dec_h_no_c0 = decoder_h[:, 1:, :] # (msel, n-1, dim_dec)
            logits = torch.einsum("bkd,kdv->bkv", dec_h_no_c0, self.audio_head[:self.args.use_codebooks - 1, :, :])
            token_loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                audio_gt.reshape(-1),
                reduction="none",
            ).view(logits.size(0), logits.size(1))
            
            per_level_losses = token_loss.mean(dim=0)
            num_levels = per_level_losses.shape[0]
            weights = torch.linspace(
                1,
                0.1,
                num_levels+1,
                device=per_level_losses.device,
            )
            weights = weights[1:]
            weights = weights / weights.sum()

            decoder_loss = (per_level_losses * weights).sum()

        weighted_decoder_loss = self.args.lambda_depth * decoder_loss
        total_loss = c0_loss + weighted_decoder_loss
        return {
            "loss": total_loss,
            "backbone_loss": c0_loss.detach(),
            "decoder_loss": decoder_loss.detach(),
            "weighted_decoder_loss": weighted_decoder_loss.detach(),
            "lambda_depth": self.args.lambda_depth,
            "c0_accuracy": c0_accuracy.detach(),
            "per_level_losses": [p.detach() for p in per_level_losses],
            "decoder_h":  decoder_h,
            "last_h": h
        }
        
    
    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        backbone_cache: DynamicCache,
        input_pos: torch.Tensor,
        temperature: float,
        topk: int,
        need_finish:bool,
        do_sample: bool= False,
    ) -> torch.Tensor:
        """
        Args:
            tokens: (batch_size, seq_len, audio_num_codebooks+1)
            tokens_mask: (batch_size, seq_len, audio_num_codebooks+1)
            input_pos: (batch_size, seq_len) positions for each token
            mask: (batch_size, seq_len, max_seq_len

        Returns:
            (batch_size, audio_num_codebooks) sampled tokens
        """
        dtype = next(self.parameters()).dtype
        b, s, c = tokens.size()
        curr_backbone_mask = None
        masked_embeds = self._embed_tokens(tokens)
        masked_embeds = masked_embeds * tokens_mask.unsqueeze(-1)
        h = masked_embeds.sum(dim=2) 
     

        h = self.backbone(
            hidden_states=h, 
            position_ids=input_pos, 
            attention_mask=curr_backbone_mask,
            use_cache=True,
            past_key_values=backbone_cache
        ).to(dtype=dtype)

        last_h = h[:, -1, :]
        c0_logits = self.codebook0_head(last_h)
        if do_sample:
            c0_sample = sample_topk(c0_logits, topk, temperature, need_finish) # b,1
        else:
            c0_sample = torch.argmax(c0_logits).unsqueeze(0).unsqueeze(0)
        c0_embed = self._embed_audio(0, c0_sample) # b,1,emb_dim

        curr_h = torch.cat([last_h.unsqueeze(1), c0_embed], dim=1) # b, 2, emb_dim
        curr_sample = c0_sample.clone() 
        curr_pos = torch.arange(0, curr_h.size(1), device=curr_h.device).unsqueeze(0).repeat(curr_h.size(0), 1)

        # Decoder caches must be reset every frame.
        decoder_cache = DynamicCache(config=self.decoder.config) # craete cache
        cost_decoder_times = []
        for i in range(1, self.args.use_codebooks):

            dec_mask = None
            decoder_proj_h = self.projection(curr_h)
            start_decoder_time = time.time()
            decoder_h = self.decoder(
                hidden_states=decoder_proj_h, 
                position_ids=curr_pos, 
                attention_mask=dec_mask,
                use_cache=True,
                past_key_values=decoder_cache
            ).to(dtype=dtype)
            end_decoder_time = time.time()
            cost_decoder_time = end_decoder_time - start_decoder_time
            cost_decoder_times.append(cost_decoder_time)
            ci_logits = torch.mm(decoder_h[:, -1, :], self.audio_head[i - 1])
            
            if do_sample:
                ci_sample = sample_topk(ci_logits, topk, temperature, True)
            else:
                ci_sample = torch.argmax(ci_logits).unsqueeze(0).unsqueeze(0)
            ci_embed = self._embed_audio(i, ci_sample)

            curr_h = ci_embed
            curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
            curr_pos = curr_pos[:, -1:] + 1


        return curr_sample

    def load_weights(self, weights):
        self.model.load_state_dict(weights, strict=True)

