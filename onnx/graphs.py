"""Explicit FP32 M5 and Qwen codec graphs with tensor-only cache interfaces."""

from types import SimpleNamespace

import torch
from torch import nn


class LlamaCore(nn.Module):
    """Explicit tensor KV state avoids exporting HF cache mutation/control flow."""

    def __init__(self, decoder, use_sdpa=False, fuse_projections=False):
        super().__init__()
        self.layers = decoder.layers
        self.norm = decoder.norm
        self.register_buffer("inv_freq", decoder.position_embedding.inv_freq.clone())
        self.rope_scale = decoder.position_embedding.attention_scaling
        self.heads = decoder.config.num_attention_heads
        self.kv_heads = decoder.config.num_key_value_heads
        self.dim = decoder.config.hidden_size // self.heads
        self.use_sdpa = use_sdpa
        self.qkv = nn.ModuleList() if fuse_projections else None
        self.gate_up = nn.ModuleList() if fuse_projections else None
        if fuse_projections:
            for layer in self.layers:
                attn = layer.self_attn
                self.qkv.append(self.combine_linears(attn.q_proj, attn.k_proj, attn.v_proj))
                self.gate_up.append(self.combine_linears(layer.mlp.gate_proj, layer.mlp.up_proj))

    @staticmethod
    def combine_linears(*modules):
        weight = torch.cat([module.weight.detach() for module in modules], dim=0)
        has_bias = any(module.bias is not None for module in modules)
        combined = nn.Linear(weight.shape[1], weight.shape[0], bias=has_bias,
                             device=weight.device, dtype=weight.dtype)
        with torch.no_grad():
            combined.weight.copy_(weight)
            if has_bias:
                combined.bias.copy_(torch.cat([
                    module.bias.detach() if module.bias is not None else
                    torch.zeros(module.out_features, device=weight.device, dtype=weight.dtype)
                    for module in modules
                ]))
        return combined

    def empty_cache(self):
        return torch.empty(len(self.layers), 2, 1, self.kv_heads, 0, self.dim)

    def forward(self, hidden, positions, past):
        freqs = positions.float().unsqueeze(-1) * self.inv_freq
        freqs = torch.cat((freqs, freqs), dim=-1)
        cos = (freqs.cos() * self.rope_scale).to(hidden.dtype).unsqueeze(1)
        sin = (freqs.sin() * self.rope_scale).to(hidden.dtype).unsqueeze(1)
        key_pos = torch.arange(past.shape[-2] + hidden.shape[1], device=hidden.device)
        allowed = key_pos[None, None, None, :] <= positions[:, None, :, None]
        present = []
        for index, layer in enumerate(self.layers):
            attn = layer.self_attn
            x = layer.input_layernorm(hidden)
            if self.qkv is None:
                q, k, v = attn.q_proj(x), attn.k_proj(x), attn.v_proj(x)
            else:
                q, k, v = self.qkv[index](x).split(
                    [self.heads * self.dim, self.kv_heads * self.dim, self.kv_heads * self.dim], dim=-1
                )
            q = q.reshape(1, -1, self.heads, self.dim).transpose(1, 2)
            k = k.reshape(1, -1, self.kv_heads, self.dim).transpose(1, 2)
            v = v.reshape(1, -1, self.kv_heads, self.dim).transpose(1, 2)
            q_half = torch.cat((-q[..., self.dim // 2:], q[..., :self.dim // 2]), dim=-1)
            k_half = torch.cat((-k[..., self.dim // 2:], k[..., :self.dim // 2]), dim=-1)
            q = q * cos + q_half * sin
            k = k * cos + k_half * sin
            k = torch.cat((past[index, 0], k), dim=-2)
            v = torch.cat((past[index, 1], v), dim=-2)
            present.append(torch.stack((k, v)))
            repeats = self.heads // self.kv_heads
            keys = k[:, :, None].expand(-1, -1, repeats, -1, -1).reshape(1, self.heads, -1, self.dim)
            vals = v[:, :, None].expand(-1, -1, repeats, -1, -1).reshape(1, self.heads, -1, self.dim)
            if self.use_sdpa:
                out = torch.nn.functional.scaled_dot_product_attention(q, keys, vals, attn_mask=allowed)
            else:
                scores = (q @ keys.transpose(-1, -2)) * (self.dim ** -0.5)
                scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
                out = scores.softmax(-1) @ vals
            out = out.transpose(1, 2).reshape(1, -1, self.heads * self.dim)
            hidden = hidden + attn.o_proj(out)
            x = layer.post_attention_layernorm(hidden)
            if self.gate_up is None:
                hidden = hidden + layer.mlp(x)
            else:
                gate, up = self.gate_up[index](x).chunk(2, dim=-1)
                hidden = hidden + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)
        return self.norm(hidden), torch.stack(present)


class Backbone(nn.Module):
    def __init__(self, model, fuse_projections=False):
        super().__init__()
        self.core = LlamaCore(model.backbone, fuse_projections=fuse_projections)

    def forward(self, hidden, positions, past):
        hidden, present = self.core(hidden, positions, past)
        return hidden[:, -1], present


class DepthFrame(nn.Module):
    """All sixteen codebooks in one ORT invocation, retaining local KV state."""

    def __init__(self, model, topk=10, temperature=0.9, use_sdpa=False, fuse_projections=False,
                 fast_sample=False):
        super().__init__()
        self.core = LlamaCore(model.decoder, use_sdpa=use_sdpa, fuse_projections=fuse_projections)
        self.projection = model.projection
        self.head0 = model.codebook0_head
        self.audio_head = model.audio_head
        self.audio_embeddings = model.audio_embeddings
        self.vocab = model.args.audio_vocab_size
        self.codebooks = model.args.use_codebooks
        self.topk = topk
        self.temperature = temperature
        self.fast_sample = fast_sample
        # Projection is linear and bias-free: precompute the repeated lookup work.
        self.register_buffer("projected_audio", model.projection(model.audio_embeddings.weight).detach())

    def sample(self, logits, noise, allow_eos):
        forbidden = (torch.arange(self.vocab, device=logits.device) == 0) & ~allow_eos
        logits = logits.masked_fill(forbidden, -1e5) / self.temperature
        cutoff = torch.topk(logits, self.topk, dim=-1).values[:, -1:]
        scores = logits.masked_fill(logits < cutoff, -float("inf"))
        if self.fast_sample:
            # In FP32, argmax(softmax(scores) / noise) is exactly argmax(scores - log(noise)).
            # Keep this opt-in: reduced-precision softmax changes tie-breaking.
            return (scores - torch.log(noise)).argmax(-1, keepdim=True)
        probs = torch.softmax(scores, dim=-1)
        return (probs / noise).argmax(-1, keepdim=True)

    def forward(self, hidden, noise, allow_eos):
        c0 = self.sample(self.head0(hidden), noise[0], allow_eos)
        codes = [c0]
        local = torch.cat((self.projection(hidden[:, None]), self.projected_audio[c0]), dim=1)
        past = self.core.empty_cache().to(hidden)
        pos = torch.arange(2, device=hidden.device)[None]
        for index in range(1, self.codebooks):
            decoded, past = self.core(local, pos, past)
            logits = decoded[:, -1] @ self.audio_head[index - 1]
            code = self.sample(logits, noise[index], torch.ones_like(allow_eos))
            codes.append(code)
            local = self.projected_audio[code + index * self.vocab]
            pos = pos[:, -1:] + 1
        codes = torch.cat(codes, dim=1)
        offsets = torch.arange(self.codebooks, device=hidden.device) * self.vocab
        next_hidden = self.audio_embeddings(codes + offsets).sum(dim=1, keepdim=True)
        return codes, next_hidden


class CodecTransformer(nn.Module):
    """An explicit causal mask bypasses non-exportable HF mask construction."""

    def __init__(self, original, qwen):
        super().__init__()
        self.original = original
        self.qwen = qwen

    def forward(self, hidden_states=None, inputs_embeds=None, **kwargs):
        module = self.original
        x = module.input_proj(inputs_embeds) if self.qwen else hidden_states
        positions = torch.arange(x.shape[1], device=x.device)
        distance = positions[:, None] - positions[None, :]
        allowed = distance >= 0
        # Mimi's installed SDPA implementation uses create_causal_mask (full).
        window = getattr(module.config, "sliding_window", None) if self.qwen else None
        if window:
            allowed = allowed & (distance < window)
        mask = torch.where(allowed, 0.0, torch.finfo(x.dtype).min)[None, None]
        if self.qwen:
            rotary = module.rotary_emb(x, positions[None])
        for layer in module.layers:
            args = dict(attention_mask=mask, position_ids=positions[None], use_cache=False)
            if self.qwen:
                args["position_embeddings"] = rotary
            out = layer(x, **args)
            x = out[0] if isinstance(out, tuple) else out
        if self.qwen:
            return SimpleNamespace(last_hidden_state=module.output_proj(module.norm(x)))
        return (x,)


class CodecEncoder(nn.Module):
    def __init__(self, codec):
        super().__init__()
        self.encoder = codec.encoder

    def forward(self, audio):
        x = self.encoder.encoder(audio[:, None], padding_cache=None)
        x = self.encoder.encoder_transformer(x.transpose(1, 2))[0].transpose(1, 2)
        x = self.encoder.downsample(x, padding_cache=None)
        return self.encoder.quantizer.encode(x, 16).transpose(0, 1)


def causal_extra_padding(module, hidden):
    # kernel - stride left padding means the required right padding is -L % stride.
    return (-hidden.shape[-1]) % module.stride


def export_graph(module, inputs, path, names, outputs, axes):
    import onnx
    if not hasattr(onnx, "load"):
        raise ImportError("Install the third-party onnx package; do not add __init__.py to the onnx/ scripts directory")
    torch.onnx.export(module.eval(), inputs, str(path), input_names=names,
                      output_names=outputs, dynamic_axes=axes, opset_version=17,
                      dynamo=False, do_constant_folding=True)
    onnx.checker.check_model(str(path))
    graph = onnx.load(str(path), load_external_data=False)
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):
        raise ValueError("External-data graphs are not supported by the current infer_onnx.py download contract")
    print(f"Exported and checked {path.name}", flush=True)


