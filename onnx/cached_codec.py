"""Export a causal Qwen decoder with explicit, bounded streaming state."""

import json

import torch
from torch import nn

from graphs import export_graph


class CachedCodec(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        config = decoder.pre_transformer.config
        self.layers = config.num_hidden_layers
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.dim = config.head_dim
        self.window = config.sliding_window
        if not self.window:
            raise ValueError("This export requires bounded sliding attention")
        self.specs = []
        for name, module in decoder.named_modules():
            kind = type(module).__name__
            if kind == "Qwen3TTSTokenizerV2CausalConvNet":
                if module.stride != 1:
                    raise ValueError(f"Unsupported strided causal convolution: {name}")
                if module.padding:
                    self.specs.append(dict(name=name, shape=[1, module.conv.in_channels, module.padding]))
            elif kind == "Qwen3TTSTokenizerV2CausalTransConvNet" and module.right_pad:
                self.specs.append(dict(name=name, shape=[1, module.conv.out_channels, module.right_pad]))

    def empty_states(self):
        return [torch.zeros(spec["shape"]) for spec in self.specs]

    def _convolution(self, module, hidden, states, updated):
        kind = type(module).__name__
        if kind == "Qwen3TTSTokenizerV2CausalConvNet":
            if module.padding:
                hidden = torch.cat((next(states), hidden), dim=-1)
                updated.append(hidden[..., -module.padding:])
            return module.conv(hidden)
        # ConvTranspose overlaps adjacent chunks. Cache the unbiased tail so
        # that the next invocation adds the shared output bias exactly once.
        output = module.conv(hidden)
        padding = module.right_pad
        if not padding:
            return output
        previous = next(states)
        bias = module.conv.bias[None, :, None] if module.conv.bias is not None else 0.0
        updated.append(output[..., -padding:] - bias)
        overlap = output[..., :padding] + previous
        return torch.cat((overlap, output[..., padding:-padding]), dim=-1)

    def _block(self, module, hidden, states, updated):
        kind = type(module).__name__
        if kind in ("Qwen3TTSTokenizerV2CausalConvNet", "Qwen3TTSTokenizerV2CausalTransConvNet"):
            return self._convolution(module, hidden, states, updated)
        if kind == "Qwen3TTSTokenizerV2ConvNeXtBlock":
            residual = hidden
            hidden = self._block(module.dwconv, hidden, states, updated).transpose(1, 2)
            hidden = module.pwconv2(module.act(module.pwconv1(module.norm(hidden))))
            return residual + (module.gamma * hidden).transpose(1, 2)
        if kind == "Qwen3TTSTokenizerV2DecoderDecoderResidualUnit":
            residual = hidden
            hidden = self._block(module.conv1, module.act1(hidden), states, updated)
            return residual + self._block(module.conv2, module.act2(hidden), states, updated)
        if kind == "Qwen3TTSTokenizerV2DecoderDecoderBlock":
            for block in module.block:
                hidden = self._block(block, hidden, states, updated)
            return hidden
        return module(hidden)

    def _transformer(self, hidden, offset, past):
        module = self.decoder.pre_transformer
        hidden = module.input_proj(hidden)
        positions = torch.arange(hidden.shape[1], device=hidden.device) + offset
        key_positions = torch.arange(past.shape[-2] + hidden.shape[1], device=hidden.device)
        key_positions = key_positions + offset - past.shape[-2]
        distance = positions[:, None] - key_positions[None, :]
        allowed = (distance >= 0) & (distance < self.window)
        mask = torch.where(allowed, 0.0, torch.finfo(hidden.dtype).min)[None, None]
        cos, sin = module.rotary_emb(hidden, positions[None])
        cos, sin = cos[:, None], sin[:, None]
        present = []
        for index, layer in enumerate(module.layers):
            attn = layer.self_attn
            x = layer.input_layernorm(hidden)
            q = attn.q_norm(attn.q_proj(x).reshape(1, -1, self.heads, self.dim)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(x).reshape(1, -1, self.kv_heads, self.dim)).transpose(1, 2)
            v = attn.v_proj(x).reshape(1, -1, self.kv_heads, self.dim).transpose(1, 2)
            q_half = torch.cat((-q[..., self.dim // 2:], q[..., :self.dim // 2]), dim=-1)
            k_half = torch.cat((-k[..., self.dim // 2:], k[..., :self.dim // 2]), dim=-1)
            q, k = q * cos + q_half * sin, k * cos + k_half * sin
            k = torch.cat((past[index, 0], k), dim=-2)
            v = torch.cat((past[index, 1], v), dim=-2)
            present.append(torch.stack((k[..., -(self.window - 1):, :], v[..., -(self.window - 1):, :])))
            repeats = self.heads // self.kv_heads
            keys = k[:, :, None].expand(-1, -1, repeats, -1, -1).reshape(1, self.heads, -1, self.dim)
            values = v[:, :, None].expand(-1, -1, repeats, -1, -1).reshape(1, self.heads, -1, self.dim)
            weights = ((q @ keys.transpose(-1, -2)) * attn.scaling + mask).softmax(-1)
            x = (weights @ values).transpose(1, 2).reshape(1, -1, self.heads * self.dim)
            hidden = hidden + layer.self_attn_layer_scale(attn.o_proj(x))
            hidden = hidden + layer.mlp_layer_scale(layer.mlp(layer.post_attention_layernorm(hidden)))
        return module.output_proj(module.norm(hidden)), torch.stack(present)

    def forward(self, codes, offset, past, *convolution_states):
        states, updated = iter(convolution_states), []
        hidden = self.decoder.quantizer.decode(codes)
        hidden = self._block(self.decoder.pre_conv, hidden, states, updated).transpose(1, 2)
        hidden, present = self._transformer(hidden, offset, past)
        hidden = hidden.transpose(1, 2)
        for blocks in self.decoder.upsample:
            for block in blocks:
                hidden = self._block(block, hidden, states, updated)
        for block in self.decoder.decoder:
            hidden = self._block(block, hidden, states, updated)
        return (hidden.clamp(-1, 1), present, *updated)


def export_cached_codec(codec, output):
    decoder = CachedCodec(codec.decoder).eval()
    inputs = ["codes", "offset", "past"] + [f"conv_{i}" for i in range(len(decoder.specs))]
    outputs = ["audio", "present"] + [f"next_conv_{i}" for i in range(len(decoder.specs))]
    axes = {"codes": {2: "frames"}, "audio": {2: "samples"},
            "past": {4: "past_length"}, "present": {4: "present_length"}}
    with torch.inference_mode():
        past = torch.zeros(decoder.layers, 2, 1, decoder.kv_heads, 3, decoder.dim)
        export_graph(decoder, (torch.zeros(1, 16, 4, dtype=torch.long), torch.tensor([3]),
                               past, *decoder.empty_states()), output / "codec_decoder_cached.onnx",
                     inputs, outputs, axes)
    metadata = dict(format_version=1, layers=decoder.layers, kv_heads=decoder.kv_heads,
                    head_dim=decoder.dim, window=decoder.window, states=decoder.specs,
                    inputs=inputs, outputs=outputs)
    (output / "codec_cached.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return decoder
