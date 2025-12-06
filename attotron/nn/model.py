import os

import torch.nn as nn
import torch.nn.functional as F
from flash_attn.flash_attn_interface import flash_attn_func
from flash_attn.layers.rotary import apply_rotary_emb

from attotron import pgm
from attotron.context_parallel import ring_attention, update_rope_for_context_parallel
from attotron.nn.norm import FlashRMSNorm, LlamaRMSNorm
from attotron.nn.rope import get_cos_sin, llama_rotary_emb


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_values = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        assert config.num_attention_heads % pgm.pgm.tp_world_size == 0, (
            "num_attention_heads should be divisible by tp_world_size"
        )
        assert config.num_key_value_heads % pgm.pgm.tp_world_size == 0, (
            "num_key_value_heads should be divisible by tp_world_size"
        )
        self.num_local_heads = config.num_attention_heads // pgm.pgm.tp_world_size
        self.num_local_kv_heads = config.num_key_value_heads // pgm.pgm.tp_world_size

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_values * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_values * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x, cos, sin):
        batch_size, seq_len, _ = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.view(batch_size, seq_len, self.num_local_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_local_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_local_kv_heads, self.head_dim)

        if os.getenv("FLASH_ATTN", "1") != "1":
            q = llama_rotary_emb(q, cos, sin)
            k = llama_rotary_emb(k, cos, sin)
        else:
            q = apply_rotary_emb(q, cos[:, : self.head_dim // 2], sin[:, : self.head_dim // 2])
            k = apply_rotary_emb(k, cos[:, : self.head_dim // 2], sin[:, : self.head_dim // 2])

        k = k.repeat_interleave(self.num_local_heads // self.num_local_kv_heads, dim=2)
        v = v.repeat_interleave(self.num_local_heads // self.num_local_kv_heads, dim=2)

        if os.getenv("CONTEXT_PARALLEL", "0") == "1":
            sm_scale = 1.0 / (q.size(-1) ** 0.5)
            out = ring_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), sm_scale, True
            ).transpose(1, 2)
        else:
            out = flash_attn_func(q, k, v, causal=True)
        out = out.reshape(batch_size, seq_len, self.num_local_heads * self.head_dim)
        out = self.out_proj(out)
        return out


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        RMSNorm = LlamaRMSNorm if os.getenv("FLASH_ATTN", "1") != "1" else FlashRMSNorm

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = Attention(config)
        self.mlp = MLP(config)

        head_dim = config.hidden_size // config.num_attention_heads
        self.cos, self.sin = get_cos_sin(config.max_position_embeddings, head_dim=head_dim)
        self.cos, self.sin = update_rope_for_context_parallel(self.cos, self.sin)

    def forward(self, x):
        x = x + self.attention(self.input_layernorm(x), self.cos, self.sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        RMSNorm = LlamaRMSNorm if os.getenv("FLASH_ATTN", "1") != "1" else FlashRMSNorm
        # sanity check
        assert config.hidden_size % config.num_attention_heads == 0
        assert config.num_attention_heads % config.num_key_value_heads == 0

        # params
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_values = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.num_layers = config.num_hidden_layers
        self.model_config = config

        # modules
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.decoder_layers = nn.ModuleList([DecoderLayer(config) for _ in range(self.num_layers)])
        self.final_norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.final_proj = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.decoder_layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.final_proj(x)
        return logits
