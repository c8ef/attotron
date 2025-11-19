import torch
from flash_attn.layers.rotary import apply_rotary_emb

from attotron.nn.rope import get_cos_sin, llama_rotary_emb


def test_rope():
    batch_size = 1
    seq_len = 32
    num_heads = 4
    head_dim = 16

    x = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32).cuda()
    cos, sin = get_cos_sin(seq_len, head_dim)
    out_flash = apply_rotary_emb(x, cos[:, : head_dim // 2], sin[:, : head_dim // 2])
    out_llama = llama_rotary_emb(x, cos, sin)
    torch.testing.assert_close(out_flash, out_llama)


if __name__ == "__main__":
    test_rope()
