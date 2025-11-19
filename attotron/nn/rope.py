import torch


def get_cos_sin(seq_len, head_dim, base=500000.0):
    assert head_dim % 2 == 0
    dtype = torch.bfloat16
    device = "cuda"

    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float().to("cpu") / head_dim)
    )
    inv_freq = inv_freq.to(device)
    position = torch.arange(seq_len).to(device).unsqueeze(1).float()
    return (
        torch.cos(position.float() * inv_freq.float()).to(dtype).repeat(1, 2),
        torch.sin(position.float() * inv_freq.float()).to(dtype).repeat(1, 2),
    )


def llama_rotary_emb(x, cos, sin):
    x = x.transpose(1, 2)
    head_dim = x.size(-1)
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    rotate_half = torch.cat([-x2, x1], dim=-1)
    x = x * cos + rotate_half * sin
    return x.transpose(1, 2)
