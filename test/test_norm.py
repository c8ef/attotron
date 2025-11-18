import torch

from attotron.nn.norm import FlashRMSNorm, LlamaRMSNorm


def test_norm():
    batch_size = 4
    seq_len = 128
    hidden_size = 768
    eps = 1e-5

    flash_norm = FlashRMSNorm(hidden_size, eps).cuda()
    llama_norm = LlamaRMSNorm(hidden_size, eps).cuda()

    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32).cuda()
    out_flash = flash_norm(x)
    out_llama = llama_norm(x)
    assert torch.allclose(out_flash, out_llama)


if __name__ == "__main__":
    test_norm()
