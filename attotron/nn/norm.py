import torch
import torch.nn as nn
from flash_attn.ops.triton.layer_norm import layer_norm_fn


class FlashRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        return layer_norm_fn(
            hidden_states,
            self.weight,
            None,
            eps=self.eps,
            is_rms_norm=True,
        )


class LlamaRMSNorm(nn.Module):
    """
    LlamaRMSNorm is equivalent to T5LayerNorm
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/pdf/1910.07467 thus variance is calculated
        # w/o mean and there is no bias.
        # Additionally we want to make sure that the accumulation for half-precision inputs is done
        # in fp32
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states
