import torch.distributed as dist
import torch.nn as nn

from . import pgm


class DataParallelNaive(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.require_backward_grad_sync = True
        self.register_backward_hook(self._allreduce_grads)

    def register_backward_hook(self, hook):
        for p in self.module.parameters():
            if p.requires_grad is True:
                p.register_hook(hook)

    def _allreduce_grads(self, grad):
        if self.require_backward_grad_sync:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=pgm.pgm.dp_group)
            grad /= pgm.pgm.dp_world_size
        return grad

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
