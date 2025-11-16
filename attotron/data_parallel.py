import torch
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
            if p.requires_grad:
                p.register_hook(hook)

    def _allreduce_grads(self, grad):
        if self.require_backward_grad_sync:
            dist.all_reduce(grad, op=dist.ReduceOp.AVG, group=pgm.pgm.dp_group)
        return grad

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)


class Bucket:
    def __init__(self, params, grad_data, process_group):
        self.params = set(params)
        self.params_with_grad_ready = set()
        self.grad_data = grad_data
        self.process_group = process_group
        self.process_group_size = dist.get_world_size(process_group)
        self.handle = None

        self.reset()

    def reset(self):
        self.params_with_grad_ready.clear()
        self.grad_data.zero_()
        self.handle = None

    def sync_gradient(self):
        assert self.handle is None
        self.grad_data /= self.process_group_size
        self.handle = dist.all_reduce(self.grad_data, group=self.process_group, async_op=True)

    def mark_param_as_ready(self, param):
        assert param in self.params and param not in self.params_with_grad_ready
        self.params_with_grad_ready.add(param)
        if len(self.params_with_grad_ready) == len(self.params):
            self.sync_gradient()

    def wait(self):
        assert self.handle is not None
        self.handle.wait()


class BucketManager:
    def __init__(self, params, process_group, bucket_size, grad_type=torch.float32):
        self.params = list(params)
        self.buckets = []
        self.process_group = process_group
        self.process_group_size = dist.get_world_size(process_group)
        self.params_to_bucket_location = {}
        self.bucket_size = bucket_size
        self.bucket_sizes = None
        self.grad_data_list = []
        self.grad_type = grad_type

        self._initialize_buckets()

    def _initialize_buckets(self):
        cur_bucket_size = 0
        cur_bucket_idx = 0

        for param in self.params:
            if not param.requires_grad:
                continue

            if cur_bucket_size + param.numel() > self.bucket_size:
                cur_bucket_idx += 1
                cur_bucket_size = 0

            self.params_to_bucket_location[param] = (
                cur_bucket_size,
                cur_bucket_size + param.numel(),
                cur_bucket_idx,
            )
            cur_bucket_size += param.numel()

        bucket_sizes = [0] * (cur_bucket_idx + 1)
        buckets_to_params = [[] for _ in range(cur_bucket_idx + 1)]
        for param, (_, end, idx) in self.params_to_bucket_location.items():
            bucket_sizes[idx] = max(bucket_sizes[idx], end)
            buckets_to_params[idx].append(param)

        for i in range(len(bucket_sizes)):
            self.grad_data_list.append(
                torch.zeros(bucket_sizes[i], dtype=self.grad_type, device="cuda")
            )
            self.buckets.append(
                Bucket(buckets_to_params[i], self.grad_data_list[i], self.process_group)
            )

        for param in self.params[::-1]:
            if not param.requires_grad:
                continue
            data_start_idx, data_end_idx, bucket_id = self.params_to_bucket_location[param]
            param.main_grad = self.grad_data_list[bucket_id][data_start_idx:data_end_idx].view(
                param.shape
            )

    def reset(self):
        for bucket in self.buckets:
            bucket.reset()

    def mark_param_as_ready(self, param):
        bucket_idx = self.params_to_bucket_location[param][2]
        self.buckets[bucket_idx].mark_param_as_ready(param)

    def wait(self):
        for bucket in self.buckets:
            bucket.wait()


class DataParallelBucket(nn.Module):
    def __init__(self, module, bucket_cap_mb=100, grad_type=torch.float32):
        super().__init__()
        self.module = module
        self.require_backward_grad_sync = True
        grad_size = 2
        bucket_size = bucket_cap_mb * 1024 * 1024 // grad_size
        self.bucket_manager = BucketManager(
            module.parameters(), pgm.pgm.dp_group, bucket_size, grad_type
        )

        self.register_backward_hook()
        self._post_backward_callback_set = False

    def register_backward_hook(self):
        self.grad_accs = []
        for param in self.module.parameters():
            if param.requires_grad:
                param_tmp = param.expand_as(param)
                grad_acc_fn = param_tmp.grad_fn.next_functions[0][0]
                grad_acc_fn.register_hook(self._make_param_hook(param, self.bucket_manager))
                self.grad_accs.append(grad_acc_fn)

    def _make_param_hook(self, param, bucket_manager):
        def param_hook(*unused):
            if param.requires_grad:
                assert param.grad is not None
                param.main_grad.add_(param.grad.data)
                param.grad = None

                if self.require_backward_grad_sync:
                    if not self._post_backward_callback_set:
                        torch.autograd.Variable._execution_engine.queue_callback(
                            self._post_backward
                        )
                        self._post_backward_callback_set = True
                    bucket_manager.mark_param_as_ready(param)

        return param_hook

    def _post_backward(self):
        self.bucket_manager.wait()
        self._post_backward_callback_set = False
        for p in self.module.parameters():
            if p.requires_grad:
                p.grad = p.main_grad.to(p.dtype)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def reset(self):
        self.bucket_manager.reset()
