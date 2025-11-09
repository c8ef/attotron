import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from . import pgm

STEP = 0
VERBOSE = os.environ.get("VERBOSE", "0") == "1"


def pipeline_communicate(operation, device, dtype, tensor=None, shape=None):
    global STEP, VERBOSE

    if operation == "recv_forward":
        if pgm.pgm.pp_is_first_stage:
            return None
        tensor = torch.empty(shape, dtype=dtype, device=device, requires_grad=True)
        src = pgm.pgm.pp_prev_rank
    elif operation == "send_forward":
        if pgm.pgm.pp_is_last_stage:
            return None
        dst = pgm.pgm.pp_next_rank
    elif operation == "recv_backward":
        if pgm.pgm.pp_is_last_stage:
            return None
        tensor = torch.empty(shape, dtype=dtype, device=device, requires_grad=True)
        src = pgm.pgm.pp_next_rank
    elif operation == "send_backward":
        if pgm.pgm.pp_is_first_stage:
            return None
        dst = pgm.pgm.pp_prev_rank

    is_send = operation.startswith("send")
    peer_rank = dst if is_send else src
    op = dist.P2POp(dist.isend if is_send else dist.irecv, tensor, peer_rank)

    if VERBOSE:
        print(
            f"{operation} | {'send-ing' if is_send else 'recv-ing'} "
            f"{operation.split('_')[1]} {pgm.pgm.pp_rank} {'->' if is_send else '<-'} "
            f"{peer_rank} | STEP:{STEP}",
            flush=True,
        )

    [req.wait() for req in dist.batch_isend_irecv([op])]
    torch.cuda.synchronize()

    if VERBOSE:
        STEP += 1

    return tensor if not is_send else None


class PipelineParallel(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        layer_distribution = self.distribute_layers(config.num_hidden_layers)

        self.embedding = model.embedding if pgm.pgm.pp_is_first_stage else nn.Identity()
        self.decoder_layers = nn.ModuleDict({
            str(i): model.decoder_layers[i] for i in layer_distribution
        })
        self.final_norm = (
            model.final_norm if pgm.pgm.pp_is_last_stage else nn.Identity()
        )
        self.final_proj = (
            model.final_proj if pgm.pgm.pp_is_last_stage else nn.Identity()
        )

    def distribute_layers(self, num_layers):
        layers_per_gpu = [
            num_layers // pgm.pgm.pp_world_size
            + (1 if i < num_layers % pgm.pgm.pp_world_size else 0)
            for i in range(pgm.pgm.pp_world_size)
        ]
        start_layer = sum(layers_per_gpu[: pgm.pgm.pp_rank])
        return list(range(start_layer, start_layer + layers_per_gpu[pgm.pgm.pp_rank]))

    def forward(self, input_ids, hidden_states):
        x = hidden_states if hidden_states is not None else input_ids
        x = self.embedding(x)
        for layer in self.decoder_layers.values():
            x = layer(x)
        x = self.final_norm(x)
        x = self.final_proj(x)
        return x

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        if input_tensor is not None:
            input_tensor.retain_grad()
        if output_tensor is None:
            output_tensor_grad = torch.ones_like(
                output_tensor, memory_format=torch.preserve_format
            )
        torch.autograd.backward(
            output_tensor,
            grad_tensors=output_tensor_grad,
            retain_graph=False,
            create_graph=False,
        )
        return input_tensor.grad if input_tensor is not None else None


def train_step_pipeline_afab(model, data_loader, tensor_shapes, device, dtype):
    logging_loss = 0.0
    input_tensors = []
    output_tensors = []
    requires_grad_sync = pgm.pgm.dp_world_size > 1

    for _ in range(data_loader.grad_acc_steps):
        input_tensor = pipeline_communicate(
            operation="recv_forward", device=device, dtype=dtype, shape=tensor_shapes
        )
        batch = next(data_loader)
        batch["hidden_states"] = (
            input_tensor.to(device) if input_tensor is not None else input_tensor
        )
        output_tensor = model.forward(
            input_ids=batch["input_ids"].to(device),
            hidden_states=batch["hidden_states"],
        )
        pipeline_communicate(
            operation="send_forward",
            device=device,
            dtype=dtype,
            tensor=output_tensor,
        )

        if pgm.pgm.pp_is_last_stage:
            output_tensor = F.cross_entropy(
                output_tensor.transpose(1, 2),
                batch["target_ids"].to(device),
                reduction="mean",
            )
            logging_loss += output_tensor.item() / data_loader.grad_acc_steps

        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)

    for ith_microbatch in range(data_loader.grad_acc_steps):
        if requires_grad_sync:
            is_last_iteration = ith_microbatch == data_loader.grad_acc_steps - 1
            model.require_backward_grad_sync = is_last_iteration
        output_tensor_grad = pipeline_communicate(
            operation="recv_backward", device=device, dtype=dtype, shape=tensor_shapes
        )
        input_tensor, output_tensor = input_tensors.pop(0), output_tensors.pop(0)
        input_tensor_grad = model.backward(
            input_tensor, output_tensor, output_tensor_grad
        )
        pipeline_communicate(
            operation="send_backward",
            device=device,
            dtype=dtype,
            tensor=input_tensor_grad,
        )
    return logging_loss
