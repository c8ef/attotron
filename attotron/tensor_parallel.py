import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from . import pgm


class Copy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        if pgm.pgm.tp_world_size > 1:
            dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=pgm.pgm.tp_group)
        return grad_output


class Reduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        if pgm.pgm.tp_world_size > 1:
            dist.all_reduce(input, op=dist.ReduceOp.SUM, group=pgm.pgm.tp_group)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        if pgm.pgm.tp_world_size == 1:
            return input
        last_dim = input.dim() - 1
        input = input.contiguous()
        output_list = [torch.empty_like(input) for _ in range(pgm.pgm.tp_world_size)]
        output_list[pgm.pgm.tp_rank] = input
        dist.all_gather(output_list, input, group=pgm.pgm.tp_group)
        output = torch.cat(output_list, dim=last_dim).contiguous()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if pgm.pgm.tp_world_size == 1:
            return grad_output
        last_dim = grad_output.dim() - 1
        assert grad_output.size(last_dim) % pgm.pgm.tp_world_size == 0, (
            f"{grad_output.size(last_dim)} is not divisible by {pgm.pgm.tp_world_size}"
        )
        last_dim_size = grad_output.size(last_dim) // pgm.pgm.tp_world_size
        chunks = torch.split(grad_output, last_dim_size, dim=last_dim_size)
        return chunks[pgm.pgm.tp_rank]


class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias, gather_output=False):
        super().__init__()
        self.tp_world_size = pgm.pgm.tp_world_size
        self.tp_rank = pgm.pgm.tp_rank
        self.in_features = in_features
        self.out_features = out_features
        assert self.out_features % self.tp_world_size == 0, (
            "Hidden dimension must be divisible by tp_world_size"
        )
        self.output_size_per_partition = out_features // self.tp_world_size
        self.gather_output = gather_output

        self.weight = nn.Parameter(
            torch.Tensor(self.output_size_per_partition, self.in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_size_per_partition))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.tp_world_size == 1:
            k = 1 / self.weight.size(1)
            bound = math.sqrt(k)
            nn.init.uniform_(self.weight, -bound, bound)
            return

        master_weight = torch.empty(
            self.out_features,
            self.in_features,
            dtype=self.weight.dtype,
            requires_grad=False,
        )
        k = 1 / master_weight.size(1)
        bound = math.sqrt(k)
        nn.init.uniform_(master_weight, -bound, bound)
        weight_list = torch.split(master_weight, self.output_size_per_partition, dim=0)
        self.weight.data = weight_list[self.tp_rank].contiguous()

    def forward(self, input):
        input_parallel = Copy.apply(input)
        output = F.linear(input_parallel, self.weight, self.bias)
        if self.gather_output:
            output = Gather.apply(output)
        return output


class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias):
        super().__init__()
        self.tp_world_size = pgm.pgm.tp_world_size
        self.tp_rank = pgm.pgm.tp_rank
        self.in_features = in_features
        self.out_features = out_features
        assert self.in_features % self.tp_world_size == 0, (
            "Hidden dimension must be divisible by tp_world_size"
        )
        self.input_size_per_partition = in_features // self.tp_world_size

        self.weight = nn.Parameter(
            torch.Tensor(self.out_features, self.input_size_per_partition)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.tp_world_size == 1:
            k = 1 / self.weight.size(1)
            bound = math.sqrt(k)
            nn.init.uniform_(self.weight, -bound, bound)
            return

        master_weight = torch.empty(
            self.out_features,
            self.in_features,
            dtype=self.weight.dtype,
            requires_grad=False,
        )
        k = 1 / master_weight.size(1)
        bound = math.sqrt(k)
        nn.init.uniform_(master_weight, -bound, bound)
        weight_list = torch.split(master_weight, self.input_size_per_partition, dim=1)
        self.weight.data = weight_list[self.tp_rank].contiguous()

    def forward(self, input):
        output_parallel = F.linear(input, self.weight)
        output = Reduce.apply(output_parallel)
        return output if self.bias is None else output + self.bias


class VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
    ):
        super().__init__()
        self.tp_world_size = pgm.pgm.tp_world_size
        self.tp_rank = pgm.pgm.tp_rank
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        assert num_embeddings % self.tp_world_size == 0, (
            "num_embeddings is not divisible by tp_world_size"
        )
        self.num_embeddings_per_partition = num_embeddings // self.tp_world_size
        self.vocab_start_index = self.tp_rank * self.num_embeddings_per_partition
        self.vocab_end_index = (
            self.vocab_start_index + self.num_embeddings_per_partition
        )
        self.weight = nn.Parameter(
            torch.Tensor(self.num_embeddings_per_partition, self.embedding_dim)
        )
        self.reset_parameters()

    def reset_parameters(self):
        if self.tp_world_size == 1:
            nn.init.normal_(self.weight, mean=0.0, std=1.0)
            return

        master_weight = torch.empty(
            self.num_embeddings,
            self.embedding_dim,
            dtype=self.weight.dtype,
            requires_grad=False,
        )
        nn.init.normal_(master_weight, mean=0.0, std=1.0)
        weight_list = torch.split(
            master_weight, self.num_embeddings_per_partition, dim=0
        )
        self.weight.data = weight_list[self.tp_rank].contiguous()

    def forward(self, input):
        input_mask = (input < self.vocab_start_index) | (input >= self.vocab_end_index)
        masked_input = input.clone() - self.vocab_start_index
        masked_input[input_mask] = 0
        output_parallel = F.embedding(
            masked_input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        output_parallel[input_mask, :] = 0.0
        output = Reduce.apply(output_parallel)
        return output


def apply_tensor_parallel(model):
    def _replace_module(_module, _linear_proj_name, _style, args={}):
        assert _style in ["column", "row", "vocab"]

        linear_layer = getattr(_module, _linear_proj_name)
        if _style == "column":
            new_linear_layer = ColumnParallelLinear(
                in_features=linear_layer.in_features,
                out_features=linear_layer.out_features,
                bias=linear_layer.bias is not None,
                gather_output=args.get("gather_output", False),
            )
        elif _style == "row":
            new_linear_layer = RowParallelLinear(
                in_features=linear_layer.in_features,
                out_features=linear_layer.out_features,
                bias=linear_layer.bias is not None,
            )
        else:
            new_linear_layer = VocabParallelEmbedding(
                num_embeddings=linear_layer.num_embeddings,
                embedding_dim=linear_layer.embedding_dim,
            )
        setattr(_module, _linear_proj_name, new_linear_layer)

    module_linear_name_style_mapping_list = [
        ("attention", "q_proj", "column"),
        ("attention", "k_proj", "column"),
        ("attention", "v_proj", "column"),
        ("attention", "out_proj", "row"),
        ("mlp", "up_proj", "column"),
        ("mlp", "gate_proj", "column"),
        ("mlp", "down_proj", "row"),
    ]

    for layer in model.decoder_layers:
        for (
            module_name,
            linear_proj_name,
            style,
        ) in module_linear_name_style_mapping_list:
            _replace_module(getattr(layer, module_name), linear_proj_name, style)
    _replace_module(model, "embedding", "vocab")
    _replace_module(model, "final_proj", "column", args={"gather_output": True})

    return model
