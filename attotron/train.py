"""
torchrun \
    --nproc_per_node 4 \
    -m attotron.train \
    --num_proc 16 \
    --seq_len 128 \
    --micro_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --max_tokens 40960 \
    --pp_size 4 \
    --pp_engine 1f1b \
    --use_wandb \
    --run_name pp_1f1b
"""

import argparse
import datetime
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from torch.optim import AdamW
from transformers import AutoConfig

from . import pgm
from .data_parallel import DataParallelBucket
from .dataloader import MicroBatchDataLoader
from .model import Llama
from .pgm import setup_pgm
from .pipeline_parallel import (
    PipelineParallel,
    train_step_pipeline_1f1b,
    train_step_pipeline_afab,
)
from .tensor_parallel import apply_tensor_parallel
from .utils import print, readable, set_all_seed


def all_reduce_loss_across_dp_ranks(loss, device):
    reduced_loss = torch.tensor(
        [loss if loss is not None else 0.0], dtype=torch.float32, device=device
    )
    if pgm.pgm.pp_is_last_stage:
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.AVG, group=pgm.pgm.dp_group)
    return reduced_loss.item()


def train_step(model, dataloader, device):
    acc_loss = 0.0
    requires_grad_sync = pgm.pgm.dp_world_size > 1

    for i in range(dataloader.grad_acc_steps):
        batch = next(dataloader)
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        if requires_grad_sync:
            model.require_backward_grad_sync = i == dataloader.grad_acc_steps - 1

        outputs = model(input_ids)

        batch_size, seq_len = input_ids.shape
        target_ids = target_ids.reshape(-1)
        outputs = outputs.view(batch_size * seq_len, -1)
        loss = F.cross_entropy(outputs, target_ids, reduction="mean") / dataloader.grad_acc_steps

        loss.backward()
        acc_loss += loss.item()

    return acc_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for Llama model")

    # Environment arguments
    parser.add_argument("--omp_num_threads", type=str, default="1")
    parser.add_argument("--tokenizers_parallelism", type=str, default="false")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-360M-Instruct")
    parser.add_argument("--num_hidden_layers", type=int, default=32)
    parser.add_argument("--num_attention_heads", type=int, default=16)
    parser.add_argument("--num_key_value_heads", type=int, default=4)

    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_proc", type=int, default=4)

    # Training arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=1e6)

    # Distributed training arguments
    parser.add_argument("--dp_size", type=int, default=1, help="Data parallel size")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--pp_engine", type=str, default="afab", choices=["afab", "1f1b"])

    # Logging arguments
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default="default_run")

    args = parser.parse_args()

    # Set environment variables
    os.environ["OMP_NUM_THREADS"] = args.omp_num_threads
    os.environ["TOKENIZERS_PARALLELISM"] = args.tokenizers_parallelism
    os.environ["DEVICE"] = "cuda"

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl"
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16

    dist.init_process_group(
        rank=global_rank,
        world_size=world_size,
        backend=backend,
        init_method="env://",
        timeout=datetime.timedelta(minutes=2),
    )
    setup_pgm(args.dp_size, args.tp_size, args.pp_size)
    set_all_seed(args.seed)

    is_log_rank = pgm.pgm.tp_rank == 0 and pgm.pgm.dp_rank == 0 and pgm.pgm.pp_is_last_stage
    if is_log_rank and args.use_wandb:
        wandb.init(
            project="attotron",
            name=f"{args.run_name}-{pgm.pgm}",
            config={
                "model": args.model_name,
                "seed": args.seed,
                "learning_rate": args.learning_rate,
                "data_parallel_size": pgm.pgm.dp_world_size,
                "tensor_parallel_size": pgm.pgm.tp_world_size,
                "pipeline_parallel_size": pgm.pgm.pp_world_size,
            },
        )

    model_config = AutoConfig.from_pretrained(args.model_name)
    model_config.num_hidden_layers = args.num_hidden_layers
    model_config.num_attention_heads = args.num_attention_heads
    model_config.num_key_value_heads = args.num_key_value_heads
    model_config.max_position_embeddings = args.seq_len

    model = Llama(config=model_config)

    if pgm.pgm.tp_world_size > 1:
        model = apply_tensor_parallel(model)

    if pgm.pgm.pp_world_size > 1:
        model = PipelineParallel(model, model_config)

    model.to(dtype).to(device)

    if pgm.pgm.dp_world_size > 1:
        model = DataParallelBucket(model)

    model.train()
    dist.barrier()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    dist.barrier()

    # Create dataloader
    dataloader = MicroBatchDataLoader(
        seed=args.seed,
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size,
        grad_acc_steps=args.gradient_accumulation_steps,
        dataset_name=args.dataset_name,
        tokenizer_name=args.model_name,
        max_tokens=args.max_tokens,
        num_workers=args.num_workers,
        num_proc=args.num_proc,
    )
    tokens_per_step = dataloader.global_batch_size * args.seq_len
    print(
        "Tokens per step:",
        readable(tokens_per_step),
        is_print_rank=is_log_rank,
    )

    trained_tokens, step = 0, 0
    tensor_shapes = (
        dataloader.micro_batch_size,
        dataloader.seq_len,
        model_config.hidden_size,
    )
    dist.barrier()

    # Training loop
    while trained_tokens < args.max_tokens:
        step_start_time = time.time()

        optimizer.zero_grad()

        if pgm.pgm.pp_world_size > 1:
            if args.pp_engine == "afab":
                loss = train_step_pipeline_afab(model, dataloader, tensor_shapes, device, dtype)
            elif args.pp_engine == "1f1b":
                loss = train_step_pipeline_1f1b(model, dataloader, tensor_shapes, device, dtype)
            else:
                raise ValueError(f"Invalid pipeline parallel engine: {args.pp_engine}")
        else:
            loss = train_step(model, dataloader, device)
        loss = all_reduce_loss_across_dp_ranks(loss, device)

        optimizer.step()

        step_duration = time.time() - step_start_time
        trained_tokens += tokens_per_step
        step += 1

        if hasattr(model, "reset"):
            model.reset()

        print(
            f"[rank {pgm.pgm.global_rank}] Step: {step}, Loss: {loss:.4f}, "
            f"Global batch size (with seq_len): {readable(tokens_per_step)}, "
            f"Tokens/s: {readable(tokens_per_step / step_duration)}, "
            f"Tokens/s/GPU: {readable(tokens_per_step / step_duration / world_size)}, "
            f"Tokens: {readable(trained_tokens)}/{readable(args.max_tokens)}, "
            f"Memory usage: {torch.cuda.memory_reserved() / 1e9:.2f}GB",
            is_print_rank=is_log_rank,
        )
        if is_log_rank and args.use_wandb:
            wandb.log(
                {
                    "loss": loss,
                    "tokens_per_step": tokens_per_step,
                    "tokens_per_second": tokens_per_step / step_duration,
                    "memory_usage": torch.cuda.memory_reserved() / 1e9,
                    "trained_tokens": trained_tokens,
                }
            )

    if is_log_rank and args.use_wandb:
        wandb.finish()

    dist.destroy_process_group()
