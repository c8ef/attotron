"""
torchrun \
    --nproc_per_node 4 \
    -m attotron.train \
    --num_proc 16 \
    --seq_len 128 \
    --micro_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --max_tokens 40960 \
    --tp_size 4 \
    --use_wandb \
    --run_name dataloader
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
from .dataloader import MicroBatchDataLoader
from .model import Llama
from .pgm import setup_pgm
from .tensor_parallel import apply_tensor_parallel
from .utils import print, readable, set_all_seed


def train_step(model, dataloader, device):
    acc_loss = 0.0

    for _ in range(dataloader.grad_acc_steps):
        batch = next(dataloader)
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        outputs = model(input_ids)

        batch_size, seq_len = input_ids.shape
        target_ids = target_ids.reshape(-1)
        outputs = outputs.view(batch_size * seq_len, -1)
        loss = (
            F.cross_entropy(outputs, target_ids, reduction="mean")
            / dataloader.grad_acc_steps
        )

        loss.backward()
        acc_loss += loss.item()

    return acc_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for Llama model")

    # Environment arguments
    parser.add_argument("--omp_num_threads", type=str, default="1")
    parser.add_argument("--tokenizers_parallelism", type=str, default="false")

    # Model arguments
    parser.add_argument(
        "--model_name", type=str, default="HuggingFaceTB/SmolLM-360M-Instruct"
    )
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

    is_log_rank = pgm.pgm.global_rank == 0
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

    model.to(dtype).to(device)
    model.train()
    dist.barrier()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    dist.barrier()

    # Create dataloader
    dataloader = MicroBatchDataLoader(
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
    dist.barrier()

    # Training loop
    while trained_tokens < args.max_tokens:
        step_start_time = time.time()

        optimizer.zero_grad()
        loss = train_step(model, dataloader, device)
        optimizer.step()

        step_duration = time.time() - step_start_time
        trained_tokens += tokens_per_step
        step += 1

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
            wandb.log({
                "loss": loss,
                "tokens_per_step": tokens_per_step,
                "tokens_per_second": tokens_per_step / step_duration,
                "memory_usage": torch.cuda.memory_reserved() / 1e9,
                "trained_tokens": trained_tokens,
            })

    if is_log_rank and args.use_wandb:
        wandb.finish()

    dist.destroy_process_group()
