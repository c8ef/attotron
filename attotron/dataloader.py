from functools import partial

import numpy as np
import torch
from datasets import Features, Sequence, Value, load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from . import pgm


class MicroBatchDataLoader(DataLoader):
    def __init__(
        self,
        seq_len,
        micro_batch_size,
        grad_acc_steps,
        dataset_name,
        tokenizer_name,
        max_tokens,
        num_workers,
        num_proc,
        seed,
        split="train",
    ):
        self.seq_len = seq_len
        self.micro_batch_size = micro_batch_size
        self.grad_acc_steps = grad_acc_steps
        self.global_batch_size = (
            micro_batch_size * grad_acc_steps * pgm.pgm.dp_world_size
        )

        self.dataset = load_dataset(dataset_name, split=split)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.tokenized_dataset = self.tokenize_dataset(
            self.dataset, "text", seq_len, num_proc
        )
        total_tokens = self.tokenized_dataset.num_rows * (seq_len + 1)
        assert total_tokens >= max_tokens, (
            f"Not enough tokens. "
            f"Have {total_tokens} tokens but need {max_tokens} tokens"
        )

        self.sampler = DistributedSampler(
            self.tokenized_dataset,
            num_replicas=pgm.pgm.dp_world_size,
            rank=pgm.pgm.dp_rank,
            seed=seed,
            shuffle=False,
        )

        super().__init__(
            self.tokenized_dataset,
            batch_size=micro_batch_size,
            shuffle=False,
            sampler=self.sampler,
            num_workers=num_workers,
            collate_fn=self.collate_batch,
            pin_memory=True,
        )

    def tokenize_group_text(self, examples, tokenizer, seq_len):
        tokenized_text_batch = tokenizer.batch_encode_plus(
            examples, return_tensors="np"
        )
        concatenated_tokens = {
            "input_ids": np.concatenate(tokenized_text_batch["input_ids"])
        }
        total_len = len(concatenated_tokens["input_ids"])

        if total_len >= seq_len + 1:
            total_len = ((total_len - 1) // seq_len) * seq_len + 1

        result = {
            "input_ids": [
                concatenated_tokens["input_ids"][i : i + seq_len + 1]
                for i in range(0, total_len - seq_len, seq_len)
            ]
        }
        return result

    def tokenize_dataset(self, dataset, text_column_name, seq_len, num_proc):
        tokenizer_func = partial(
            self.tokenize_group_text, tokenizer=self.tokenizer, seq_len=seq_len
        )
        tokenized_dataset = dataset.map(
            tokenizer_func,
            input_columns=text_column_name,
            remove_columns=dataset.column_names,
            features=Features({
                "input_ids": Sequence(feature=Value(dtype="int64"), length=seq_len + 1)
            }),
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=True,
            desc=f"Grouping texts in chunks of {seq_len + 1}",
        )
        return tokenized_dataset

    def collate_batch(self, batch):
        batch_input_ids = torch.stack([
            torch.tensor(item["input_ids"]) for item in batch
        ])
        input_ids = batch_input_ids[:, :-1].contiguous()
        target_ids = batch_input_ids[:, 1:].contiguous()
        return {"input_ids": input_ids, "target_ids": target_ids}

    def __iter__(self):
        if self._iterator is None:
            self._iterator = super().__iter__()
        return self

    def __next__(self):
        if self._iterator is None:
            self._iterator = super().__iter__()
        try:
            batch = next(self._iterator)
        except StopIteration as exc:
            self._iterator = None
            raise StopIteration from exc
        return batch
