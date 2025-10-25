import os

import torch
import torch.distributed as dist


class PGM:
    def __init__(self, dp_size, tp_size, pp_size):
        self.world_size = dist.get_world_size()
        self.global_rank = dist.get_rank()
        self.local_rank = int(
            os.environ.get("LOCAL_RANK", self.global_rank % self.world_size)
        )

        assert self.world_size == dp_size * tp_size * pp_size, (
            f"WORLD({self.world_size}) != DP({dp_size}) * TP({tp_size}) * PP({pp_size})"
        )

        self.grid = torch.arange(self.world_size).view(dp_size, tp_size, pp_size)
        self.dp_rank, self.tp_rank, self.pp_rank = (
            (self.grid == self.global_rank).nonzero().flatten().tolist()
        )

        self.world_group = dist.group.WORLD
        self.dp_group = dist.new_subgroups_by_enumeration([
            self.grid[:, t, p].tolist() for t in range(tp_size) for p in range(pp_size)
        ])[0]
        self.tp_group = dist.new_subgroups_by_enumeration([
            self.grid[d, :, p].tolist() for d in range(dp_size) for p in range(pp_size)
        ])[0]
        self.pp_group = dist.new_subgroups_by_enumeration([
            self.grid[d, t, :].tolist() for d in range(dp_size) for t in range(tp_size)
        ])[0]

        self.dp_group_ids = self.grid[:, self.tp_rank, self.pp_rank]
        self.tp_group_ids = self.grid[self.dp_rank, :, self.pp_rank]
        self.pp_group_ids = self.grid[self.dp_rank, self.tp_rank, :]

        self.dp_world_size = dist.get_world_size(self.dp_group)
        self.dp_first_rank = self.dp_group_ids[0]
        self.dp_last_rank = self.dp_group_ids[-1]

        self.tp_world_size = dist.get_world_size(self.tp_group)
        self.tp_first_rank = self.tp_group_ids[0]
        self.tp_last_rank = self.tp_group_ids[-1]

        self.pp_world_size = dist.get_world_size(self.pp_group)
        self.pp_first_rank = self.pp_group_ids[0]
        self.pp_last_rank = self.pp_group_ids[-1]
        self.pp_is_first_stage = self.pp_rank == 0
        self.pp_is_last_stage = self.pp_rank == self.pp_world_size - 1
        self.pp_next_rank = (
            None
            if self.pp_is_last_stage
            else int(self.grid[self.dp_rank, self.tp_rank, self.pp_rank + 1].item())
        )
        self.pp_prev_rank = (
            None
            if self.pp_is_first_stage
            else int(self.grid[self.dp_rank, self.tp_rank, self.pp_rank - 1].item())
        )

    def __str__(self):
        return (
            f"RANK({self.global_rank})-"
            f"DP({self.dp_world_size})-"
            f"TP({self.tp_world_size})-"
            f"PP({self.pp_world_size})"
        )


def setup_pgm(dp_size, tp_size, pp_size):
    global pgm
    pgm = PGM(dp_size, tp_size, pp_size)
