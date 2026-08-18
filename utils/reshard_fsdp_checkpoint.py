"""Convert a contiguous FSDP checkpoint to a smaller world size.

This utility is intentionally conservative: it only supports an integer
down-shard ratio and preserves the source files alongside the converted ones.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--source-world-size", type=int, required=True)
    return parser.parse_args()


def load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def save(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    target_world_size = dist.get_world_size()
    source_world_size = args.source_world_size
    if source_world_size % target_world_size:
        raise ValueError("source world size must be divisible by target world size")

    actor_dir = args.checkpoint / "actor"
    ratio = source_world_size // target_world_size
    source_ranks = range(rank * ratio, (rank + 1) * ratio)
    mesh = init_device_mesh("cuda", (target_world_size,), mesh_dim_names=("fsdp",))

    model_shards = [
        load(actor_dir / f"model_world_size_{source_world_size}_rank_{source_rank}.pt")
        for source_rank in source_ranks
    ]
    converted_model = model_shards[0].__class__()
    for key in model_shards[0]:
        source_value = model_shards[0][key]
        local_value = torch.cat([shard[key].to_local() for shard in model_shards], dim=0).cuda()
        converted_model[key] = DTensor.from_local(
            local_value,
            mesh,
            (Shard(0),),
            shape=source_value.shape,
            stride=source_value.stride(),
            run_check=False,
        )
    save(converted_model, actor_dir / f"model_world_size_{target_world_size}_rank_{rank}.pt")
    del model_shards, converted_model
    torch.cuda.empty_cache()

    optim_shards = [
        load(actor_dir / f"optim_world_size_{source_world_size}_rank_{source_rank}.pt")
        for source_rank in source_ranks
    ]
    converted_optim = {"state": {}, "param_groups": optim_shards[0]["param_groups"]}
    for parameter_id, source_state in optim_shards[0]["state"].items():
        converted_state = {}
        for key, value in source_state.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                converted_state[key] = torch.cat(
                    [shard["state"][parameter_id][key] for shard in optim_shards], dim=0
                )
            else:
                converted_state[key] = value
        converted_optim["state"][parameter_id] = converted_state
    save(converted_optim, actor_dir / f"optim_world_size_{target_world_size}_rank_{rank}.pt")

    # Scheduler state is identical on all ranks. Preserve one RNG stream from
    # each contiguous source-rank group for deterministic, distinct workers.
    extra = load(actor_dir / f"extra_state_world_size_{source_world_size}_rank_{rank * ratio}.pt")
    save(extra, actor_dir / f"extra_state_world_size_{target_world_size}_rank_{rank}.pt")
    dist.barrier()
    if rank == 0:
        print(f"Converted {args.checkpoint} from world size {source_world_size} to {target_world_size}")


if __name__ == "__main__":
    main()
