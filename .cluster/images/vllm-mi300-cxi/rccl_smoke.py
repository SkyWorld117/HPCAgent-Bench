#!/usr/bin/env python3
"""Two-node RCCL/OFI all-reduce smoke test launched by Slurm.

PyTorch retains the backend name ``nccl`` on ROCm; RCCL implements it.
Run one task and at least one allocated GPU per node for initial validation.
"""

from __future__ import annotations

from datetime import timedelta
import os
import socket
import sys

import torch
import torch.distributed as dist


def required_int(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"missing required environment variable: {name}")
    return int(value)


def main() -> int:
    rank = required_int("SLURM_PROCID")
    world_size = required_int("SLURM_NTASKS")
    local_rank = int(os.environ.get("SLURM_LOCALID", "0"))

    if not torch.version.hip:
        raise RuntimeError("PyTorch is not a ROCm build")
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm device is not visible through torch.cuda")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"local rank {local_rank} but only {torch.cuda.device_count()} GPU(s) visible"
        )

    torch.cuda.set_device(local_rank)
    props = torch.cuda.get_device_properties(local_rank)
    arch = getattr(props, "gcnArchName", "unknown")

    print(
        f"host={socket.gethostname()} rank={rank}/{world_size} "
        f"HIP={torch.version.hip} device={props.name!r} arch={arch!r}",
        flush=True,
    )

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=5),
    )

    value = torch.tensor([rank + 1.0], device=f"cuda:{local_rank}")
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    expected = world_size * (world_size + 1) / 2
    result = value.item()
    ok = result == expected
    print(
        f"host={socket.gethostname()} rank={rank}/{world_size} "
        f"sum={result} expected={expected} ok={ok}",
        flush=True,
    )

    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR on {socket.gethostname()}: {exc}", file=sys.stderr, flush=True)
        raise
