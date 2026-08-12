"""Reproduce the RCCL failure that kills kimi pipeline parallelism, without paying for vLLM.

vLLM hands a pipeline stage off with ``get_pp_group().irecv_tensor_dict(...)``, which reaches
``torch.distributed.irecv`` on the PP process group. That group has size 4 (four PP stages), but a
point-to-point op on it needs a communicator holding only the two ranks involved, so RCCL builds a
fresh 2-rank one on first use -- the thing torch warns about as "an unbatched P2P op (send/recv) was
called on this ProcessGroup with size 4 ... a new 2-rank NCCL communicator to be created".

In job 590270 that creation failed with ``ncclInternalError: Internal check failed`` while every
collective was healthy, which is why this probe does BOTH: an allreduce over the world group to show
the fabric works at all, then the P2P chain over a size-4 group that straddles nodes. A run where
allreduce passes and P2P fails is the kimi bug; a run where both pass clears the env under test.

Layout mirrors the real one. With 2 nodes x 4 ranks the group is [0, 1, L, L+1], so the chain covers
an intra-node hop and a cross-node hop, and the group stays size 4 -- at size 2 the group's own
communicator would serve the P2P directly and the lazy path this exists to test never runs.
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
import traceback

import torch
import torch.distributed as dist


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def report(key: str, value: object) -> None:
    """One machine-greppable line per fact; the sbatch verdict keys off these."""
    print(f"{key}={value}", flush=True)


def pp_group_ranks(world_size: int, local_world_size: int) -> list[int]:
    """Two ranks from the first node and two from the second: size 4, straddling the fabric."""
    if world_size < 4:
        raise ValueError(f"need at least 4 ranks to build a size-4 group, got {world_size}")
    if local_world_size < 2 or world_size < 2 * local_world_size:
        raise ValueError(f"need >= 2 nodes of >= 2 ranks, got world={world_size} local={local_world_size}")
    return [0, 1, local_world_size, local_world_size + 1]


def occupy_memory(device: torch.device, fraction: float, rank: int):
    """Hold ``fraction`` of free GPU memory, so communicators are built on a nearly-full device.

    An empty-GPU probe is not a fair model of the failing run. vLLM had ~37 GiB of weights and ~67
    GiB of KV cache resident per GPU when the pipeline handoff created its communicator, and CXI's
    memory registration draws on a pool that large allocations have already eaten into -- probe
    590216 failed with ``cxil_map: write error``, which is libcxi's registration ioctl, not a
    transport error. Returns the ballast so the caller keeps it alive; None when disabled.
    """
    if fraction <= 0.0:
        return None
    free, total = torch.cuda.mem_get_info(device)
    want = int(free * fraction)
    ballast = torch.empty(want, dtype=torch.uint8, device=device)
    if rank == 0:
        report("fill_gib", round(want / (1 << 30), 2))
        report("free_after_fill_gib", round(torch.cuda.mem_get_info(device)[0] / (1 << 30), 2))
        report("total_gib", round(total / (1 << 30), 2))
    return ballast


def run_allreduce(device: torch.device, rank: int, world_size: int) -> None:
    tensor = torch.full((1024, 1024), float(rank + 1), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    expected = float(world_size * (world_size + 1) // 2)
    got = tensor[0, 0].item()
    if rank == 0:
        report("allreduce_expected", expected)
        report("allreduce_got", got)
        report("allreduce_correct", got == expected)
    if got != expected:
        raise ValueError(f"allreduce wrong: expected {expected}, got {got}")


def run_p2p_chain(group, ranks: list[int], device: torch.device, rank: int, elements: int) -> None:
    """Forward a payload down the group as a LINE, the way a PP stage forwards hidden states.

    A line, not a ring. Creating the lazy 2-rank communicator is itself blocking and collective over
    exactly that pair, so a ring in which every rank issues its irecv first deadlocks on communicator
    creation alone -- rank 0 waits on comm{0,last} while the last rank waits on comm{last-1,last} --
    and reports a TCPStore rendezvous timeout that has nothing to do with the fabric. That is a bug
    in the test, not in RCCL, and this probe hit it before the wraparound came out. vLLM's pipeline
    is a line: stage 0 only sends, the last stage only receives, the middle both.

    Non-blocking isend/irecv on purpose: that is the exact torch entry point vLLM uses
    (``irecv_tensor_dict``), and the blocking form takes a different path inside the process group.
    """
    position = ranks.index(rank)
    payload = torch.full((elements, ), float(rank), device=device)

    if position > 0:
        src = ranks[position - 1]
        inbox = torch.empty((elements, ), device=device)
        dist.irecv(inbox, src=src, group=group).wait()
        torch.cuda.synchronize()
        got = inbox[0].item()
        if got != float(src):
            raise ValueError(f"rank {rank} expected {src} from src {src}, got {got}")
    if position < len(ranks) - 1:
        dist.isend(payload, dst=ranks[position + 1], group=group).wait()
        torch.cuda.synchronize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, default=1 << 20, help="payload floats per P2P hop")
    parser.add_argument("--timeout-seconds", type=int, default=90, help="collective timeout, per group")
    parser.add_argument("--fill-fraction",
                        type=float,
                        default=0.0,
                        help="occupy this fraction of free GPU memory BEFORE building groups, so the "
                        "communicator is created under the memory pressure a loaded model imposes")
    args = parser.parse_args()

    rank = env_int("RANK", 0)
    world_size = env_int("WORLD_SIZE", 1)
    local_rank = env_int("LOCAL_RANK", 0)
    local_world_size = env_int("LOCAL_WORLD_SIZE", world_size)

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    timeout = datetime.timedelta(seconds=args.timeout_seconds)

    # A finite timeout is the point: the vLLM symptom was a wedge, and a wedged probe teaches
    # nothing that a raised DistBackendError does not teach faster.
    dist.init_process_group(backend="nccl", timeout=timeout)

    if rank == 0:
        report("world_size", world_size)
        report("local_world_size", local_world_size)
        report("torch_version", torch.__version__)
        report("nccl_version", ".".join(str(v) for v in torch.cuda.nccl.version()))

    stage = "allreduce"
    ballast = None
    try:
        # Before any group is built: the failure under investigation happens at communicator
        # CREATION, so the pressure has to be in place by then, not added afterwards.
        ballast = occupy_memory(device, args.fill_fraction, rank)
        run_allreduce(device, rank, world_size)
        if rank == 0:
            report("stage_allreduce", "PASS")

        stage = "group_create"
        wide = pp_group_ranks(world_size, local_world_size)
        pair = [0, local_world_size]
        if rank == 0:
            report("pp_group_ranks", ",".join(str(r) for r in wide))
            report("pair_group_ranks", ",".join(str(r) for r in pair))
        # Every rank must call new_group, members or not -- it is collective over the world group,
        # and both groups must be built before either is used so the two stages cannot interleave.
        wide_group = dist.new_group(ranks=wide, timeout=timeout)
        pair_group = dist.new_group(ranks=pair, timeout=timeout)
        if rank == 0:
            report("stage_group_create", "PASS")

        # The discriminating pair of stages. torch only mints a lazy 2-rank communicator when the
        # group is WIDER than the pair actually exchanging data; on a size-2 group the group's own
        # communicator serves the P2P directly. So size-4 FAIL + size-2 PASS means pipeline
        # parallelism is fine at pp=2 and only the lazy sub-communicator is broken -- which is the
        # difference between "kimi needs a different topology" and "kimi cannot run here at all".
        # size-2 runs FIRST, deliberately. It is the stage expected to pass, and a failed collective
        # leaves the world group in a state the next stage cannot be trusted through -- so ordering
        # the likely-failing stage last is what guarantees both verdicts are actually recorded.
        stage = "p2p_size2"
        if rank in pair:
            run_p2p_chain(pair_group, pair, device, rank, args.elements)
        dist.barrier()
        if rank == 0:
            report("stage_p2p_size2", "PASS")

        stage = "p2p_size4"
        if rank in wide:
            run_p2p_chain(wide_group, wide, device, rank, args.elements)
        dist.barrier()
        if rank == 0:
            report("stage_p2p_size4", "PASS")
            report("verdict", "PASS")
    except Exception as exc:  # noqa: BLE001 -- the failure IS the measurement; report it, do not raise
        # EVERY rank reports its own failure, not just rank 0. Under memory pressure the cross-node
        # allreduce came back as zeros on the SECOND node while rank 0 computed the right answer and
        # printed PASS -- so a verdict keyed on rank 0 alone called a silent-wrong-answer run green.
        report(f"stage_{stage}", "FAIL")
        report("failed_stage", stage)
        report("failed_rank", rank)
        report("failure", repr(exc))
        report("verdict", "FAIL")
        traceback.print_exc()
        del ballast
        dist.destroy_process_group()
        return 1

    del ballast  # referenced here so the fill cannot be collected before the last communicator
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
