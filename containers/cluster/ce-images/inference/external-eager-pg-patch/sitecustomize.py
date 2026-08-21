import faulthandler
import inspect
import os
import signal
import sys

import torch
import torch.distributed as dist

_original_init_process_group = dist.init_process_group
_original_new_group = dist.new_group
_new_group_parameters = inspect.signature(_original_new_group).parameters


def local_device() -> torch.device:
    local_rank = int(os.environ.get("LOCAL_RANK", str(torch.cuda.current_device())))
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


def backend_from_call(args, kwargs, positional_index):
    backend = kwargs.get("backend")
    if backend is None and len(args) > positional_index:
        backend = args[positional_index]
    return backend


# Escape hatch for the truncation probe: eager init (device_id set) turns unbatched P2P into
# independent collectives serialized on the group, and `received 1024 instead of 256` on
# isend_tensor_dict -- a receiver matching the WRONG message -- killed 3 of 4 four-node probes
# in warmup (603524/603714/603718). 0 skips the device_id everywhere, returning the world and
# every subgroup to lazy init while KEEPING the collective/P2P split below, which may be what
# the original lazy PG5 hang actually needed. Default 1 = today's behavior, byte-identical.
EAGER_DEVICE_ID = os.environ.get("VLLM_EAGER_PG_DEVICE_ID", "1") == "1"


def eager_init_process_group(*args, **kwargs):
    backend = backend_from_call(args, kwargs, 0)

    if (EAGER_DEVICE_ID and "nccl" in str(backend).lower() and kwargs.get("device_id") is None
            and torch.cuda.is_available()):
        device = local_device()
        kwargs["device_id"] = device

        print(
            "[external-eager-pg] init_process_group "
            f"backend={backend} device_id={device}",
            file=sys.stderr,
            flush=True,
        )

    return _original_init_process_group(*args, **kwargs)


def eager_new_group(*args, **kwargs):
    # new_group signature:
    # ranks, timeout, backend, ...
    backend = backend_from_call(args, kwargs, 2)

    if (EAGER_DEVICE_ID and "device_id" in _new_group_parameters and "nccl" in str(backend).lower()
            and kwargs.get("device_id") is None and torch.cuda.is_available()):
        device = local_device()
        kwargs["device_id"] = device

        print(
            "[external-eager-pg] new_group "
            f"backend={backend} device_id={device} "
            f"ranks={kwargs.get('ranks', args[0] if args else None)}",
            file=sys.stderr,
            flush=True,
        )

    return _original_new_group(*args, **kwargs)


# The pipeline group's collectives move to their own communicator.
#
# vLLM puts BOTH kinds of traffic on the pipeline group's `device_group`: the inter-stage
# activation transfer (`isend_tensor_dict`/`irecv_tensor_dict`, v1/worker/gpu_worker.py) and
# ordinary torch collectives. The eager init above is what makes that combination fatal -- torch
# states the consequence itself:
#
#   "An unbatched P2P op (send/recv) was called on this ProcessGroup with size 4. In eager
#    initialization mode, unbatched P2P ops are treated as independent collective ops, and are
#    thus serialized with all other ops on this ProcessGroup, including other P2P ops."
#
# 600262 and 600263 both died on it: a 4-element BROADCAST on the pp group (PG 5, ranks
# [0,4,8,12] and its three siblings) sat at SeqNum=1 -- its FIRST collective -- for the full
# 600 s watchdog while P2P was in flight, taking the engine and then the run with it. Neither
# arm decoded a single token.
#
# Upstream already ships the remedy and already uses it, for exactly one broadcast:
# v1/worker/gpu/pp_utils.py builds a `make_sibling_device_group` "so it does not serialize on the
# wire with the inter-stage hidden-state p2p send/recv ops". This extends that to EVERY collective
# on the group, so the pp `device_group` ends up carrying point-to-point traffic and nothing else.
# The sibling has identical membership, so global rank ids (`self.ranks[src]`) still address the
# same processes and no call site needs to change.
COLLECTIVE_SIBLING_GROUPS = ("pp", )


def patch_pp_collectives() -> None:
    """Give the pp GroupCoordinator a second communicator and route its collectives onto it.

    Installed from :func:`eager_init_process_group` rather than at import: sitecustomize runs
    before vllm exists, but every GroupCoordinator is built AFTER `init_process_group`, so by then
    the module is importable and no coordinator has been missed.
    """
    from vllm.distributed.parallel_state import GroupCoordinator

    original_group_init = GroupCoordinator.__init__

    def group_init(self, *args, **kwargs):
        original_group_init(self, *args, **kwargs)
        # Sentinel-valued on every coordinator, so the wrappers below ask what it IS rather than
        # whether it exists. `unique_name` is "<group_name>:<n>" (parallel_state._get_unique_name).
        self.collective_group = None
        if self.unique_name.rsplit(":", 1)[0] not in COLLECTIVE_SIBLING_GROUPS or self.world_size <= 1:
            return
        # Collective over the WORLD: it mints one group per rank set, so every rank must reach it
        # in the same order. Coordinator construction is already in lockstep, which is why this
        # sits here and not at the first collective, where only the group's own ranks would call.
        self.collective_group = self.make_sibling_device_group(group_desc="external_pp_collectives")
        print(f"[external-eager-pg] {self.unique_name}: collectives split onto a sibling communicator",
              file=sys.stderr,
              flush=True)

    def on_collective_group(method):
        """Run ``method`` with ``device_group`` pointing at the sibling.

        Swapping the attribute rather than threading a group argument through: the collectives all
        read ``self.device_group`` at entry, and `broadcast_tensor_dict` even accepts a ``group``
        parameter and then overwrites it, so the attribute is the only honest seam. Single-threaded
        per rank on this path -- the PP handler uses a side CUDA stream, not a side thread.
        """

        def wrapper(self, *args, **kwargs):
            sibling = self.collective_group
            if sibling is None:
                return method(self, *args, **kwargs)
            main_group = self.device_group
            self.device_group = sibling
            try:
                return method(self, *args, **kwargs)
            finally:
                self.device_group = main_group

        return wrapper

    GroupCoordinator.__init__ = group_init
    GroupCoordinator.broadcast = on_collective_group(GroupCoordinator.broadcast)
    GroupCoordinator.broadcast_object_list = on_collective_group(GroupCoordinator.broadcast_object_list)
    GroupCoordinator.broadcast_tensor_dict = on_collective_group(GroupCoordinator.broadcast_tensor_dict)


SPLIT_PP_COLLECTIVES = os.environ.get("VLLM_PP_COLLECTIVE_SPLIT", "1") == "1"


def eager_init_and_split(*args, **kwargs):
    """`init_process_group`, then install the pp collective split on top of the fresh world."""
    result = eager_init_process_group(*args, **kwargs)
    if SPLIT_PP_COLLECTIVES:
        patch_pp_collectives()
    return result


dist.init_process_group = eager_init_and_split
dist.new_group = eager_new_group

# Opt-in stack dumper. py-spy is not in this image, so the only way to see WHERE a worker is
# blocked during the ~200s ramp (599301: prefill completes, then decode does not start) is to
# ask the process itself. SIGUSR1 then dumps every thread's Python stack to stderr, which lands
# in that rank's vllm log. Off unless DUMP_STACKS_ON_SIGUSR1 is set, so normal runs are
# byte-identical.
if os.environ.get("DUMP_STACKS_ON_SIGUSR1"):
    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=True)
    print("[external-eager-pg] SIGUSR1 stack dumper armed", file=sys.stderr, flush=True)
