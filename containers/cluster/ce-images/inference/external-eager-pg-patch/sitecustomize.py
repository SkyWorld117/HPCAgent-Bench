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


# Eager init (device_id set) serializes unbatched P2P against the group's other traffic, and
# `received 1024 instead of 256` on isend_tensor_dict -- a receiver matching the WRONG message --
# killed 3 of 4 four-node probes in warmup (603524/603714/603718). 0 drops device_id everywhere,
# returning to lazy init while KEEPING the collective/P2P split below. Default 1 = today.
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


# Move the pipeline group's collectives to their own communicator.
#
# vLLM puts inter-stage activation transfer (isend/irecv_tensor_dict) and ordinary collectives on
# ONE device_group, which eager init above makes fatal: torch serializes unbatched P2P against
# everything else on the group. 600262/600263 both died there, a 4-element broadcast stuck at
# SeqNum=1 for the full 600 s watchdog while P2P was in flight; neither decoded a token.
#
# Upstream ships this remedy for exactly one broadcast (make_sibling_device_group in
# v1/worker/gpu/pp_utils.py); this extends it to every collective, leaving device_group carrying
# P2P alone. Membership is identical, so global rank ids still address the same processes.
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

# The MLA chunked-prefill context path transposes its log-sum-exp on ROCm.
#
# `mask_empty_context` unpacks lse as [num_heads, num_tokens] and sizes its mask from the second
# dimension. That holds for vllm-flash-attn; ROCm takes the upstream-flash_attn branch of
# `_flash_attn_varlen_diff_headdims`, which returns [num_tokens, num_heads], so the mask comes out
# sized num_heads and the fill raises "expanded size (3591) must match the existing size (16)".
# 603980 died there ~58 min in, taking all 121 agents with it.
#
# Reached only when some prefill in a chunk has run out of context and others have not, so it
# needs concurrent prefills of UNEQUAL length across chunks -- and `chunked_prefill_workspace_size`
# is capped at 64k in code with no CLI knob, so a busy server chunks small and hits it constantly.
# No config escape either: ROCm's prefill priority is [ROCM_AITER_FA, FLASH_ATTN] and aiter's
# master switch breaks MLA prefill on gfx942.
#
# Transposes ONLY the view this helper gets; the caller's own reference feeds merge_attn_states,
# which already handles the ROCm layout. A transposed view is free -- the helper passes
# stride(0)/stride(1) to the triton kernel. The swap fires only on the unambiguous transposed
# shape, so any other layout still raises exactly as today.
FIX_MLA_LSE_LAYOUT = os.environ.get("VLLM_FIX_MLA_LSE_LAYOUT", "1") == "1"


def patch_mla_empty_context_mask() -> None:
    """Point mla_attention's `mask_empty_context` name at a layout-tolerant wrapper.

    mla_attention does `from ... import mask_empty_context`, so the name has to be rebound in
    THAT module -- patching the defining module would leave the existing binding untouched.
    """
    from vllm.model_executor.layers.attention import mla_attention

    original = mla_attention.mask_empty_context

    def mask_empty_context(lse, output, query_start_loc, context_start_loc):
        # [num_tokens, num_heads] against an output of [num_tokens, num_heads, head_dim]. The
        # square case is genuinely ambiguous, so leave it to the original rather than guess.
        if (lse.ndim == 2 and output.ndim == 3 and lse.shape[0] == output.shape[0] and lse.shape[1] == output.shape[1]
                and output.shape[0] != output.shape[1]):
            lse = lse.transpose(0, 1)
        return original(lse, output, query_start_loc, context_start_loc)

    mla_attention.mask_empty_context = mask_empty_context
    print("[external-eager-pg] mla mask_empty_context: ROCm lse layout tolerated", file=sys.stderr, flush=True)


def eager_init_and_split(*args, **kwargs):
    """`init_process_group`, then install the pp collective split on top of the fresh world."""
    result = eager_init_process_group(*args, **kwargs)
    if SPLIT_PP_COLLECTIVES:
        patch_pp_collectives()
    if FIX_MLA_LSE_LAYOUT:
        patch_mla_empty_context_mask()
    return result


dist.init_process_group = eager_init_and_split
dist.new_group = eager_new_group

# Opt-in stack dumper: py-spy is not in this image, so asking the process itself is the only way
# to see where a worker blocks during the ramp (599301: prefill completes, decode never starts).
# SIGUSR1 dumps every thread's stack to that rank's vllm log.
if os.environ.get("DUMP_STACKS_ON_SIGUSR1"):
    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=True)
    print("[external-eager-pg] SIGUSR1 stack dumper armed", file=sys.stderr, flush=True)
