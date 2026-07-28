# DESIGN: container protocol -- a task declares needs, a family renders them

## State today

`container_backends.txt` makes the BACKEND data: family (`oci` / `sif` / `ce`), kind
(`exec` wrapper vs `srun_env` flag), verb, bind, workdir, env, `gpu.nvidia`,
`gpu.amd`, `image_form`, `rootless`. `containers.py` and `run_agent_in_container.sh`
both read it, so the two launch paths cannot drift.

What is NOT data: what the RUN needs. That is spelled per site, by hand, in four
places -- the GPU flag is picked from a hardware string, the mounts are typed into
each EDF, the Cray fabric hook is typed into `[annotations]`, and the need for a host
network is implicit in whoever remembered it. Four hand-edits that must agree, with
nothing checking that they do. `foundation.toml.example` already carries a comment
warning that forgetting the fabric hook reads as poor scaling rather than as a
misconfigured launch. That is the failure mode this design removes.

## Decision: one request object, one renderer per family

```python
@dataclass(frozen=True)
class ContainerRequest:
    gpu: str = "none"        # none | nvidia | amd
    mounts: Tuple[str, ...] = ()
    network: str = "none"    # none | host   (host only for a served endpoint)
    fabric: bool = False     # rank-to-rank traffic -> Cray hook / UCX passthrough
```

Four fields. Not five. Everything else the launcher already knows (image, workdir,
env) is derived, not declared.

Families render it two different ways, and that split is the point:

- `kind=exec` (podman / docker / apptainer) -> **argv**. Same code path as today,
  reading the same rows, plus two new ones: `<b>.network.host`, `<b>.mount`.
- `kind=srun_env` (ce) -> **a file**. CE takes no wrapper argv; the request becomes
  EDF fields: `mounts`, `[annotations] com.hooks.*`. So the renderer emits TOML.

That second renderer is what kills the hand-written EDF: `foundation.toml.example`
becomes the OUTPUT of `hpcagent-bench container edf --gpu=none --fabric=0`, not a
file someone keeps in sync. The MPI track passes `--fabric=1` and the hook appears
because it was asked for, not because it was remembered.

## New rows needed

```
podman.network.host=--network host      # already inside .verb today; hoist it
docker.network.host=--network host
apptainer.network.host=                 # apptainer shares the host net by default
ce.hook.fabric=com.hooks.aws_ofi_nccl.enabled
ce.hook.fabric.variant=com.hooks.aws_ofi_nccl.variant
```

`gpu.nvidia` / `gpu.amd` already exist. No other row is new.

## Refusal, never silent drop

A field a family cannot satisfy raises, naming the field and the family. Precedent is
already set: `srun_container_flags` raises when `ce` has no EDF, and `default_image("ce")`
raises rather than inventing a path. A container that quietly ran without the GPU it
asked for produces a plausible, wrong number -- which is worse than a crash.

## Gate

- Same request, rendered for podman / docker / apptainer, differs ONLY in the flag
  spellings the file declares. This is the existing parity test, extended.
- The generated EDF parses as TOML and, for `--fabric=1`, contains the hook; for
  `--fabric=0`, contains no `[annotations]` at all.
- `hpcagent-bench container edf` output for the foundation track is byte-identical to
  the checked-in example, or the example is deleted in favour of generating it.
