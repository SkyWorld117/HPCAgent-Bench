# vLLM on CSCS AMD MI300 with Slingshot

This bundle builds a persistent CSCS Container Engine image in a one-time
rootless Podman Slurm job. It adapts the supplied GH200/Kubernetes setup to the
AMD ROCm/RCCL stack and to CSCS Container Engine hooks.

## Validated design

The GH200 image manually builds CUDA-specific networking pieces and later
mounts Cray/CXI resources into a privileged Kubernetes pod. That design should
not be translated layer by layer.

| GH200/Kubernetes design | MI300/Container Engine design |
|---|---|
| CUDA and NCCL | ROCm and RCCL; PyTorch still names the backend `nccl` |
| Build and bake `aws-ofi-nccl` | Enable the CSCS `aws_ofi_nccl` EDF hook |
| Build GDRCopy | Omit it; it is NVIDIA-specific |
| Build a private matching libfabric | Install generic libfabric runtime/tools; let the CXI hook replace it at runtime |
| Mount `/opt/cray`, `libcxi`, `/dev/cxi*`; privileged and host networking | Omit Kubernetes host mounts/security configuration; the CSCS hooks inject supported libraries, dependencies and devices |
| aarch64 image | AMD MI300 images are x86_64 |

The image contains ROCm/PyTorch, vLLM, Ray, generic libfabric and diagnostics.
The supported RCCL/OFI/CXI network stack is selected at runtime in the EDF.

## Important compatibility boundary

The current public CSCS hook documentation lists only `rocm5` and `rocm6` for
AMD/RCCL, and says both are statically linked to specific ROCm versions. It does
not list a `rocm7` variant.

Upstream vLLM now recommends its official image:

```text
vllm/vllm-openai-rocm:<PINNED_TAG>
```

However, current official vLLM images have moved to newer ROCm stacks. Do not
pair a moving `latest` image with the CSCS `rocm6` hook merely because both are
for AMD.

For a concrete, known ROCm 6 compatibility pin, this bundle defaults to:

```text
docker.io/rocm/vllm:rocm6.4.1_vllm_0.10.1_20250909
```

That older AMD image is documented for ROCm 6.4.1 and vLLM 0.10.1. Upstream
vLLM now marks the `rocm/vllm` image family deprecated in favour of
`vllm/vllm-openai-rocm`; it is used here only because it gives an explicit ROCm
6 pin matching the currently documented CSCS hook. Replace it with an official
upstream image as soon as Opus exposes a compatible ROCm hook or CSCS provides a
validated Opus image/tag.

The build job always checks:

```python
import torch
print(torch.version.hip)
```

and refuses to import the image unless the result begins with `6.`.

## What public CSCS documentation does not establish

The public pages do not describe an `Opus` vCluster or its Slurm account,
partition, GPU-resource syntax, host network-stack versions, or custom hook
variants. They also do not list an AMD vLLM Alps Extended Image. Therefore this
bundle deliberately leaves the account/partition/GPU directives as placeholders
and uses the vCluster's configured default `com.hooks.netstack.source`.

If Opus has private documentation or a custom `rocm7`/dynamic AMD hook, follow
that information instead of the public `rocm6` assumption.

## Files

- `Containerfile`: extends the pinned ROCm/vLLM image, installs libfabric tools,
  and ensures Ray is present.
- `build_once.slurm`: rootless Podman build plus same-allocation Enroot import.
- `vllm-mi300-rocm6.toml`: Container Engine EDF with the RCCL/OFI hook.
- `run_smoke.slurm`: two-node Slingshot/RCCL validation job.
- `rccl_smoke.py`: cross-node PyTorch all-reduce test.

## One-time build

Edit these lines in `build_once.slurm`:

```bash
#SBATCH --account=REPLACE_ME
#SBATCH --partition=REPLACE_ME
```

Submit from this directory:

```bash
sbatch build_once.slurm
```

The defaults are equivalent to:

```bash
export VLLM_ROCM_IMAGE=docker.io/rocm/vllm:rocm6.4.1_vllm_0.10.1_20250909
export RAY_SPEC='ray[default]==2.48.0'
sbatch build_once.slurm
```

The Ray pin is a conservative reproducibility choice for this older vLLM
generation, not an Opus-certified matrix. If the base image already includes
Ray, the Containerfile preserves it. Override `RAY_SPEC` only after testing the
chosen vLLM/Python combination.

The job performs the following operations in one allocation:

1. Writes Podman's rootless storage configuration under `/dev/shm/$USER`.
2. Builds the OCI image on the compute node.
3. Checks the PyTorch, ROCm, vLLM, Ray and libfabric installation.
4. Rejects a non-ROCm-6 image.
5. Imports the image to
   `$SCRATCH/ce-images/vllm-mi300-rocm6.sqsh` before the ephemeral Podman store
   disappears.

It also applies CSCS's recommended Lustre striping to the image directory when
`lfs` is available.

## Equivalent interactive build

After allocating a compute-node shell, run from this directory:

```bash
CONTAINERS_CONF_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/containers"
mkdir -p "$CONTAINERS_CONF_DIR"
cat > "$CONTAINERS_CONF_DIR/storage.conf" <<EOF_STORAGE
[storage]
driver = "overlay"
runroot = "/dev/shm/$USER/runroot"
graphroot = "/dev/shm/$USER/root"
EOF_STORAGE

podman info | grep -A 4 'store:'

export VLLM_ROCM_IMAGE=docker.io/rocm/vllm:rocm6.4.1_vllm_0.10.1_20250909
podman build --pull=always \
  --build-arg VLLM_ROCM_IMAGE="$VLLM_ROCM_IMAGE" \
  --build-arg 'RAY_SPEC=ray[default]==2.48.0' \
  -t vllm-mi300-rocm6:local \
  -f Containerfile .

podman run --rm --entrypoint python3 vllm-mi300-rocm6:local \
  -c 'import torch, vllm, ray; print(torch.version.hip, vllm.__version__, ray.__version__)'

mkdir -p "$SCRATCH/ce-images"
enroot import -x mount \
  -o "$SCRATCH/ce-images/vllm-mi300-rocm6.sqsh" \
  podman://vllm-mi300-rocm6:local
```

The import must occur in the same Slurm allocation as the build because the
recommended Podman graph store resides on ephemeral `/dev/shm`.

## Container Engine configuration

Export the local image path:

```bash
export VLLM_MI300_SQSH="$SCRATCH/ce-images/vllm-mi300-rocm6.sqsh"
```

The EDF contains:

```toml
[annotations]
com.hooks.aws_ofi_nccl.enabled = "true"
com.hooks.aws_ofi_nccl.variant = "rocm6"
```

The hook supports RCCL, injects the AWS OFI plugin, implicitly enables the CXI
hook, and configures the Slingshot-related environment. Do not also add the CXI
annotation, and do not copy the GH200 image's hard-coded `NCCL_NET_PLUGIN`,
`FI_PROVIDER`, `LD_LIBRARY_PATH`, `/opt/cray` mounts, privileged setting or host
networking into this EDF.

The EDF also carries CSCS's PyTorch recommendations:

```toml
TORCH_NCCL_ASYNC_ERROR_HANDLING = "1"
MPICH_GPU_SUPPORT_ENABLED = "0"
```

It mounts only `${SCRATCH}` rather than all of `$HOME`. Add `/capstor` or
`/iopsstor` only if they exist on Opus and are required. For heavy random-access
model/cache workloads, CSCS recommends an IOPS-oriented filesystem when one is
available; large sequential data and checkpoints are generally better suited
to Capstor.

## Single-node preflight

```bash
export VLLM_MI300_SQSH="$SCRATCH/ce-images/vllm-mi300-rocm6.sqsh"

srun --environment=./vllm-mi300-rocm6.toml \
  bash -lc '
    python3 - <<"PY"
import torch
print("torch:", torch.__version__)
print("HIP:", torch.version.hip)
print("visible GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU:", p.name)
    print("architecture:", getattr(p, "gcnArchName", "unknown"))
PY
    fi_info -p cxi
  '
```

`fi_info -p cxi` is the documented CSCS check that the CXI provider is visible
inside the container.

## Two-node Slingshot/RCCL test

Edit the account, partition and Opus-specific GPU allocation directive in
`run_smoke.slurm`, then run:

```bash
sbatch run_smoke.slurm
```

The job stages the Python test under `${SCRATCH}` because that is the path the
EDF mounts. It then:

1. runs `fi_info -p cxi` on both nodes;
2. initializes PyTorch's `nccl` backend, which is implemented by RCCL on ROCm;
3. performs a cross-node all-reduce;
4. prints the HIP version, GPU model and GCN architecture.

With `NCCL_DEBUG=INFO`, verify in the Slurm output that the OFI plugin is loaded
and that the job does not fall back to sockets.

## Why MPI, OpenMPI, NVSHMEM and CPE are not included

vLLM's normal distributed inference path uses Ray plus PyTorch/RCCL, not MPI.
The MPICH, Cray MPICH and OpenMPI pages are relevant only if another component
of your workload explicitly launches MPI ranks. In that case, the MPI library
must be built/configured for libfabric and Slingshot separately.

NVSHMEM is an NVIDIA GPU programming interface, so it does not belong in this
AMD image. CPE is not needed for vLLM and the public CSCS page says CPE has
limited support and does not list an MI300 image.

## Known-issue notes

- Use a glibc-based Ubuntu image. CSCS documents Alpine/musl incompatibilities
  with hooks that call `ldconfig`.
- CSCS documents a local Ubuntu package mirror workaround if `apt` is slow.
  The published example uses the `ubuntu-ports` repository and was validated on
  NVIDIA/Arm images; this bundle does not apply it blindly to the x86_64 AMD
  image. If package downloads fail on Opus, obtain the correct amd64 mirror
  configuration from CSCS.
- Keep `srun --environment=...`; CSCS discourages putting `--environment` in an
  `#SBATCH` directive.
- Use an absolute EDF path or a relative path beginning with `./`.

## Production checklist

- Pin the base image by immutable digest after the first successful build.
- Record `torch.__version__`, `torch.version.hip`, `vllm.__version__` and
  `ray.__version__` from the build log.
- Keep the ROCm-major guard.
- Change `NCCL_DEBUG` from `INFO` to `WARN` after validation.
- Run a real vLLM tensor/pipeline-parallel test in addition to the small RCCL
  collective.
- Confirm the Opus-specific Ray bind address and network interface before a
  multi-node serving deployment; the public CSCS pages do not document those
  details for Opus.
