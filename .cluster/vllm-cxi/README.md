# vllm-cxi

Custom vLLM container image with Slingshot (HSN) NCCL transport for CSCS
forno-tds cluster.

---

## Contents

1. [Background](#background)
2. [The software stack](#the-software-stack)
3. [Why each component is needed](#why-each-component-is-needed)
4. [What is NOT in this image](#what-is-not-in-this-image)
5. [Building the image](#building-the-image)
6. [Runtime pod configuration](#runtime-pod-configuration)
   - [Volume mounts](#volume-mounts)
   - [Network](#network)
   - [Security context](#security-context)
   - [Environment variables](#environment-variables)
7. [Updating component versions](#updating-component-versions)
8. [Version matrix](#version-matrix)
9. [Troubleshooting](#troubleshooting)

---

## Background

The official `vllm/vllm-openai` image uses NCCL for GPU collective operations
(all-reduce, all-gather) but ships with NCCL's built-in TCP and InfiniBand
transport layer.  On forno-tds the inter-node interconnect is
[HPE Slingshot](https://www.hpe.com/us/en/compute/hpc/slingshot-interconnect.html),
a high-performance network based on the libfabric OFI API.  The standard NCCL
TCP path ignores this fabric entirely and falls back to a slow Ethernet
path — or fails outright for multi-node jobs.

To use Slingshot, NCCL must be told to route its traffic through the
`aws-ofi-nccl` plugin, which bridges NCCL's network API to libfabric's OFI
API, which then drives the CXI provider for Slingshot.

This repo builds and maintains that augmented image.

---

## The software stack

```
vLLM  (tensor-parallel / pipeline-parallel inference)
 │
 └── PyTorch  torch.distributed
       │  (manages process groups, wraps collective calls)
       │
       └── NCCL  (GPU collective ops: allreduce, allgather, …)
             │  NCCL_NET_PLUGIN=/opt/aws-ofi-nccl/lib/libnccl-net.so
             │
             └── aws-ofi-nccl  (NCCL net plugin — dlopen'd at startup)
                   │  OFI_NCCL_PROTOCOL=SENDRECV
                   │
                   └── libfabric  (OFI abstraction API)
                         │  FI_PROVIDER=cxi
                         │
                         └── CXI provider  (Slingshot driver — from host mount)
                               │
                               └── /dev/cxi[0-3]  (one per HSN port per node)
                                     └── Slingshot HSN (hsn0-3 interfaces)
```

Each arrow represents a runtime dependency loaded from the layer below.
Only the top three layers (vLLM, PyTorch, NCCL) are baked into the image.
Everything below `aws-ofi-nccl` comes from the node at runtime.

---

## Why each component is needed

### aws-ofi-nccl

NCCL exposes a net-plugin API: at startup it looks for a shared library
named by `NCCL_NET_PLUGIN` and calls its `ncclNet_v*` vtable.  Without a
plugin, NCCL defaults to its built-in TCP transport which bypasses Slingshot
entirely.  `aws-ofi-nccl` provides an implementation of that vtable backed
by libfabric.

### libfabric (built from source, not the Cray version)

`aws-ofi-nccl` needs libfabric *headers* at build time and a *stub SO* to
link against.  We build a generic libfabric from source at the same version
as the Cray-supplied one on the compute nodes so the ABI matches.

> **ABI compatibility note**: libfabric changed its ABI from version 1.6 to
> 1.7 at the 1.20.0 release.  If the build-time version and the runtime
> (Cray) version straddle this boundary, the plugin will fail to load with
> `symbol lookup error` or a segfault.
> Check the node: `ls /opt/cray/libfabric/` and set `LIBFABRIC_VER`
> accordingly.

The actual CXI provider that talks to Slingshot hardware is NOT compiled into
our libfabric; it comes from the Cray installation on the host node via the
`/opt/cray` hostPath mount.

### GDRCopy

GPU Direct RDMA copy library.  Enables zero-copy transfers between GPU memory
and the NIC, bypassing the CPU bounce buffer.  Required by the libfabric build
(`--with-gdrcopy`) to expose GPU Direct paths.  Built with
`--enable-gdrcopy-dlopen` so the library is loaded lazily; it degrades
gracefully on nodes where GDRCopy is unavailable.

### Why vLLM itself is NOT rebuilt

vLLM is unaware of NCCL's network transport layer.  It calls
`torch.distributed` collectives which call into NCCL, and NCCL then calls the
plugin.  No vLLM or PyTorch source changes are needed.  This image simply
adds the plugin to the official vLLM image.

---

## What is NOT in this image

The following are deliberately absent and must be provided by the node at
runtime:

| What | Where on the node | Mounted at |
|------|-------------------|------------|
| Cray libfabric with CXI provider | `/opt/cray/libfabric/<ver>/` | `/opt/cray/` |
| libcxi.so.1 | `/usr/lib64/libcxi.so.1` | `/host/lib64/libcxi.so.1` |
| libcxiutils.so.0 | `/usr/lib64/libcxiutils.so.0` | `/host/lib64/libcxiutils.so.0` |
| CXI NIC devices | `/dev/cxi[0-3]` | `/dev/cxi[0-3]` |
| HSN network interfaces | `hsn0`, `hsn1`, `hsn2`, `hsn3` | via `hostNetwork: true` |

---

## Building the image

### Via GitLab CI (recommended)

Push to the default branch or create a git tag:

```
git tag v0.9.1-ofi1.17.2
git push origin v0.9.1-ofi1.17.2
```

The pipeline builds each arch natively (kaniko can't cross-compile) and pushes
arch-suffixed tags, where `<arch>` is `x86_64` or `aarch64`:
- `jfrog.svc.cscs.ch/docker/vllm-cxi:latest-<short-sha>-<arch>` — always (immutable)
- `jfrog.svc.cscs.ch/docker/vllm-cxi:latest-<arch>` — rolling
- `jfrog.svc.cscs.ch/docker/vllm-cxi:<tag>-<arch>` — on a git tag

forno-tds HPC nodes are GH200 (aarch64); use the `-aarch64` tag there.

Use the `:latest-<short-sha>-<arch>` or `:<tag>-<arch>` form in production to pin
to an immutable image.  Never pin `:latest-<arch>` in ArgoCD values.

### Locally (for debugging the Dockerfile)

```bash
# CUDA_BUILD_IMAGE and VLLM_IMAGE have no defaults — both must be passed.
docker build \
  --build-arg CUDA_BUILD_IMAGE=docker.io/nvidia/cuda:13.1.0-devel-ubuntu24.04 \
  --build-arg VLLM_IMAGE=docker.io/vllm/vllm-openai:glm52 \
  -t vllm-cxi:local .

# Override component versions (defaults shown)
docker build \
  --build-arg CUDA_BUILD_IMAGE=docker.io/nvidia/cuda:13.1.0-devel-ubuntu24.04 \
  --build-arg VLLM_IMAGE=docker.io/vllm/vllm-openai:glm52 \
  --build-arg LIBFABRIC_VER=2.3.1 \
  --build-arg AWS_OFI_NCCL_VER=v1.19.1 \
  --build-arg GDRCOPY_VER=2.5.1 \
  -t vllm-cxi:local .

# Inspect the built plugin
docker run --rm vllm-cxi:local ls -lh /opt/aws-ofi-nccl/lib/
docker run --rm vllm-cxi:local ldd /opt/aws-ofi-nccl/lib/libnccl-net.so
```

---

## Runtime pod configuration

### Volume mounts

```yaml
extraVolumes:
  - name: cray-libs
    hostPath:
      path: /opt/cray
  - name: libcxi
    hostPath:
      path: /usr/lib64/libcxi.so.1
      type: File
  - name: libcxiutils
    hostPath:
      path: /usr/lib64/libcxiutils.so.0
      type: File
  - name: dev-cxi0
    hostPath:
      path: /dev/cxi0
  - name: dev-cxi1
    hostPath:
      path: /dev/cxi1
  - name: dev-cxi2
    hostPath:
      path: /dev/cxi2
  - name: dev-cxi3
    hostPath:
      path: /dev/cxi3

extraVolumeMounts:
  - name: cray-libs
    mountPath: /opt/cray
    readOnly: true
  - name: libcxi
    mountPath: /host/lib64/libcxi.so.1
    readOnly: true
  - name: libcxiutils
    mountPath: /host/lib64/libcxiutils.so.0
    readOnly: true
  - name: dev-cxi0
    mountPath: /dev/cxi0
  - name: dev-cxi1
    mountPath: /dev/cxi1
  - name: dev-cxi2
    mountPath: /dev/cxi2
  - name: dev-cxi3
    mountPath: /dev/cxi3
```

> Why only `libcxi.so.1` and `libcxiutils.so.0` and not all of `/usr/lib64`?
> On SLES 15 SP6, the host's `libssl.so.3` requires GLIBC 2.38, which is
> newer than the container's libc.  Mounting the entire `/usr/lib64` causes
> Python's `import ssl` to crash at startup.  Mount only the CXI-specific
> files needed by libfabric's dlopen path.

### Network

```yaml
hostNetwork: true
```

Required so the High-Speed Network interfaces (`hsn0`–`hsn3`) are visible
inside the container.  Without this, NCCL falls back to the management
network (`nmn0` / `nid*`) which is not routed for payload traffic.

### Security context

```yaml
securityContext:
  privileged: true
```

The `/dev/cxi*` character devices are hostPath-mounted, but kubelet does not
add them to the container's cgroup device allow-list automatically.  Without
`privileged: true`, `open()` on these devices returns `EPERM` even with
individual capabilities (`IPC_LOCK`, `SYS_ADMIN`, etc.) granted.

The proper long-term fix is a Kubernetes device plugin that calls the
`Allocate` gRPC method to populate the cgroup allow-list via the kubelet.
Until that is deployed, `privileged: true` is required.

### Environment variables

All of the following are set in the pod spec, not in the image, because some
values (the libfabric version path, the pod IP) are not known at image build
time.

```yaml
env:
  # -------------------------------------------------------------------------
  # Library path — must include all three layers:
  #   1. aws-ofi-nccl plugin libs (in the image)
  #   2. Cray libfabric with CXI provider (from /opt/cray host mount)
  #   3. /host/lib64 where libcxi.so.1 is mounted
  #
  # forno-tds nodes have libfabric 2.3.1 at /opt/cray/libfabric/2.3.1/
  # Verified: kubectl exec cray-debug -- ls /opt/cray/libfabric/ → 2.3.1
  # -------------------------------------------------------------------------
  - name: LD_LIBRARY_PATH
    value: /opt/aws-ofi-nccl/lib:/opt/cray/libfabric/2.3.1/lib64:/host/lib64

  # -------------------------------------------------------------------------
  # libfabric / CXI provider
  # -------------------------------------------------------------------------

  # Force libfabric to use the CXI provider (Slingshot).
  - name: FI_PROVIDER
    value: "cxi"

  # GPU Direct RDMA level.  PHB = GPU and CXI NIC share the same PCIe
  # complex → allowed to do peer DMA without going through the CPU.
  - name: NCCL_NET_GDR_LEVEL
    value: "PHB"

  # Multi-rail: stripe NCCL traffic across all four HSN interfaces.
  - name: NCCL_CROSS_NIC
    value: "1"

  # Fallback socket path also uses HSN, not the management network (nmn0).
  - name: NCCL_SOCKET_IFNAME
    value: "hsn0,hsn1,hsn2,hsn3"

  # Required by Cray's CXI provider.  Disables host memory registration
  # (not needed when using GPU Direct RDMA).
  - name: FI_CXI_DISABLE_HOST_REGISTER
    value: "1"

  # Completion queue size.  Larger value prevents overflows under high message
  # rates (multi-GPU all-reduce with many small messages).
  - name: FI_CXI_DEFAULT_CQ_SIZE
    value: "131072"

  # Hybrid matching mode: CXI picks between hardware and software tag matching
  # per message.  Avoids overflow in software-matching mode during burst load.
  - name: FI_CXI_RX_MATCH_MODE
    value: "hybrid"

  # -------------------------------------------------------------------------
  # aws-ofi-nccl protocol
  # -------------------------------------------------------------------------

  # CXI does not implement the RDMA FI capability set that aws-ofi-nccl 1.10+
  # requests by default.  SENDRECV only requires FI_EP_RDM + FI_TAGGED, which
  # CXI fully supports.  Without this, aws-ofi-nccl logs
  # "fi_getinfo: -38 (ENOSYS)" and falls back or crashes.
  # Note: OFI_NCCL_ prefix (not NCCL_OFI_) — set by the OFI_NCCL_PARAM_STR
  # macro in aws-ofi-nccl source.
  - name: OFI_NCCL_PROTOCOL
    value: "SENDRECV"

  # -------------------------------------------------------------------------
  # NCCL debug logging — set to WARN in production; INFO here for bring-up
  # -------------------------------------------------------------------------
  - name: NCCL_DEBUG
    value: "INFO"
  - name: NCCL_DEBUG_SUBSYS
    value: "NET,INIT"

  # libfabric logging: warn level to reduce noise; CXI provider only.
  - name: FI_LOG_LEVEL
    value: "warn"
  - name: FI_LOG_PROV
    value: "cxi"

  # -------------------------------------------------------------------------
  # Gloo (PyTorch CPU side-channel — used for barrier/broadcast ops)
  # -------------------------------------------------------------------------

  # The pod hostname (nidNNNNNN) is not in DNS, so Gloo's default hostname
  # resolution falls back to 127.0.0.1 and every remote rank gets "connection
  # refused".  Pin Gloo to hsn0 (the routed HSN interface, 172.28.x/16).
  - name: GLOO_SOCKET_IFNAME
    value: "hsn0"

  # -------------------------------------------------------------------------
  # vLLM / Ray — required for multi-node pipeline-parallel setups
  # -------------------------------------------------------------------------

  # vLLM needs VLLM_HOST_IP to equal the IP Ray binds to.  Under
  # hostNetwork: true, status.podIP equals the node's HSN IP (hsn0), which
  # is what the headless Service DNS resolves to.  Without this, Ray may bind
  # to loopback or the management NIC and workers cannot reach the head node.
  - name: VLLM_HOST_IP
    valueFrom:
      fieldRef:
        fieldPath: status.podIP
```

---

## Updating component versions

### Step 1 — check the libfabric version on the target nodes

```bash
kubectl exec -it <any-pod-on-hpc-node> -- ls /opt/cray/libfabric/
```

The directory name is the version.  Update `LIBFABRIC_VER` in the Dockerfile
(and `ARG` default) to match.  If the major ABI version changed (see note in
Dockerfile), test thoroughly before promoting to `:latest`.

### Step 2 — check the CUDA version in the target vLLM image

```bash
docker run --rm docker.io/vllm/vllm-openai:<new-tag> nvcc --version
```

Update `CUDA_BUILD_IMAGE` in the Dockerfile to match the same CUDA
major.minor version.

### Step 3 — update ARG defaults in the Dockerfile

```dockerfile
ARG VLLM_IMAGE=docker.io/vllm/vllm-openai:<new-version>
ARG CUDA_BUILD_IMAGE=docker.io/nvidia/cuda:<matching-version>-devel-ubuntu22.04
ARG LIBFABRIC_VER=<node-version>
ARG AWS_OFI_NCCL_VER=v<new-version>
ARG GDRCOPY_VER=<new-version-if-changed>
```

### Step 4 — tag the commit

Use the naming convention `v<vLLM>-ofi<aws-ofi-nccl>`:

```bash
git tag v0.9.1-ofi1.17.2
git push origin v0.9.1-ofi1.17.2
```

---

## Version matrix

| Image tag | arch | vLLM | CUDA build | libfabric | aws-ofi-nccl | GDRCopy |
|-----------|------|------|------------|-----------|--------------|---------|
| `latest-x86_64` | x86_64 | `glm52` | 13.1.0 | 2.3.1 | v1.19.1 | 2.5.1 |
| `latest-aarch64` | aarch64 | `glm52` | 13.1.0 | 2.3.1 | v1.19.1 | 2.5.1 |

Update this table when you create a new pinned tag.

---

## Troubleshooting

### NCCL does not load the plugin

Check that the plugin path is correct and the file exists:
```bash
# Inside the running pod:
ls -lh /opt/aws-ofi-nccl/lib/libnccl-net.so
echo $NCCL_NET_PLUGIN
```

Look for this line in the container logs at startup:
```
NCCL INFO NET/OFI Using aws-ofi-nccl ...
```
If you see `NCCL INFO NET Using internal Network Socket` instead, the plugin
was not loaded.

### FI_PROVIDER: No such device / fi_getinfo: -38 (ENOSYS)

The CXI provider could not be found.  Check:
1. `/dev/cxi[0-3]` are mounted and accessible (`ls -la /dev/cxi*`)
2. `/opt/cray` is mounted (`ls /opt/cray/libfabric/`)
3. `LD_LIBRARY_PATH` includes the correct versioned Cray libfabric path
4. `FI_LOG_LEVEL=debug FI_LOG_PROV=cxi` for verbose provider discovery logs

### EPERM when opening /dev/cxi*

The container is not running privileged.  Verify `securityContext.privileged: true`.
Checking:
```bash
# Inside the pod — should succeed with privileged: true
dd if=/dev/cxi0 of=/dev/null bs=1 count=0 2>&1
```

### Workers cannot reach the Ray head node (connection refused on port 6379)

`VLLM_HOST_IP` is not set or is set to the wrong interface.  Verify:
```bash
# Inside the pod:
echo $VLLM_HOST_IP
ip addr show hsn0
# The two IPs should match.
```

### NCCL hangs on ncclCommInitRank

Possible causes:
1. **Double plugin registration**: both `NCCL_NET=ofi` and `NCCL_NET_PLUGIN`
   are set.  In NCCL ≥ 2.26, `NCCL_NET=ofi` is removed — using it with
   `NCCL_NET_PLUGIN` causes double-registration.  Remove `NCCL_NET=ofi`.
2. **Protocol mismatch**: `OFI_NCCL_PROTOCOL` is not set to `SENDRECV`.
3. **libfabric ABI mismatch**: build-time vs runtime libfabric versions
   straddle an ABI boundary.  Check versions; rebuild with matching
   `LIBFABRIC_VER`.

### Python `import ssl` crashes at startup

`/usr/lib64` was mounted wholesale into the container, bringing in the host's
`libssl.so.3` which requires a newer GLIBC than the container provides.
Mount only the specific files needed (`libcxi.so.1`, `libcxiutils.so.0`),
not the entire directory.
