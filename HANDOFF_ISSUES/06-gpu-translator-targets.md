# 06 -- numpy->X GPU translator targets (CUDA C++, OpenACC Fortran, OpenMP C/C++)

**STATUS (checked 2026-07-25): STILL OPEN, NOT STARTED.** This is a design/requirements handoff,
not a bug with a runnable reproducer -- verified by absence: grepped
`hpcagent_bench/numpy_translators/src/` for `acc_set_cuda_stream`, `!$acc`, `__global__`,
`deviceptr` -- zero hits. No `openacc`/`cuda` emission code exists anywhere in the translator tree.
The CUDA/HIP names that DO exist (`hpcagent_bench/languages.py`, `flags.py`: `compose_cuda`,
`compose_hip`, `LINK_LANG_ORDER`) are for compiling pre-existing native **reference** sources
(`baseline_ref`), unrelated to this doc's ask (numpy->X emitting new GPU kernel bodies). Nothing to
mark fixed; still fully queued, scope lock from 2026-07-14 still the design of record.

**Owner:** hpcagent_bench (numpy->X). nest-forge consumes the emitted GPU source as an `ExternalCall` variant
and links each compiled `.a` into the DaCe-generated whole-program driver. This doc is a *design +
requirements* handoff: nest-forge M4 (GPU arena) needs the translator to emit GPU kernel sources, and
the ABI below is fixed by what the DaCe driver already provides.

## SCOPE (locked 2026-07-14, user decision)

GPU support = **two paths only**:
1. **DaCe-GPU backend** -- the `ExpandDaceReference` GPU variant (numpy->dace->`auto_optimize` GPU
   schedule->DaCe CUDA codegen). This is the reference/competitor row; it emits its own CUDA, NOT via
   the numpy->X emitter.
2. **Fortran OpenACC** -- the one numpy->X *external-leg* GPU target. `emit_bridge` decorates the
   existing Fortran emitter with `!$acc` directives + `deviceptr` + `acc_set_cuda_stream`.

Both bind the shared `__dace_current_stream` **directly** (no event bridge), so the env model stays
single-stream-clean (one cudart, one primary ctx, one device, DaCe owns the stream). **OpenMP-target is
DROPPED as a target** (kept below only as documented proof that it *could* be bridged). CUDA-C++ emit
and vendor-lib shims (cuBLAS/...) remain documented-feasible below but are **not in initial scope** --
add them only when a nest needs one.

## What nest-forge needs (the ask)

Add GPU kernel targets to `emit_bridge`/`numpyto_*`, scoped by language:

| target | language | mechanism | stream bind | new codegen or decoration? |
|--------|----------|-----------|-------------|----------------------------|
| `openacc` **(IN SCOPE)** | **Fortran only** | existing Fortran body + `!$acc` directives | direct (`acc_set_cuda_stream`+`async`) | **pragma-decoration of the Fortran emitter** |
| `cuda` (feasible, deferred) | C++ | `__global__` kernel + `<<<grid,block,0,stream>>>` launch | direct (`<<<...,stream>>>`) | **genuine new emitter** |
| vendor libs (`cublas`/`cufft`/`cusolver`) (feasible, deferred) | C++ shim | call the precompiled lib | direct (`cublasSetStream` etc.) | **shim only** (one extern-C wrapper, no kernel codegen) |
| ~~`openmp`~~ **DROPPED as a target** | ~~C/C++~~ | ~~`#pragma omp target`~~ | **NONE -- needs CUDA-event bridge** | not a target; documented below only as bridge proof |
| `hip` (later) | C++ | `hipLaunchKernelGGL` / `<<<>>>`; CUDA->HIP macro swap | direct | new emitter (alt backend, mutually exclusive with cuda) |

Rationale for the language scoping (user-set): OpenACC is the natural Fortran GPU path; OpenMP target
is the natural C/C++ GPU path; do not cross them (no OpenACC-C, no OpenMP-Fortran). CUDA/HIP are C++.

**The pragma-decoration insight is the cheap win.** For `openacc` and `openmp`, the compute body is the
*same* Fortran / C / C++ the translator already emits for the CPU targets. The GPU variant is that body
with directive lines added (which are inert comments to a non-GPU compiler) plus a device-pointer clause
and a stream binding. It is NOT a new codegen path -- it is a decoration layer over the existing emitters.
Only `cuda`/`hip` require true new emission (`__global__`, thread-index arithmetic, launch config).

## The ABI is fixed (device pointers + stream; the driver owns movement)

The extracted loop-nest **assumes all data is already resident on the GPU and coordinated elsewhere**.
The DaCe C++ driver owns every allocation, host<->device copy, and the stream. So each GPU kernel `.a`
entry receives **raw device pointers + the stream**, does zero data movement, and launches on the
passed stream. This extends nest-forge's current CPU extern-C entry
(`nestforge/libnode.py::proto_and_call`) with one trailing parameter:

```c
/* CPU today  */ extern "C" void <sym>(const double* A, double* out, int64_t N);
/* GPU (cuda) */ extern "C" void <sym>(const double* A, double* out, int64_t N, void* stream);
```

`stream` is a `cudaStream_t` (an opaque pointer). The driver passes DaCe's own stream, named
`__state->gpu_context->streams[i]` (or the pre-bound local `__dace_current_stream` inside a host
tasklet) -- see `dace/codegen/targets/cuda.py:1842` and `dace/codegen/targets/cpp.py:860`. The `A`/`out`
pointers are device addresses (`cudaMalloc`ed by the driver); the kernel must treat them as such and
must NOT copy.

## Per-target emission contract

### `cuda` (C++) -- new emitter
```cpp
__global__ void <sym>_kernel(const double* A, double* out, long N) {
    long i = blockIdx.x*(long)blockDim.x + threadIdx.x;
    if (i < N) out[i] = /* elementwise body */;
}
extern "C" void <sym>(const double* A, double* out, long N, void* stream) {
    int t = 256; long b = (N + t - 1) / t;
    <sym>_kernel<<<(unsigned)b, t, 0, (cudaStream_t)stream>>>(A, out, N);
}
```
Reductions map to a block/tree reduction or (simplest, correct) an `atomicAdd`-based accumulate; the
numpy oracle is the reference, tree/atomic reassociation is an FP-mode concern, not a correctness bug.

### `openacc` (Fortran) -- decorate the existing Fortran emitter
Take the emitted Fortran body and: (1) receive pointers as `type(c_ptr), value` + `c_f_pointer` them to
Fortran array pointers; (2) add `deviceptr(...)` so OpenACC does not map/copy already-resident data;
(3) bind the CUDA stream to an async queue and launch `async`:
```fortran
subroutine <sym>(A, out, N, stream) bind(C, name="<sym>")
  use iso_c_binding; use openacc
  type(c_ptr), value :: A, out, stream
  integer(c_long), value :: N
  real(c_double), pointer :: pa(:), po(:)
  integer(c_int) :: q, ierr
  call c_f_pointer(A, pa, [N]); call c_f_pointer(out, po, [N])
  q = 1
  ierr = acc_set_cuda_stream(q, stream)          ! bind async queue -> the driver's CUDA stream
  !$acc parallel loop deviceptr(pa, po) async(q)
  do i = 1, N
     po(i) = /* elementwise body */
  end do
end subroutine
```
`acc_set_cuda_stream` C signature: `int acc_set_cuda_stream(int async, void* stream)` -- declare an
explicit `bind(C)` interface (see the working prototype). This is the whole trick: same loop body, a
`deviceptr` clause, an `async(q)` launch, one stream-binding call.

**Queue id must be per-leg, not a hardcoded `q=1` (review finding, 2026-07-14).** The prototype hardcodes
`qid=1`, which is fine for a single-stream driver but a collision hazard once multiple OpenACC legs run on
DIFFERENT DaCe streams: `acc_set_cuda_stream(1, sA)` then `acc_set_cuda_stream(1, sB)` rebinds the SAME
async queue, so leg A's in-flight `async(1)` work serializes on the wrong stream. The emitter must give
each leg a distinct async-queue id (e.g. derived from the nest index or the stream index) rather than a
literal `1`.

### `openmp` (C / C++) -- DROPPED as a target (kept as documented bridge proof only)
**Out of scope per the 2026-07-14 scope lock** -- the GPU external-leg target is OpenACC Fortran; OpenMP
is not emitted. Retained here only because the prototype proves it *could* be bridged, and to record the
caveat below. **Bridge caveat (review finding, 2026-07-14):** the event bridge in `omp_probe.cpp` waits on
`(cudaStream_t)0`, i.e. it assumes the `omp target` region executes on CUDA's default stream. OpenMP
permits an internal non-default stream; on such a runtime the bridge would order the wrong stream and race.
It passes 0/30 here only because this nvhpc/CUDA build happens to dispatch OMP on stream 0 -- do NOT treat
the bridge as portable without querying the region's actual stream. (Another reason OpenMP is not a target.)

Same body; device pointers via `is_device_ptr(...)`; stream via OpenMP 5.1 `interop` (bind the target
region to the driver's CUDA stream) or run `nowait depend(...)` and let the driver event-sync:
```c
void <sym>(const double* A, double* out, long N, void* stream) {
    #pragma omp target teams distribute parallel for is_device_ptr(A, out)  /* + interop(stream) */
    for (long i = 0; i < N; ++i) out[i] = /* body */;
}
```
`is_device_ptr` is the OpenMP analogue of OpenACC `deviceptr`. **Stream binding: there is no standard for
it** (checked through OpenMP 6.0). Unlike OpenACC (`acc_set_cuda_stream(q,s)` + `async(q)` binds a
caller-supplied stream), OpenMP-target has *no directive equivalent to specify a CUDA stream*. What the
standard gives instead, and the exact directions:
- `nowait` (+ enclosing task) makes a `target` region async, but the RUNTIME picks the stream -- you cannot
  name it.
- the `interop` construct standardizes interop the OTHER way: `init(targetsync: obj)` makes OpenMP CREATE a
  foreign sync object (a stream OMP owns); `omp_get_interop_ptr(obj, omp_ipr_targetsync)` extracts it to
  hand OUTWARD to a foreign lib (cuBLAS...). `depend(...)` on the construct imposes happens-before ordering
  between OMP tasks and that foreign stream (both directions) -- the standard, OpenMP-native form of the
  event bridge below. There is NO standard "init interop FROM my existing cudaStream_t"; init always mints
  OMP's own.

Two standard-compliant options for our "DaCe owns the stream" ABI: (1) the **CUDA-event bridge** (keep
DaCe's stream, OMP runs default, driver orders with an event -- what the prototype does); or (2) **invert
ownership** -- let OMP's `interop init(targetsync)` create the stream and pass THAT to the CUDA/cuBLAS/
OpenACC legs (fully standard, but DaCe no longer owns the stream, so it clashes with this ABI). Prefer
CUDA/cuBLAS for the C/C++ GPU path; reach for OMP-target only when required, and bridge it.

**Empirically confirmed** (prototype, 2026-07-14): an OMP-target leg run after CUDA work on our stream with
no sync **races 29/30 runs**. The event bridge is proven to work (**0/30 races**):
```cpp
cuda_scale(..., ourStream);                       // work on the shared stream
cudaEventRecord(ev, ourStream);                   // mark it
cudaStreamWaitEvent((cudaStream_t)0, ev, 0);      // OMP's default stream waits for ours
omp_add10(...);                                    // OMP-target region, now ordered after cuda_scale
cudaEventRecord(ev, (cudaStream_t)0);             // mark OMP's work
cudaStreamWaitEvent(ourStream, ev, 0);            // our stream waits for OMP  -- no host sync anywhere
```
So OMP-target is usable but is the ONLY leg the driver must bridge; every other leg (CUDA, OpenACC,
cuBLAS/cuFFT/cuSOLVER) binds the stream directly. Prefer OpenACC for Fortran-GPU and CUDA/cuBLAS for
C++-GPU; reach for OMP-target only when required, and bridge it.

## External precompiled vendor libs (cuBLAS/cuFFT/cuSOLVER) -- the "pass a stream to an .a" case

The sharpest form of the question: can we hand our stream to a *closed-source, precompiled* library we do
not compile? Yes. Every NVIDIA math library exposes a stream setter that binds ALL subsequent calls to the
given stream, and every one takes raw device pointers:
- cuBLAS: `cublasSetStream(handle, stream)`
- cuFFT:  `cufftSetStream(plan, stream)`
- cuSOLVER: `cusolverDnSetStream(handle, stream)`
The nest-forge shim is one extern-C wrapper (`extern "C" void <sym>(double* d_x, long n, void* stream)`)
that does `cublasSetStream(h, (cudaStream_t)stream); cublasDscal(h, n, &a, d_x, 1);` -- no kernel codegen,
no data movement. Link with `nvc++ ... -cudalib=cublas` (pulls the shared cuBLAS; the static
`libcublas_static.a` also needs `libcublasLt` -- prefer shared, and never static-link cudart, see below).
The `cudaStream_t` handle is just a pointer and is safe to pass across shared-library boundaries as long as
all parties share the one GPU device context.

## Proven feasible -- a 4-leg shared-stream prototype runs bit-exact (2026-07-14)

A standalone prototype (`nest-forge/prototypes/gpu_stream_interop/`, RTX 4050 / nvhpc 26.3 / CUDA 13.3)
demonstrates the full cross-toolchain + shared-stream + device-pointer story end to end. A C++ driver
creates ONE `cudaStream_t`, `cudaMalloc`s device buffers, `cudaMemcpyAsync`es H2D, and calls four
separately-compiled `.a` kernels passing device pointers + that one stream:
- `libcuda_scale.a` (nvcc, `__global__`, `<<<...,stream>>>`): `mid = a*2`.
- `libacc_add1.a` (nvfortran `-acc`, `deviceptr` + `acc_set_cuda_stream` + `async(q)`): `b = mid+1`.
- `libcublas_leg.a` (**cuBLAS vendor lib**, `cublasSetStream` + `cublasDscal`): `b *= 3`.
- `libomp_leg.a` (nvc `-mp=gpu`, `is_device_ptr`): the OMP-target leg -- bridged, see below.
- The three stream-bindable legs on one stream: `maxerr=0`, bit-exact PASS. Each read what the previous
  wrote, so all three serialized on the single shared stream. Negative control (separate streams, no sync):
  **29/30 races** -- ordering is real, not luck.
- OMP-target leg (`omp_probe.cpp`): no-bridge run **races 29/30**; CUDA-event bridge **0/30** (PASS). OMP is
  the one leg that cannot bind the external stream and must be event-bridged by the driver.
- Link: `nvc++ -cuda -acc -gpu=cc89 driver.cpp -lcuda_scale -lacc_add1 -lcublas_leg -cudalib=cublas
  -fortranlibs` (`-fortranlibs` pulls the nvfortran runtime; `-cuda -acc` pull cudart + OpenACC runtime;
  `-cudalib=cublas` pulls the vendor lib). OMP probe links `nvc++ -mp=gpu -cuda`.

Constraints this validates (carry into the arena/env model): **one vendor backend** (CUDA-C++ +
nvfortran-OpenACC + nvhpc-OMP coexist; a HIP `.a` cannot join -- it is the mutually-exclusive alt
backend); **one dynamically-linked libcudart, one primary CUDA context, one device** (DaCe uses the
runtime-API primary context, which nvfortran/OpenACC share transparently -- do NOT static-link cudart or
a second runtime state invalidates the handle); **DaCe owns stream lifetime** (kernels receive the
handle, never create/destroy it).

## Self-check caveat (unlike the CPU targets)

The CPU targets validate cheaply against the numpy oracle in-process. A GPU target's self-check needs a
device + the nvhpc toolchain, so `_emit_dace`-style in-process validation does not apply. Minimum
self-check: emit + compile with `nvcc`/`nvfortran`/`nvc++` (syntactic + device-codegen check) guarded
behind a "gpu toolchain present" probe; full numeric validation happens in the nest-forge GPU arena on a
GPU box (daint / a local RTX). Do not gate CI on GPU presence.

## nest-forge-side follow-on (not hpcagent_bench's job, listed for context)
`libnode.py::proto_and_call` gains the `void* stream` param; a sibling `ExpandExternCallGPU` builds the
tasklet on GPU scope and appends the stream to the extern-C call; `ExternLibEnv` links `cudart` +
`-cudalib=cublas` + the OpenACC/OMP runtime; a `device` strategy in `strategies.py` cuts the host/device
boundary on `ScheduleType.GPU_Device`; `arena.py` adds nvcc/nvfortran/nvc++ (and hipcc) families.

### Where the tasklet GETS the stream to pass (DaCe ABI, old vs new codegen)
The stream-lowering pass wires the stream to each GPU-scheduled node, so the shim references ONE fixed
magic keyword -- **`__dace_current_stream`** -- and it works for BOTH codegens (confirmed against the
new-gpu-codegen branch, 2026-07-14):
- **Old codegen:** the node carries an int attribute `_cuda_stream` (the stream index); the codegen emits
  a local `__dace_current_stream` in the host-tasklet prelude (`dace/codegen/targets/cpp.py:861,865`,
  index picked in `cuda.py:862`). The shim just references `__dace_current_stream`.
- **New codegen:** the node carries an explicit **in-connector literally named `__dace_current_stream`**,
  type `gpuStream_t` (HIP-portable alias, not raw `cudaStream_t`). The stream-lowering pass adds it
  (`dace/transformation/passes/gpu_specialization/stream_lowering_helpers.py:153`
  `add_in_connector(in_conn, gpuStream_t)`); the name is the constant `CURRENT_STREAM_NAME =
  "__dace_current_stream"` (`dace/libraries/standard/helper.py:12`), aliased `STREAM_CONNECTOR`
  (`dace/transformation/passes/gpu_specialization/helpers/gpu_helpers.py:20`). The codegen reads the
  in-connector and binds the same-named local (`cpp.py:887` reads it, `cpp.py:896`
  `__dace_current_stream = <conn>`).
So `ExpandExternCallGPU` does the SAME thing for both: reference `__dace_current_stream` and append it --
`<sym>(..., __dace_current_stream);`. For new codegen the libnode must additionally declare an in-connector
named `__dace_current_stream` so the lowering pass wires it. The shim never creates or owns the stream -- it
receives DaCe's and forwards it verbatim. (Type is `gpuStream_t`, so the same shim compiles under the HIP
backend once that path exists.)
