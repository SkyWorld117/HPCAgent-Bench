---
name: pytorch-to-numpy
description: Port a KernelBench PyTorch model into an HPCAgent-Bench numpy kernel -- what to strip, which torch defaults are numerics, and how to prove the port against torch.
---

Turn one PyTorch `Model` into a numpy kernel the repo can lower to C, C++, Fortran and DaCe from one
source.

**The kernel this produces has to satisfy the `python-to-numpy` skill, which is the contract**: the
ndarray-and-scalar rule, sizes taken from the manifest's symbols rather than re-derived from a
buffer, the buffer-out signature, the manifest, the landmines and the verification ladder all live
there and are not restated here. Read that page first. What follows is only what is specific to
coming FROM torch -- the translation table, the defaults that are numerics, and the parity check.

## What to strip, and what is numerics

Strip what exists only for training or a device: `requires_grad`, `.detach()`, `.cpu()`, `.cuda()`,
`.to()`, `.item()`, optimizer state, dropout (eval-mode dropout is the identity). Keep anything that
changes inference numerics.

| PyTorch | numpy | watch for |
|---|---|---|
| `.view` / `.reshape` | `np.reshape` into a FRESH buffer | rank change onto a live name is a CNF violation |
| `.permute` | `np.transpose` into a fresh buffer | changes strides, so a later reshape copies |
| `dim=` | `axis=` | `keepdim` is `keepdims` |
| `F.relu` | `np.maximum(x, 0)` | |
| `nn.Linear` | `x @ W.T + b` | torch stores `weight` as (out, in) |
| `nn.Conv2d/3d` | the tap loop above | weight is (out_c, in_c/groups, k...), NCHW throughout |
| `nn.BatchNorm2d` | eval mode uses `running_mean`/`running_var`, NOT batch stats | eps `1e-5`, stats reshaped to (1, C, 1, 1) |
| `nn.LayerNorm` | mean/var over the LAST dims | eps `1e-5`, different axes than BatchNorm |
| `MaxPool/AvgPool` | tap loop | `ceil_mode`, and `count_include_pad` for avg |
| `nn.Softmax(dim=d)` | subtract the max along `d` first | omitting the shift overflows in fp32 |
| `padding='same'` | explicit pad | torch splits odd padding asymmetrically |

Defaults are numerics. An eps or a padding convention taken from memory rather than from the torch
docs is the commonest way a port comes out plausible and wrong. **BatchNorm in eval mode is the
trap** -- the training-mode formula looks fine on random data and is not the operator.

A fresh port is checked against torch, not against a baseline: import the original dynamically, call
`get_init_inputs()`/`get_inputs()` if present, instantiate the `Model`, `.eval()`, and seed the numpy
arrays FROM its parameters. Start at `rtol=1e-4, atol=1e-5` and tighten. Tests may import torch; the
kernel file may not.

Level 3 models are whole networks built from level 1 primitives, so one correct convolution and one
correct normalisation carry most of a ResNet. The recurrent and attention models carry traps a
convolution does not -- gate ordering in a packed LSTM/GRU weight, hidden-state init shape,
`batch_first`, and where attention masking needs `-inf` rather than a large negative -- and each
repeats across every remaining model, so settle it against torch the first time.


## Parity against torch is not optional

A port you have not run against PyTorch is not a port.

- Import the original dynamically; call `get_init_inputs()` / `get_inputs()` if present.
- Instantiate the torch `Model`, call `.eval()`, and seed your numpy arrays from ITS parameters --
  do not initialise the two independently.
- Compare forward outputs. Start at `rtol=1e-4, atol=1e-5` for fp32-heavy kernels and tighten once
  it is stable.
- Shrink oversized dims so it runs on CPU, but keep the structure representative -- a 1x1 conv
  proves nothing about a 3x3 with padding.
- Classify a failure before fixing it: unsupported construct, shape/init mistake, tolerance, or
  harness. They have different fixes and guessing wastes the run.

**Do not weaken a check, a tolerance, or the guide to make something pass.** If a PyTorch feature
does not fit the surface above, stop and say which rule is missing rather than bending the port
around it.

## Level 3 specifically

Level 3 kernels are whole networks composed of level 1 primitives, so the primitives dominate the
work -- get one convolution and one normalisation exactly right and most of a ResNet follows.

The recurrent and attention models carry traps a convolution does not, and each one will repeat
itself across every remaining model unless you settle it against torch the first time:
- **gate ordering** in a packed LSTM/GRU weight matrix,
- **hidden state initialisation** (zeros, and the shape convention for layers/directions),
- **sequence-major vs batch-major** (`batch_first`),
- **masking** semantics in attention, and where `-inf` versus a large negative constant matters.

## Documentation

- `torch.nn` reference -- the defaults (eps, padding, weight layout) that decide numerics -- https://docs.pytorch.org/docs/stable/nn.html
- NumPy reference, for the operation you are replacing it with -- https://numpy.org/doc/stable/reference/
- KernelBench, the upstream this corpus ports from -- https://github.com/ScalingIntelligence/KernelBench
