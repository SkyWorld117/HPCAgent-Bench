# CI reds on main, runs 30889625880 and 30903553218 -- OPEN

Filed 2026-08-04 against `main` @ `e1a22420` (the 250-kernel KernelBench port push), plus one more
found on the follow-up run `30903553218` @ `ee460c4b`. Four jobs red, **four unrelated causes**, and
only one of them is a test failure. Recording them separately matters: a run page showing "4 failed"
reads as one systemic break, and the fix for each is in a different place.

| Job | Cause | Real failure? |
|---|---|---|
| `translators (numpyto op suite)` | `KNOWN_NON_LOWERING` stale, `cumsum_exclusive` regressed | **yes** |
| `unit (format + structure + agent-bench + e2e[numba,jax])` | job exceeded the 1h30m ceiling | no -- duration |
| `e2e[c,cpp,fortran] + integration sweep` | runner shutdown signal, exit 143 | no -- infrastructure |
| `agent-bench container image (build + launch)` | `springer13/hptt` clone 403s | no -- anonymous-clone throttle, INTERMITTENT |

---

## 1. `cumsum_exclusive` regressed the ABI corpus-agreement ratchet

`hpcagent_bench/numpy_translators/tests/test_abi_corpus_agreement.py:92`:

```
AssertionError: KNOWN_NON_LOWERING list is stale.
    NEWLY disagreeing (a regression -- the positional call is now wrong): ['ml/cumsum_exclusive/cumsum_exclusive']
    FIXED, delete the entry: []
```

Root cause is one rung down, and it is a real translator gap, not a pin that drifted:

```
NotImplementedError: np.cumsum(__hcall1, axis=dim): axis must be a compile-time integer
    (got 'dim'); the emitted loop nest is chosen by it
```

Two things are tangled here and both need saying. The axis is the **symbol** `dim`, and the operand
`__hcall1` is a **helper-call temporary** -- so the kernel is blocked by the symbolic axis, and the
helper machinery is what put the temp there.

**Fix: implement the lowering, do NOT widen the waiver.** Adding `cumsum_exclusive` to
`KNOWN_NON_LOWERING` would turn a ratchet that caught a real regression into one that documents it.
The same gap shows up independently in the Fortran corpus sweep, so the lowering pays for itself
twice.

**RESOLVED** (2026-08-04, pending merge from `worktree-agent-ad0dd5e950ccb2784`). The decision was
run-time specialisation, not folding: `dim` is a genuine ABI argument, and `_structural_constants`
excludes ABI arguments on purpose after the gmres `min(max_iter, N)` -> `min(100, N)` miscompile.
The rank is known at compile time, so the emitter now writes one nest per axis value and selects at
run time -- `if ((dim == 0) || (dim == -2)) ... else if ((dim == 1) || (dim == -1))`, with no
trailing `else`, because making the last axis the fallthrough is the silent clamp. An out-of-range
axis matches nothing and writes nothing. Verified on ONE compiled artifact called with dim in
{0, 1, -1, -2}: exact agreement with numpy; `KNOWN_NON_LOWERING` stayed `{}`.

## 2. The `unit` job no longer fits in 1h30m

```
The job has exceeded the maximum execution time of 1h30m0s
```

It died in **Phase 6 (integration-marked tests)** having already passed Phases 0, 1, 2, 3 and 5.
Nothing is broken -- the job simply does more work than the ceiling allows, and it will fail again on
the next push without anyone touching the code.

This is the failure mode that wastes the most reviewer time, because the run page says "unit failed"
and the first four phases were green. Options, cheapest first:

- Split Phase 6 into its own job. It is the only phase that builds and runs real artifacts, so it has
  a different cost profile from the rest of the job and no reason to share a budget with it.
- Raise the ceiling. Treats the symptom and the job keeps growing.

Do not "fix" it by dropping integration tests from the sweep.

## 3. `e2e[c,cpp,fortran]` was preempted, not failed

```
The runner has received a shutdown signal. This can happen when the runner service is
stopped, or a manually started runner is canceled.
Process completed with exit code 143.
```

Before that it reported **1104 passed, 48 skipped in 189s** -- Phases 4b, 5 and 5b all green. It was
killed partway into Phase 5c (the KernelBench translation ratchet, `[c] @ S`).

So the kernelbench ratchet did NOT fail; it never finished. This is the same "cancelled reads as
failed" trap already recorded for sibling cells, and the cost is that a green ratchet looks like a
red one.

**Action:** re-run the job before concluding anything about the `[c]` ratchet. If preemption recurs,
the ratchet phase wants to be resumable or split out, so a kill mid-phase does not discard the 1104
tests that already passed.

## 4. The container image cannot fetch HPTT from a CI runner

`hpcagent_bench.Dockerfile:99`, step `RUN sh /build-hptt.sh`:

```
Cloning into '/tmp/tmp.Hr24Cw47wC'...
fatal: unable to access 'https://github.com/springer13/hptt.git/': The requested URL returned error: 403
```

**Correction, same day.** The first read of this was "the upstream repository is gone or private,
because a 403 is not rate limiting". That was wrong, and checking took one command:

```
$ git ls-remote https://github.com/springer13/hptt.git HEAD
942538649b51ff14403a0c73a35d9825eab2d7de        HEAD
```

The repo is public and answering. So the 403 is not about the repository existing -- it is about
*who is asking*: an unauthenticated clone from a GitHub Actions runner, out of a shared egress pool,
which GitHub throttles with 403 rather than 429. That makes this red **intermittent**, not permanent,
which is a materially different thing to plan around: a re-run may pass, and a green run does not
prove it is fixed.

Note what it takes down with it: the failing step is early in the image, so the CPU image never gets
built and `tests/test_container_launch.py` never runs. One throttled clone red-lines the whole
container track.

Options, in the order they should be considered:

- **Pin a commit and fetch it authenticated.** The build script clones `master` unauthenticated.
  Cloning a pinned SHA with the workflow's own `GITHUB_TOKEN` moves the request out of the anonymous
  pool that is being throttled, and pinning removes the "master moved under us" risk at the same
  time. Known-good SHA as of 2026-08-04: `942538649b51ff14403a0c73a35d9825eab2d7de`.
- **Vendor it.** HPTT is a small, stable tensor-transpose library; a known-good tarball in-repo or in
  a release asset removes the network dependency entirely and is the only option that cannot fail
  this way again. Costs a vendored-source update path.
- **Retry with backoff.** Cheapest, and appropriate for a throttle rather than an outage, but it
  raises the floor on build time and hides the signal when the cause really is an outage.

Mirrors, if one is ever wanted: `https://github.com/tongsucn/hptt.git` also resolves (verified same
day) -- but it is a separate lineage, not a mirror, so it drifts and is not a drop-in.

Whichever is chosen, the build script should fail with a message that names the dependency and says
it could not be fetched -- right now the diagnosis costs a trip into the buildkit log.

---

## Cross-cutting

Three of the four reds are non-failures that a reader cannot distinguish from real ones without
opening the logs -- a timeout, a preemption and a throttled clone. Worth a follow-up: have the run
summary distinguish **failed** from **timed out**, **cancelled** and **could not fetch a
dependency**, so the count on the run page means what it appears to mean.
