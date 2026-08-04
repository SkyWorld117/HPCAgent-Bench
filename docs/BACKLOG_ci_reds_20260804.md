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
| `agent-bench container image (build + launch)` | `springer13/hptt` clone now 403s | no -- upstream vanished |

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

**RESOLVED** -- specialised, not folded. `dim` reaches the ABI, so the manifest's value is not the
value the harness passes; the emitted kernel carries one loop nest per axis and picks between them
at run time (`frontend._specialize_runtime_axis`). Scope is the whole body: `dim` also drives the
narrow, the take, the expand_dims and the concatenate, and the temporaries between them have a
different shape per axis. A negative axis shares its branch with `axis + rank`, numpy-style; an
out-of-range one matches no branch and the kernel writes nothing (numpy raises there, and a void
kernel cannot). `KNOWN_NON_LOWERING` stays empty.

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

## 4. The container image cannot fetch HPTT any more

`hpcagent_bench.Dockerfile:99`, step `RUN sh /build-hptt.sh`:

```
Cloning into '/tmp/tmp.Hr24Cw47wC'...
fatal: unable to access 'https://github.com/springer13/hptt.git/': The requested URL returned error: 403
```

A 403 on an anonymous clone means the upstream repository is gone or has been made private -- this is
not rate limiting (that reports 429) and not our credentials (the step never had any). Nothing in
this repo changed; the build simply depends on a URL that stopped answering, so **every** image build
fails from now on until the dependency is re-sourced.

Note what it takes down with it: the failing step is early in the image, so the CPU image never gets
built and `tests/test_container_launch.py` never runs. One dead URL red-lines the whole container
track.

Options, in the order they should be considered:

- **Vendor it.** HPTT is a small, stable, header-plus-sources tensor-transpose library; pinning a
  known-good tarball in-repo (or in a release asset) removes the network dependency entirely and is
  the only option that cannot break again the same way.
- **Point at a surviving fork.** Cheapest to write, but re-acquires the same failure mode against a
  different owner, and forks drift.
- **Make the step optional.** Only correct if nothing in the corpus needs HPTT; that has to be
  checked rather than assumed, because a silently-absent library turns a build failure into a
  runtime one.

Whichever is chosen, the build script should fail with a message that names the dependency and says
it could not be fetched -- right now the diagnosis costs a trip into the buildkit log.

---

## Cross-cutting

Three of the four reds are non-failures that a reader cannot distinguish from real ones without
opening the logs -- a timeout, a preemption and a dead upstream URL. Worth a follow-up: have the run
summary distinguish **failed** from **timed out**, **cancelled** and **could not fetch a
dependency**, so the count on the run page means what it appears to mean.
