# CI reds on main, run 30889625880 -- OPEN

Filed 2026-08-04 against `main` @ `e1a22420` (the 250-kernel KernelBench port push). Three jobs red,
**three unrelated causes**, and only one of them is a test failure. Recording them separately
matters: a run page showing "3 failed" reads as one systemic break, and the fix for each is in a
different place.

| Job | Cause | Real failure? |
|---|---|---|
| `translators (numpyto op suite)` | `KNOWN_NON_LOWERING` stale, `cumsum_exclusive` regressed | **yes** |
| `unit (format + structure + agent-bench + e2e[numba,jax])` | job exceeded the 1h30m ceiling | no -- duration |
| `e2e[c,cpp,fortran] + integration sweep` | runner shutdown signal, exit 143 | no -- infrastructure |

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

Blocked on: deciding whether a symbolic scan axis is lowered by specialising the loop nest per axis
value, or refused with a message that names the kernel. Either is defensible; silently emitting the
wrong nest is not.

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

---

## Cross-cutting

Two of the three reds are non-failures that a reader cannot distinguish from real ones without
opening the logs. Worth a follow-up: have the run summary distinguish **failed** from **timed out**
and **cancelled**, so the count on the run page means what it appears to mean.
