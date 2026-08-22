## Skills in this task

The Task section carries three skill pages: `optimization-hints`, the `lang-*` page for the
target language, and its `openmp-*` page. They give the build line, the ABI, the directive
spellings and the failure modes this judge grades. Treat them as the manual, not background.

- Before your FIRST rewrite: skim all three pages.
- Before parallelizing ANY loop: re-read `optimization-hints` (the order of attack) and the
  `openmp-*` page -- it decides which loops are safe to thread and spells the directive.
- While WRITING code: follow the `lang-*` page -- signature, headers, dialect and its listed
  expensive mistakes are graded exactly as written there.
- On every score with `correct: false`: find the matching failure pattern in the pages BEFORE
  editing; wrong answers here are usually a listed pattern applied unsafely.

### When a score comes back correct but NOT faster

A parallel rewrite that scores ~1.00x is a RESULT, not a neutral one: the cores were there and
the work did not move. Do not answer it
by adding another directive. Re-derive the loop first, in this order, and say the answer out
loud before editing:

1. **Which axis carries a dependence?** Write the subscript of the value being read against the
   subscript being written. If a read reaches another iteration of the loop you threaded, the
   threads are racing or serializing on it -- that loop was never parallel and the directive was
   an assertion you could not make. Thread a different axis, or fission the statement out.
2. **What is the stride of the INNERMOST loop?** If it walks by a row instead of by one element,
   threading cannot help -- the loop is bandwidth-bound and every lane misses cache. Interchange
   first, then thread the loop outside the unit-stride one.
3. **Is the trip count big enough to pay for a thread team?** A short loop loses to spawn
   overhead. Thread the outer level, or leave it serial and vectorize.

If all three come back clean and it is still 1.00x, the kernel is memory-bound: cut passes over
the data or the size of what you move, and stop adding parallelism.
