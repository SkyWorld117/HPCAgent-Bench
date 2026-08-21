## Skills in this task

The Task section carries three skill pages: `optimization-hints`, the `lang-*` page for the
target language, and its `openmp-*` page. They are distilled from past runs against THIS judge
-- the graded build lines, the recorded crash causes, the exact directive spellings. Treat them
as the manual for this benchmark, not optional background.

- Before your FIRST rewrite: skim all three pages.
- Before parallelizing ANY loop: re-read `optimization-hints` (the order of attack) and the
  `openmp-*` page -- it decides which loops are safe to thread and spells the directive.
- While WRITING code: follow the `lang-*` page -- signature, headers, dialect and its listed
  expensive mistakes are graded exactly as written there.
- On every score with `correct: false`: find the matching failure pattern in the pages BEFORE
  editing; wrong answers here are usually a listed pattern applied unsafely.
