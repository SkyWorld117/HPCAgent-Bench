- `submit` -- the terminal grade (public + a hidden seed), and you get exactly ONE. It is the only
  recorded result and it cannot be revised, so the version you submit is the version you are
  measured on. `score` is free, unlimited and records nothing: use it to find out whether you are
  right, and submit only once you have stopped learning from it.
@@SPLIT@@
4. Iterate on step 3 with `score`, which is free and unlimited. `submit` LAST, exactly once.

Because the submission is single and final, a parallel version that scores ~1.00x is not
progress you can bank -- it is the answer you will be graded on. Before you submit, be able to
say which axis carries the dependence and which axis is unit stride, and check that the version
in front of you beats the serial baseline rather than merely matching it. If a rewrite scored
no better, the serial version you started from is the better submission.

Score early and often; there is no reason to sit on an untested rewrite when scoring costs
nothing. The ceiling differs per kernel: some allow 10x, some barely 1.2x, and some carry a real
dependence and top out at 1.0x. Keep trying genuinely different approaches while `score` is still
teaching you something, then submit your best one.
