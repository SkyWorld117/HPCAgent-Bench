- `submit` -- the terminal grade (public + a hidden seed) and the ONLY recorded one. `score`
  records nothing. Submit the moment a score comes back correct, then keep improving and submit
  again: every verified submission is kept and your best one counts, so an early submit costs
  nothing and a missing one costs the whole kernel.
@@SPLIT@@
4. Iterate on step 3. `submit` (same body) every time a score comes back correct and better.

Score early and often -- after every meaningful change, never sit on an untested rewrite.
You have plenty of attempts (~1000 score calls is fine). Do not stop early. The ceiling
differs per kernel: some allow 10x, some barely 1.2x -- so never settle for your first
working speedup. Keep trying genuinely different approaches; declare a plateau only after
several distinct ideas scored no better. `score` records NOTHING: a kernel you scored but
never submitted earns nothing, however well it scored, so SUBMIT every correct improvement
as you go -- the best verified submission is what counts.
