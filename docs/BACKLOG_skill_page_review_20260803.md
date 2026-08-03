# Skill page review, 2026-08-03 -- ALL FIVE PAGES: DO NOT SHIP

Five instrument pages were reviewed against upstream documentation by one agent each, and every
finding was then adversarially re-checked by a second agent that had to reproduce the contradiction
itself. 11 agents. Findings below are the ones that SURVIVED adjudication.

**Nothing here ships until its section is fixed.** The pages are in `docs/skills_draft/`, not on
`load_skills`' search path, so nothing is in a prompt today.

## The pattern worth learning from

Three of these pages (`papi-gpu-amd`, `rocprofv3`, `rocprof-compute`) were written from web research
against hardware that does not exist on this box. Every one carries an honest "nothing here was
executed" fence, and the fence did NOT save them: a reader cannot tell a fenced-but-correct claim
from a fenced-and-fabricated one, and three of the four derived formulas on `papi-gpu-amd` are
fabricated. **An honest fence is not a substitute for a source.** Where hardware is unavailable, the
rule has to be: quote upstream verbatim with a URL, or do not write the line.

Two of the failures are self-inflicted in a way that is worth naming:

- `rocprof-compute` carries the KILOBYTES rule for `FetchSize`/`WriteSize`, calls it "the unit trap
  that turns a correct ratio into a 1000x wrong one", and then sends a rocprof-compute reader to
  apply it -- but that tool prints `Read BW` in **Bytes**. The page CREATES the error it warns about.
- `papi-gpu-amd`'s empty-bracket self-test is `if (probe > 4096)`, so a probe of **0 passes** -- and
  0 is precisely the silent-zero failure the whole page exists to prevent. This is the same
  one-directional guard the `papi-cpu` page was criticised for, reintroduced by the person who
  wrote the criticism.

## papi-gpu-amd -- DO NOT SHIP

Blocking: three of four derived-metric formulas exist in no upstream source on either component
path, and the counter names in them resolve, so wrong numbers come back silently.

1. **L282-286 fabricated formulas.** Page uses `SQ_BUSY_CU_CYCLES` as the denominator for
   `VALUBusy`, `SALUBusy`, `LDSBankConflict`. Upstream `counter_defs.yaml:9675`:
   `100*reduce(SQ_ACTIVE_INST_VALU,sum)/CU_NUM/reduce(GRBM_GUI_ACTIVE,max)`. Legacy `metrics.xml`
   disagrees differently (`*4/SIMD_NUM/GRBM_GUI_ACTIVE`). **Fix:** delete the rows; let `rocprofv3`
   compute the derived metric by name.
2. **L168/L175 sums percentages.** `gpu_total += v;` runs over a loop containing `GPUBusy`,
   `L2CacheHit`, `VALUBusy`, `VALUUtilization`, `MemUnitStalled` -- all "percentage of..." upstream.
   **Fix:** accumulate only `SQ_WAVES`, `FetchSize`, `WriteSize`; report percentages per region.
3. **"PAPI_stop forces the flush" is unsupported on this component.** Upstream README, quoted two
   lines earlier on the same page, says to add a delay before `PAPI_read`/`PAPI_stop`; `rocp_sdk_stop`
   performs no read. **Fix:** attribute the bracket to `rocp_sdk_start` re-opening the vendor context
   and `init_ctx` re-zeroing `ctx->counters`. NOTE: the NVIDIA measurement still stands on its own
   hardware -- what does not port is the MECHANISM claim.
4. **`rocm:::` prefix under a `rocp_sdk` default.** `rocp_sdk.c:88 .name = "rocp_sdk"`; its test
   runner uses `rocp_sdk:::SQ_CYCLES:device=0`. Every shell example on the page is wrong. **Fix:**
   global prefix swap + document the `DIMENSION_*=` qualifiers the page omits entirely.
5. **`GPUBusy` is not a duration.** Upstream: "The percentage of time GPU was busy",
   `100*reduce(GRBM_GUI_ACTIVE,max)/reduce(GRBM_COUNT,max)`. Dividing by it INVERTS the
   normalisation. **Fix:** use `GRBM_GUI_ACTIVE`.
6. **`AQLPROFILE_READ_API=0` is not unconditional.** Upstream: "0 for intercept mode and 1 (or
   unset) for sampling mode"; intercept is opt-in via `ROCP_HSA_INTERCEPT`, and the variable has zero
   hits in `rocp_sdk`. **Fix:** delete the unconditional export.
7. **The long-long note is component-specific.** `rocm` keeps "the binary image of a `double`
   intact"; `rocp_sdk` truncates. **Fix:** pick one component, state its mechanism, and under
   `rocp_sdk` never bit-reinterpret.
8. **Empty-bracket self-test passes on 0** (see above). **Fix:** fail on `probe == 0` too.

## rocprofv3 -- DO NOT SHIP

Blocking: the central counter-collection instruction silently discards half the requested counters.

1. **`--pmc` twice does NOT mean two passes.** `rocprofv3.py:430-438` uses `nargs="*"` with no
   `action="append"`, and `:1566` joins only the survivor. Upstream docs: "For multi-pass execution,
   include multiple `pmc` rows in the input file." **Fix:** replace the paragraph with a two-row
   input file.
2. **Copies DO carry byte volume.** `buffer_tracing.h:287 uint64_t bytes;`, emitted by Perfetto,
   rocpd and JSON (`save.hpp:701-713`). The page says you cannot get it from the trace. **Fix:**
   `--output-format csv json`, read `bytes`.
3. `Group_Segment_Size` -- the column is `LDS_Block_Size`, granule-rounded.
4. "no launch geometry in v1" is wrong: `tool.cpp:437-439` prints `grd, wgr, lds, scr, arch_vgpr,
   sgpr, wave_size`.
5. `pmc_<n>/counter_collection.csv` is pid-prefixed; use a `*_` glob.
6. The reports table omits `*_domain_stats.csv`.

## rocprof-compute -- DO NOT SHIP

Blocking: the step-5 metric table names strings the tool never emits, and the one name that collides
with a real metric inverts its meaning.

1. **`VALUUtilization` inverted.** Tool: "**VALU Utilization** -- what percent of the kernel's
   duration the VALU was busy". Divergence is **`VALU Active Threads`**, unit Work-items. **Fix:**
   retitle the table "rocprofv3 `--pmc`" and add rocprof-compute's own names as a second column.
2. **Bytes, not kilobytes** (see the pattern note above). `gfx942/1700_L2_cache.yaml` gives `Read BW`
   with `unit: (Bytes + $normUnit)`. Importing the KB rule is a 1024x error.
3. **Occupancy arithmetic wrong.** Upstream: "up to 8 wavefronts... 32 total wavefront slots on each
   CU" = **2048** work-items at wave64, and RDNA's "16 slots per SIMD" = **1024**/CU. The page says
   256 and 128. **Fix:** replace both, and delete "every 'use 256 threads' habit from NVIDIA is wrong
   by exactly that factor" -- which asserts a factor of 1 and is self-contradictory.
4. **`MemUnitStalled` paired with the wrong counter.** Upstream:
   `100*reduce(TCP_TCP_TA_DATA_STALL_CYCLES,max)/GRBM_GUI_ACTIVE/SE_NUM`. `SQ_WAIT_INST_ANY` is
   "quad-cycles spent waiting for any instruction to be issued", printed as `Issue Wait Cycles`.
5. **Replay is not the only distortion.** Upstream: "Kernel dispatches are serialized across HIP
   streams on the same GPU during profiling." Add it as an independent second distortion.
6. **Replay breaks MPI.** "This mode fails for MPI applications because running the application
   multiple times results in multiple `MPI_Init` and `MPI_Finalize` calls." Add it, plus
   `--iteration-multiplexing`.

## papi-gpu -- DO NOT SHIP

Blocking: the run loop arms five of the eight events its own later steps require, so steps 3 and 5
cannot be executed as written.

1. **Three events consumed but never armed:** `gpu__dram_throughput.pct_of_peak_sustained_elapsed:stat=avg`,
   `sm__sass_thread_inst_executed:stat=sum`, `smsp__inst_executed:stat=sum` (used at L317, L339).
2. **"within 0.1% on every row" is arithmetically false** -- row 2 of the page's own table is
   **+18.9%** (77.9 KB against 64 KB). **Fix:** "sub-0.05% at MiB scale; the 64 KB row measures
   77.9 KB." This is an error in a table that WAS measured; the measurement is right and the summary
   sentence is wrong.

Everything else on this page survived: the start/stop-vs-read-delta result, the empty-bracket
finding, the `:stat=sum`/`avg` = 3.001 partition count, and the sync redundancy.

## ncu -- DO NOT SHIP

Blocking: the page's gate section is STALE. `RmProfilingAdminOnly: 0` on this box, verified by a
successful 8-pass profile, so the page routes a reader off a working profiler onto `cuobjdump` --
which the page itself calls incapable of costing anything.

1. **Delete the gate section and the `cuobjdump` fallback**; keep one line telling the reader to
   check `/proc/driver/nvidia/params`.
2. **`--cache-control none` needs a precondition.** Upstream: valid only "if only a single kernel
   replay pass is necessary... can lead to inconsistent and out-of-bounds metric values". `--set
   basic` is 8 passes here. **Fix:** source the `none` row from a one-pass `--metrics` run and print
   `Duration` beside it. (The 640x cache-control finding itself stands -- it was measured with
   explicit `--metrics`, i.e. one pass.)
3. L213 prose contradicts the page's own L237 table on `< 0.8`; the TABLE matches NVIDIA.
4. L231 bans block-size reduction, which is NVIDIA's first-named fix.
5. L232 is wrong inside `1 <= Waves Per SM < 5` (tail rule, `speedup_threshold = 20`).

## Repair cost is not uniform

`papi-gpu` and `ncu` are hours of edits to text that is otherwise measured and sound. The three AMD
pages need their **entire metric layer regenerated** against `counter_defs.yaml` and the
`gfx942/*.yaml` panel definitions -- not patched. Do not fix them line by line from this list; it is
a symptom list, and the cause is that the layer was written from search results.
