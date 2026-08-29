# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
# The write-side mirror of the ICON zekinh cell-from-edges interpolation
# (mo_velocity_advection, via dace-fortran tests/velocity_zekinh_block.f90).
# Reimplemented in NumPy as the HPCAgent-Bench correctness reference.
"""ICON zekinh, scattered: a weighted source written through a data-dependent destination.

The read side of zekinh gathers three incident edges per cell; this is its transpose. Each
(block, edge) pair writes into ``dst[edge_blk, jk, edge_idx]`` -- two data-dependent axes
with the affine level axis between them, which is the addressing shape that separates an
unstructured kernel from a strided one.

WHY THIS STAYS A LOOP NEST. The store is an assignment, not an accumulation, and the
connectivity repeats: several cells share an edge, so several iterations target the same
cell. The result is therefore the LAST write, and which write is last is decided by the
iteration order -- an output dependence, not a reduction. ``np.add.at`` is the canonical
spelling for the accumulating scatter (`icon_scatter` uses it) and is the wrong operator
here; a fancy-index assignment would leave the tie-break to numpy's buffering rather than
to the traversal every other backend performs. The nest is the semantics.

Only the level axis is free: distinct ``jk`` write disjoint planes whatever the
connectivity does, which is the one loop a parallel schedule may take.

Layout is row-major ``[NB, NLEV, NPROMA]``, the Fortran ``(JC, JK, JB)`` tuples reversed;
the index tables are 0-based, as every index array in this corpus is.
"""

import numpy as np


def zekin_scatter(e_bln, edge_idx, edge_blk, src, dst, NB, NLEV, NPROMA):
    for jb in range(NB):
        for jk in range(NLEV):
            for jc in range(NPROMA):
                dst[edge_blk[jb, jc], jk, edge_idx[jb, jc]] = e_bln[jb, jc] * src[jb, jk, jc]
