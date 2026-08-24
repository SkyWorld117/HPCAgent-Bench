import numpy as np


def kernel(N, seq, table, complement_sum=3, pair_bonus=1):
    """Wavefront DP over diagonals.

    Diagonal ``d`` reads only diagonals below it, so ``d`` stays a loop, but its cells are
    independent and update as one max-chain. The inner k-sum is a variable-width max-plus
    reduction with no closed form, so it stays a loop whose every step is a full array op.
    """
    for d in range(1, N):
        i = np.arange(N - d)
        j = i + d
        cand = table[i, j]
        cand = np.maximum(cand, table[i, j - 1])
        cand = np.maximum(cand, table[i + 1, j])
        if d > 1:
            bonus = np.where(seq[i] + seq[j] == complement_sum, pair_bonus, 0).astype(table.dtype)
            cand = np.maximum(cand, table[i + 1, j - 1] + bonus)
        else:
            cand = np.maximum(cand, table[i + 1, j - 1])
        for m in range(1, d):
            cand = np.maximum(cand, table[i, i + m] + table[i + m + 1, j])
        table[i, j] = cand
