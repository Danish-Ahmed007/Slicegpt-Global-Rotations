# K-Block Rotation Study

This documents our investigation into K-block hybrid rotation, a middle ground between per-layer PCA (K=1) and global PCA (K=L).

**Result:** K-block rotation fails for intermediate K values. Only the endpoints (K=1 or K=L) work.



## The Idea

SliceGPT computes a rotation matrix Q for each layer, then slices to keep the top D' dimensions. When consecutive layers have different Q matrices, we need shortcut matrices at residual connections to map between bases.

**Memory cost:** For OPT-125M, shortcuts use ~15 MB. This grows with model size.

**K-block hypothesis:** If we group layers into K blocks and share one Q per block:
- Layers within a block need no shortcuts (same Q)
- Only K-1 shortcuts needed instead of L-1
- Memory savings proportional to block size



## What We Tried

For OPT-125M (12 layers, 25% sparsity):

| K | Method | Perplexity | Shortcuts |
|---|--------|------------|-----------|
| 1 | Per-layer | **39.60** | 12 |
| 2 | K-block | 1942 | 6 |
| 4 | K-block | 300 | 3 |
| 6 | K-block | 100 | 2 |
| 12 | Global | **44.31** | 1 |

K=2 gives 49× worse perplexity than K=1. Not a typo.



## Why It Fails

At block boundaries, the shortcut matrix is:

```
S = Q_prev[:, :D']ᵀ @ Q_curr[:, :D']
```

We analyzed S between blocks 0 and 1:
- Condition number: **310**
- Minimum singular value: 0.003

A condition number of 310 means S is nearly singular. Information flowing through that direction is lost.

**Root cause:** The principal components of different blocks are nearly orthogonal. When we slice to keep the top D' components, we're keeping *different* subspaces that don't align.



## Why Global PCA Works

Counter-intuitively, using one Q for the entire model works better than K blocks:

- **Global (K=L):** All layers share the same subspace. No block boundaries = no information loss. The compromise is uniform.

- **K-block (1 < K < L):** Each block finds its own optimal subspace, but these subspaces don't align. Every boundary loses information.



## Conclusion

Only two operating points are practical:

| Choice | When to use |
|--------|-------------|
| **Per-layer (K=1)** | Best quality, if memory permits |
| **Global (K=L)** | Memory-constrained scenarios (~12% worse perplexity) |

Intermediate K values are not viable.



## Implementation

The K-block implementation is in `src/slicegpt/rotate.py`:

```python
rotate_and_slice_kblock(model_adapter, dataloader, slicing_scheduler, k_block=4)
```

It uses a two-phase approach:
1. Collect covariances from the original unmodified model
2. Compute Q per block, apply rotations, add shortcuts at boundaries
