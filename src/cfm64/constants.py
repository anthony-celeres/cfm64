"""
CFM64 Constants — System-wide configuration constants.

BLOCK_SIZE_BYTES is derived from fio benchmarks showing 7 MB as the optimal
sequential read size for modern storage devices (NVMe/SSD). It is the unit of
**both** disk transfer and shuffling: each read fetches exactly this many
bytes, and the items within it are shuffled in RAM (Level-2 intra-block
permutation) before being emitted as batches.
"""

from __future__ import annotations

# Optimal I/O block size (bytes) — benchmarked via fio.
# Each disk read fetches exactly this many bytes, and is also the granularity
# at which items are shuffled.
BLOCK_SIZE_BYTES: int = 7 * 1024 * 1024  # 7 MB
