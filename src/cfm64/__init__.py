"""
CFM64 — Celeres-Feistel Mix 64

Constant-memory shuffling and I/O-efficient data loading for deep learning.

CFM64 replaces traditional O(N) Fisher-Yates shuffling with an O(1) Feistel
permutation, enabling block-sequential disk access patterns while preserving
statistical shuffle quality. The result: 24 bytes of shuffle state instead of
gigabytes, fewer disk seeks, and instant epoch transitions.

Architecture
------------
Buffer-free and two-level. Fixed 7 MB I/O blocks (benchmarked via fio) are the
unit of both transfer and shuffling: a Level-1 Feistel permutation randomises
the block visit order on disk, and a Level-2 Feistel permutation shuffles the
items within each loaded block before they are emitted as batches. Peak memory
stays proportional to a single block, independent of dataset size.

Quick Start
-----------
    >>> from cfm64 import CFM64Loader, TextBlockDataset
    >>> dataset = TextBlockDataset("train.csv", has_header=True)
    >>> loader = CFM64Loader(dataset, batch_size=64, seed=42)
    >>> for epoch in range(100):
    ...     loader.set_epoch(epoch)
    ...     for batch in loader:
    ...         train(batch)
"""

__version__ = "0.2.0"

# Constants ----------------------------------------------------------------
from cfm64.constants import BLOCK_SIZE_BYTES

# Dataset adapters ---------------------------------------------------------
from cfm64.datasets import (
    BlockDataset,
    TextBlockDataset,
)

# I/O pipeline -------------------------------------------------------------
from cfm64.loader import CFM64Loader

# Core algorithm -----------------------------------------------------------
from cfm64.shuffle import CFM64Shuffle, FeistelPermutation, SplitMix64

__all__ = [
    # Constants
    "BLOCK_SIZE_BYTES",
    # Core
    "SplitMix64",
    "FeistelPermutation",
    "CFM64Shuffle",
    # Datasets
    "BlockDataset",
    "TextBlockDataset",
    # I/O
    "CFM64Loader",
]
