"""
CFM64 Shuffle — O(1) Memory Feistel-Based Shuffling

Core shuffling algorithm using a balanced Feistel network with SplitMix64
as the stateless keyed hash (round function).

All classes delegate to the C++ native extension (``_cfm64_native``) for
maximum performance.  The thin Python wrappers below preserve the public
API while providing ~100× speedup over a pure-Python implementation.

Architecture
------------
Level 1 (Block Order)  — Permutes block IDs for I/O-efficient disk access.
Level 2 (Intra-Block)  — Permutes item offsets within each block (in RAM).

References
----------
- Steele, Lea & Flood (2014). *Fast splittable pseudorandom number generators.*
- Black & Rogaway (2002). *Ciphers with arbitrary finite domains.*
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from cfm64._cfm64_native import (
    BlockShuffle as _NativeBlockShuffle,
)
from cfm64._cfm64_native import (
    FeistelPermutation as _NativeFeistelPermutation,
)
from cfm64._cfm64_native import (  # type: ignore[import-not-found]
    SplitMix64 as _NativeSplitMix64,
)

# 64-bit mask used to clamp Python ints to unsigned 64-bit values.
_U64 = 0xFFFF_FFFF_FFFF_FFFF


# ---------------------------------------------------------------------------
# SplitMix64
# ---------------------------------------------------------------------------

class SplitMix64:
    """SplitMix64 — C++ accelerated.

    CFM64 uses the *stateless* ``hash(key, data)`` variant as the Feistel
    round function.  The stateful ``next()`` method is retained for
    compatibility.

    Parameters
    ----------
    seed : int
        Initial 64-bit state.
    """

    __slots__ = ("_native",)

    def __init__(self, seed: int) -> None:
        self._native = _NativeSplitMix64(seed & _U64)

    def next(self) -> int:
        """Advance state and return a 64-bit pseudo-random value."""
        return int(self._native.next())

    @staticmethod
    def hash(key: int, data: int) -> int:
        """Stateless keyed hash: ``finalize(key ⊕ data + γ)``."""
        return int(_NativeSplitMix64.hash(key & _U64, data & _U64))

    @staticmethod
    def derive_key(base_key: int, index: int) -> int:
        """Derive a child key.  Alias for ``hash(base_key, index)``."""
        return int(_NativeSplitMix64.derive_key(base_key & _U64, index & _U64))


# ---------------------------------------------------------------------------
# Feistel Permutation
# ---------------------------------------------------------------------------

class FeistelPermutation:
    """Balanced Feistel Network with cycle walking — C++ accelerated.

    Provides a bijective (1-to-1) permutation over an arbitrary domain
    ``[0, N)`` using O(1) memory and O(1) per-index computation.

    Parameters
    ----------
    key : int
        64-bit permutation key.
    domain_size : int
        *N* — total number of elements to permute.
    rounds : int
        Number of Feistel rounds (default 4; minimum 4 recommended).
    """

    __slots__ = ("_native", "domain_size", "key", "rounds")

    def __init__(self, key: int, domain_size: int, rounds: int = 4) -> None:
        self._native = _NativeFeistelPermutation(key & _U64, domain_size, rounds)
        self.domain_size = domain_size
        self.key = key & _U64
        self.rounds = rounds

    def permute(self, index: int) -> int:
        """Map *index* → permuted index (cycle walking for out-of-range)."""
        return int(self._native.permute(index))

    def inverse_permute(self, index: int) -> int:
        """Recover the original index from a permuted index."""
        return int(self._native.inverse_permute(index))

    def permute_batch(self, indices: np.ndarray) -> np.ndarray:
        """Vectorized batch permutation."""
        return np.asarray(self._native.permute_batch(indices), dtype=np.int64)


# ---------------------------------------------------------------------------
# CFM64 Two-Level Shuffler
# ---------------------------------------------------------------------------

class CFM64Shuffle:
    """Two-level O(1)-memory shuffler for block-based data pipelines — C++ accelerated.

    Level 1 — *inter-block*: determines **which** blocks to load and in
    what order.  Each block is read sequentially from disk, minimising
    seeks.

    Level 2 — *intra-block*: after a block is loaded into RAM, permutes
    the item offsets so that items are yielded in shuffled order.

    Parameters
    ----------
    dataset_size : int
        Total number of samples in the dataset.
    block_size : int
        Number of samples per storage block.
    seed : int
        Base 64-bit seed.
    rounds : int
        Feistel rounds per permutation (default 4).

    Examples
    --------
    >>> shuffler = CFM64Shuffle(1_000_000, block_size=1024, seed=42)
    >>> for block_pos in range(shuffler.num_blocks):
    ...     block_id = shuffler.get_block_to_load(block_pos)
    ...     data = disk.read_block(block_id)
    ...     for offset in shuffler.get_intra_block_order(block_id):
    ...         yield data[offset]
    """

    __slots__ = (
        "_native", "dataset_size", "block_size", "seed", "base_seed", "rounds",
        "num_blocks", "last_block_size",
    )

    def __init__(
        self,
        dataset_size: int, block_size: int = 1024,
        seed: int = 42, rounds: int = 4,
    ) -> None:
        self.base_seed = seed
        self._init(dataset_size, block_size, seed, rounds)

    def _init(
        self,
        dataset_size: int, block_size: int = 1024,
        seed: int = 42, rounds: int = 4,
    ) -> None:
        self.dataset_size = dataset_size
        self.block_size = block_size
        self.seed = seed
        self.rounds = rounds

        self.num_blocks = (dataset_size + block_size - 1) // block_size
        self.last_block_size = dataset_size % block_size
        if self.last_block_size == 0 and dataset_size > 0:
            self.last_block_size = block_size
        self._native = _NativeBlockShuffle(
            dataset_size, block_size, seed, rounds,
        )

    # -- Level 1: block order -----------------------------------------------

    def get_block_to_load(self, position: int) -> int:
        """Return the block ID to load at *position* in the epoch."""
        return int(self._native.get_block_to_load(position))

    def get_block_order(self) -> np.ndarray:
        """Return block IDs in shuffled loading order."""
        return np.asarray(self._native.get_block_order(), dtype=np.int64)

    def get_block_size(self, block_id: int) -> int:
        """Return the number of items in *block_id*."""
        return int(self._native.get_block_size(block_id))

    def get_block_start(self, block_id: int) -> int:
        """Return the global start index of *block_id*."""
        return int(self._native.get_block_start(block_id))

    # -- Level 2: intra-block shuffle ---------------------------------------

    def get_intra_block_order(self, block_id: int) -> np.ndarray:
        """Return shuffled offsets for items within *block_id* (in RAM)."""
        return np.asarray(
            self._native.get_intra_block_order(block_id), dtype=np.int64,
        )

    def get_item_offset(self, block_id: int, position: int) -> int:
        """Return the shuffled offset for a single position in a block."""
        return int(self._native.get_item_offset(block_id, position))

    # -- full iteration -----------------------------------------------------

    def __iter__(self) -> Iterator[tuple[int, int, int]]:
        """Yield ``(block_id, offset, global_index)`` in I/O order."""
        for pos in range(self.num_blocks):
            block_id = self.get_block_to_load(pos)
            start = self.get_block_start(block_id)
            for offset in self.get_intra_block_order(block_id):
                yield block_id, offset, start + offset

    def get_shuffled_indices(self) -> list[int]:
        """Return all shuffled global indices as a flat list."""
        return [gi for _, _, gi in self]

    def __len__(self) -> int:
        return self.dataset_size

    # -- epoch management ---------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Derive a new shuffle order for *epoch*.  O(1) operation."""
        new_seed = SplitMix64.derive_key(self.base_seed, epoch)
        self._init(self.dataset_size, self.block_size, new_seed, self.rounds)

    def get_state(self) -> tuple[int, int, int]:
        """Return minimal checkpoint state (24 bytes)."""
        return (self.seed, self.dataset_size, self.block_size)

    @classmethod
    def from_state(cls, state: tuple[int, int, int]) -> CFM64Shuffle:
        """Restore a shuffler from checkpoint state."""
        seed, dataset_size, block_size = state
        return cls(dataset_size, block_size, seed)
