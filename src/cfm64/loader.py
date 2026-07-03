"""
CFM64 Loader — Buffer-free, I/O-efficient data loading with O(1)-memory shuffling.

The 7 MB fio-derived block is the unit of **both** disk transfer and shuffling:

1. **Block order (Level 1):** A Feistel permutation over block IDs randomises
   the order blocks are read from disk each epoch — sequential I/O *within*
   each block, pseudo-random order *across* blocks.

2. **Intra-block shuffle (Level 2):** Each loaded block's items are shuffled in
   RAM by a second Feistel permutation and yielded as batches. No block is ever
   accumulated with another, so peak memory stays proportional to a single
   block (~7 MB), independent of dataset size.

Both Feistel permutations are bijections, so **every item is yielded exactly
once per epoch**. Epoch transitions are O(1): a new seed is derived from the
base seed and epoch number — no data is moved and no permutation array is
stored.

Key-derivation tags (frozen — never change)::

    _BLOCK_TAG        = 0xB10C_0000_0000_0000   # block-order key
    _INTRA_BLOCK_TAG  = 0x1B0C_0000_0000_0000   # intra-block shuffle key
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Any, Iterator

# NOTE: torch is an *optional* dependency (see pyproject `[project.optional-
# dependencies]`). It is imported lazily inside iteration so that
# ``import cfm64`` — and everything except batch collation — works without it.
from cfm64.datasets import BlockDataset
from cfm64.shuffle import FeistelPermutation, SplitMix64

logger = logging.getLogger(__name__)

# Key-derivation tags — frozen, never change.
_BLOCK_TAG: int = 0xB10C_0000_0000_0000
_INTRA_BLOCK_TAG: int = 0x1B0C_0000_0000_0000


class CFM64Loader:
    """Buffer-free, I/O-efficient data loader for PyTorch.

    Blocks are visited in Feistel-permuted order, and each block's items are
    shuffled in RAM before being emitted as batches. Peak memory is proportional
    to a single block, not to the dataset — the shuffle *state* itself is O(1).

    Parameters
    ----------
    dataset : BlockDataset
        A byte-block-aware dataset adapter.
    batch_size : int
        Samples per batch.
    seed : int
        Base random seed.
    rounds : int
        Feistel rounds for both the block-order and intra-block permutations.

    Examples
    --------
    >>> from cfm64 import CFM64Loader, TextBlockDataset
    >>> dataset = TextBlockDataset("train.csv", has_header=True)
    >>> loader = CFM64Loader(dataset, batch_size=64, seed=42)
    >>> for epoch in range(100):
    ...     loader.set_epoch(epoch)   # O(1)
    ...     for batch in loader:
    ...         train(batch)
    """

    def __init__(
        self,
        dataset: BlockDataset,
        batch_size: int = 32,
        seed: int = 42,
        rounds: int = 4,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0 (got {batch_size})")

        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.rounds = rounds
        self._epoch = 0

        if len(dataset) == 0 or dataset.num_blocks == 0:
            logger.warning(
                "CFM64Loader initialized on an empty dataset — "
                "no batches will be produced."
            )

        logger.info(
            f"Initialized CFM64Loader: {len(dataset)} items, "
            f"batch_size={batch_size}, num_blocks={dataset.num_blocks}"
        )

        # Dedicated ThreadPoolExecutors: one reads/prefetches the next block
        # while the current block is being extracted and shuffled.
        self._io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="CFM_IOWorker"
        )
        self._process_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="CFM_ProcessWorker"
        )

    def __del__(self) -> None:
        """Clean up background executors."""
        if hasattr(self, "_io_executor"):
            self._io_executor.shutdown(wait=False)
        if hasattr(self, "_process_executor"):
            self._process_executor.shutdown(wait=False)

    # -- epoch management ---------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Derive a new shuffle order for *epoch*.  O(1) operation."""
        self._epoch = epoch

    # -- iteration ----------------------------------------------------------

    def __len__(self) -> int:
        """Number of batches per epoch."""
        n = len(self.dataset)
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Any]:
        """Iterate all items exactly once, shuffled via the two-level scheme."""
        yield from self._iterate_batches()

    def _iterate_batches(self) -> Iterator[Any]:
        num_blocks = self.dataset.num_blocks
        if num_blocks == 0:
            return

        # torch is only needed to collate batches — import it lazily so the
        # rest of the package does not require torch to be installed.
        from torch.utils.data import default_collate

        # Derive epoch-specific keys.
        epoch_key = SplitMix64.derive_key(self.seed, self._epoch)
        block_key = SplitMix64.derive_key(epoch_key, _BLOCK_TAG)
        intra_key_base = SplitMix64.derive_key(epoch_key, _INTRA_BLOCK_TAG)

        # Level 1: Feistel permutation over block IDs (disk visit order).
        block_perm = FeistelPermutation(block_key, num_blocks, self.rounds)

        def fetch_block(position: int) -> Any:
            block_id = block_perm.permute(position)
            return self.dataset.load_block(block_id)

        def process_block(
            position: int, fut: concurrent.futures.Future
        ) -> list[Any]:
            try:
                block_data = fut.result()
                return [
                    self.dataset.get_item(block_data, j)
                    for j in range(len(block_data))
                ]
            except Exception as exc:  # attach block context, then re-raise
                block_id = block_perm.permute(position)
                raise RuntimeError(
                    f"CFM64Loader failed while loading/processing block "
                    f"{block_id} (visit position {position})"
                ) from exc

        current_batch: list[Any] = []

        # Prefetch pipeline: read block 0, then read block i+1 while the
        # current block i is being shuffled and emitted.
        io_fut = self._io_executor.submit(fetch_block, 0)
        proc_fut = self._process_executor.submit(process_block, 0, io_fut)

        for i in range(num_blocks):
            items = proc_fut.result()

            # Kick off the next block before shuffling/emitting the current one.
            if i + 1 < num_blocks:
                next_io_fut = self._io_executor.submit(fetch_block, i + 1)
                proc_fut = self._process_executor.submit(
                    process_block, i + 1, next_io_fut
                )

            n_items = len(items)
            if n_items == 0:
                continue

            # Level 2: shuffle items within this block.
            intra_key = SplitMix64.derive_key(intra_key_base, i)
            item_perm = FeistelPermutation(intra_key, n_items, self.rounds)

            for j in range(n_items):
                current_batch.append(items[item_perm.permute(j)])
                if len(current_batch) == self.batch_size:
                    yield default_collate(current_batch)
                    current_batch = []

        if current_batch:
            yield default_collate(current_batch)

    # -- checkpointing ------------------------------------------------------

    def get_state(self) -> dict:
        """Return minimal checkpoint dict."""
        return {
            "seed": self.seed,
            "epoch": self._epoch,
            "dataset_size": len(self.dataset),
            "batch_size": self.batch_size,
        }

    @classmethod
    def from_checkpoint(
        cls, checkpoint: dict, dataset: BlockDataset, **kwargs: Any
    ) -> CFM64Loader:
        """Restore a loader from a checkpoint dict.

        Raises
        ------
        ValueError
            If the checkpoint's ``dataset_size`` does not match *dataset*; the
            shuffle order is only reproducible against the original dataset.
        """
        ckpt_size = checkpoint.get("dataset_size")
        if ckpt_size is not None and ckpt_size != len(dataset):
            raise ValueError(
                f"Checkpoint dataset_size ({ckpt_size}) does not match the "
                f"provided dataset ({len(dataset)}); the shuffle order would "
                f"not be reproducible. Restore against the original dataset."
            )
        loader = cls(
            dataset=dataset,
            batch_size=checkpoint.get("batch_size", 32),
            seed=checkpoint["seed"],
            **kwargs,
        )
        loader.set_epoch(checkpoint["epoch"])
        return loader

    # -- diagnostics --------------------------------------------------------

    def get_io_stats(self) -> dict:
        """Return I/O efficiency statistics."""
        n = len(self.dataset)
        nb = self.dataset.num_blocks
        return {
            "dataset_size": n,
            "dataset_memory_mb": self.dataset.dataset_memory_size / (1024**2),
            "num_blocks": nb,
            "block_size_mb": self.dataset.block_size_bytes / (1024**2),
            "seeks_per_epoch": nb,
            "seeks_random_shuffle": n,
            "seek_reduction": n / nb if nb > 0 else 1,
        }
