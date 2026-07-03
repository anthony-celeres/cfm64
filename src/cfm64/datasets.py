"""
CFM64 Datasets — Block-aware dataset adapters for raw data.

Blocks are defined by **byte size**, not item count.  Each block contains a
variable number of items whose combined on-disk size is at most
``block_size_bytes`` (defaults to ``BLOCK_SIZE_BYTES``; pass the value
``auto-fio`` reports for your storage device to tune blocks to the hardware).

Supported modalities:
- **TextBlockDataset** — Raw text/CSV files via O(1) byte-offset seeking.

Other modalities (images, etc.) are reserved for a future version.
"""

from __future__ import annotations

import bisect
import logging
import os
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable

from cfm64.constants import BLOCK_SIZE_BYTES

logger = logging.getLogger(__name__)


def _validate_block_size(block_size_bytes: int) -> int:
    """Validate a block-size argument and return it.

    The block size is the byte budget each block is packed up to — the unit of
    both transfer and shuffle. It should be the value ``auto-fio`` reports for
    the target storage device (run once per machine, then pass it in).
    """
    if not isinstance(block_size_bytes, int) or isinstance(block_size_bytes, bool):
        raise TypeError(
            f"block_size_bytes must be an int, got {type(block_size_bytes).__name__}"
        )
    if block_size_bytes <= 0:
        raise ValueError(f"block_size_bytes must be positive, got {block_size_bytes}")
    return block_size_bytes


# ---------------------------------------------------------------------------
# Abstract Base
# ---------------------------------------------------------------------------

class BlockDataset(ABC):
    """Abstract base class for byte-block-aware datasets.

    Blocks are sized by **bytes** (``block_size_bytes``), so each block
    may contain a different number of items.  Subclasses must implement
    ``__len__``, ``dataset_memory_size``, ``num_blocks``, ``load_block``,
    and ``get_item``, and set ``self._block_size_bytes``.
    """

    _block_size_bytes: int = BLOCK_SIZE_BYTES

    @property
    def block_size_bytes(self) -> int:
        """Byte budget each block is packed up to (transfer + shuffle unit)."""
        return self._block_size_bytes

    @abstractmethod
    def __len__(self) -> int:
        """Total number of samples in the dataset."""

    @property
    @abstractmethod
    def dataset_memory_size(self) -> int:
        """Total dataset size in bytes (on-disk)."""

    @property
    @abstractmethod
    def num_blocks(self) -> int:
        """Number of byte-sized blocks (each ≤ ``BLOCK_SIZE_BYTES``)."""

    @abstractmethod
    def load_block(self, block_id: int) -> Any:
        """Load all raw data for *block_id* from storage."""

    @abstractmethod
    def get_item(self, block_data: Any, offset: int) -> Any:
        """Extract a single item from loaded block data."""

    def load_block_items(self, block_id: int) -> list[Any]:
        """Load a block and return all items ready for consumption.

        Combines ``load_block`` and ``get_item`` into a single call,
        which is what the :class:`~cfm64.loader.CFM64Loader` uses to
        populate the shuffle buffer.
        """
        block_data = self.load_block(block_id)
        return [self.get_item(block_data, i) for i in range(len(block_data))]


# ---------------------------------------------------------------------------
# Raw Text / CSV
# ---------------------------------------------------------------------------

class TextBlockDataset(BlockDataset):
    """Raw text/CSV dataset — byte-block seeking with line-aligned boundaries.

    Builds (or loads from cache) a byte-offset index for every line, then
    groups consecutive lines into blocks of ``BLOCK_SIZE_BYTES`` bytes.
    Each block is guaranteed to start and end on a line boundary.

    Parameters
    ----------
    path : str
        Path to the text/CSV file.
    transform : callable, optional
        Applied to each raw line (``bytes``) when ``get_item`` is called.
    has_header : bool
        If ``True``, the first line is skipped.
    block_size_bytes : int, optional
        Byte budget per block (transfer + shuffle unit). Defaults to the
        package constant ``BLOCK_SIZE_BYTES``. Pass the value ``auto-fio``
        reports for your storage device to tune blocks to the hardware.
    """

    def __init__(
        self,
        path: str,
        transform: Callable | None = None,
        has_header: bool = False,
        block_size_bytes: int = BLOCK_SIZE_BYTES,
    ) -> None:
        self.path = path
        self.transform = transform
        self._block_size_bytes = _validate_block_size(block_size_bytes)

        self.offsets: list[int] = []
        idx_path = path + ".idx.txt"

        # Build or load byte offsets
        needs_rebuild = True
        if os.path.exists(idx_path):
            if os.path.getmtime(idx_path) >= os.path.getmtime(path):
                needs_rebuild = False
            else:
                logger.info(f"Cache {idx_path} is stale. Rebuilding...")

        if needs_rebuild:
            logger.info(f"Building byte offsets for {path}...")
            with open(path, "rb") as f:
                offset = 0
                for line in f:
                    self.offsets.append(offset)
                    offset += len(line)
            with open(idx_path, "w") as f:
                for off in self.offsets:
                    f.write(f"{off}\n")
        else:
            with open(idx_path) as f:
                self.offsets = [int(line.strip()) for line in f]

        if has_header and len(self.offsets) > 0:
            self.offsets = self.offsets[1:]

        self._dataset_size = len(self.offsets)
        self._file_size = os.path.getsize(path)

        # Memory size = bytes spanned by data lines (from first data line to EOF)
        if self._dataset_size > 0:
            self._memory_size = self._file_size - self.offsets[0]
        else:
            self._memory_size = 0

        # Group lines into ~BLOCK_SIZE_BYTES blocks (line-aligned)
        self._block_ranges = self._build_block_ranges()
        self._num_blocks = len(self._block_ranges)

        # Persistent file handle per worker thread
        self.thread_local = threading.local()

        logger.info(
            f"TextBlockDataset: {self._dataset_size} lines, "
            f"{self._memory_size / (1024**2):.1f} MB, "
            f"{self._num_blocks} blocks"
        )

    def _build_block_ranges(self) -> list[tuple[int, int]]:
        """Group consecutive lines into blocks of ≤ BLOCK_SIZE_BYTES.

        Returns a list of ``(start_line_idx, end_line_idx)`` tuples
        where *end_line_idx* is exclusive.
        """
        if not self.offsets:
            return []

        ranges: list[tuple[int, int]] = []
        start = 0

        while start < len(self.offsets):
            target_byte = self.offsets[start] + self._block_size_bytes
            end = bisect.bisect_left(self.offsets, target_byte, start)
            if end <= start:
                end = start + 1  # at least one line per block
            ranges.append((start, end))
            start = end

        return ranges

    # -- file handle --------------------------------------------------------

    def __del__(self) -> None:
        """Cleanly close the file handle when the dataset is destroyed."""
        if hasattr(self, "thread_local") and hasattr(self.thread_local, "handle"):
            try:
                self.thread_local.handle.close()
            except Exception:
                pass

    def _get_file_handle(self):
        if not hasattr(self, "thread_local"):
            self.thread_local = threading.local()
        if not hasattr(self.thread_local, "handle"):
            self.thread_local.handle = open(self.path, "rb")
        return self.thread_local.handle

    # -- BlockDataset interface ---------------------------------------------

    def __len__(self) -> int:
        return self._dataset_size

    @property
    def dataset_memory_size(self) -> int:
        return self._memory_size

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    def load_block(self, block_id: int) -> list[bytes]:
        """Load all lines belonging to *block_id*."""
        start_line, end_line = self._block_ranges[block_id]

        start_offset = self.offsets[start_line]
        if end_line < len(self.offsets):
            end_offset = self.offsets[end_line]
            bytes_to_read = end_offset - start_offset
        else:
            bytes_to_read = -1  # read to EOF

        f = self._get_file_handle()
        f.seek(start_offset)
        chunk = f.read(bytes_to_read) if bytes_to_read > 0 else f.read()

        return chunk.splitlines(keepends=True)

    def get_item(self, block_data: list[bytes], offset: int) -> Any:
        line = block_data[offset]
        if self.transform is not None:
            line = self.transform(line)
        return line

