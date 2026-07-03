"""
CFM64 Datasets — Block-aware dataset adapters for raw data.

Blocks are defined by **byte size** (7 MB), not item count.  Each block
contains a variable number of items whose combined on-disk size is at
most ``BLOCK_SIZE_BYTES``.

Supported modalities:
- **TextBlockDataset** — Raw text/CSV files via O(1) byte-offset seeking.
- **ImageBlockDataset** — Image folder structure via cumulative file sizes.
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


# ---------------------------------------------------------------------------
# Abstract Base
# ---------------------------------------------------------------------------

class BlockDataset(ABC):
    """Abstract base class for byte-block-aware datasets.

    Blocks are sized by **bytes** (``BLOCK_SIZE_BYTES``), so each block
    may contain a different number of items.  Subclasses must implement
    ``__len__``, ``dataset_memory_size``, ``num_blocks``, ``load_block``,
    and ``get_item``.
    """

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
    """

    def __init__(
        self,
        path: str,
        transform: Callable | None = None,
        has_header: bool = False,
    ) -> None:
        self.path = path
        self.transform = transform

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
            target_byte = self.offsets[start] + BLOCK_SIZE_BYTES
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


# ---------------------------------------------------------------------------
# Raw Images / Directory
# ---------------------------------------------------------------------------

class ImageBlockDataset(BlockDataset):
    """Image-folder dataset — blocks sized by cumulative file bytes.

    Scans *root* for image files and groups them into blocks whose total
    on-disk size is at most ``BLOCK_SIZE_BYTES`` (7 MB).

    Parameters
    ----------
    root : str
        Root directory to scan for images.
    extensions : list[str], optional
        File extensions to include (default: jpg, jpeg, png, webp).
    transform : callable, optional
        Applied to each PIL image in ``get_item``.
    label_fn : callable, optional
        Maps file path → integer label.
    """

    def __init__(
        self,
        root: str,
        extensions: list[str] | None = None,
        transform: Callable | None = None,
        label_fn: Callable[[str], int] | None = None,
    ) -> None:
        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png", ".webp"]
        self.root = root
        self.extensions = [e.lower() for e in extensions]
        self.transform = transform
        self.label_fn = label_fn

        self.files: list[str] = []
        self.labels: list[int] = []
        for dirpath, _, filenames in os.walk(root):
            for fname in sorted(filenames):
                if any(fname.lower().endswith(ext) for ext in self.extensions):
                    filepath = os.path.join(dirpath, fname)
                    self.files.append(filepath)
                    if label_fn is not None:
                        self.labels.append(label_fn(filepath))

        self._dataset_size: int = len(self.files)
        self._file_sizes = [os.path.getsize(f) for f in self.files]
        self._memory_size = sum(self._file_sizes)

        # Group files into blocks of ≤ BLOCK_SIZE_BYTES
        self._block_ranges = self._build_block_ranges()
        self._num_blocks = len(self._block_ranges)

        logger.info(
            f"ImageBlockDataset: {self._dataset_size} images, "
            f"{self._memory_size / (1024**2):.1f} MB, "
            f"{self._num_blocks} blocks"
        )

    def _build_block_ranges(self) -> list[tuple[int, int]]:
        """Group files by cumulative size into blocks ≤ BLOCK_SIZE_BYTES."""
        if not self.files:
            return []

        ranges: list[tuple[int, int]] = []
        start = 0
        current_bytes = 0

        for i, size in enumerate(self._file_sizes):
            current_bytes += size
            if current_bytes >= BLOCK_SIZE_BYTES:
                ranges.append((start, i + 1))
                start = i + 1
                current_bytes = 0

        if start < len(self.files):
            ranges.append((start, len(self.files)))

        return ranges

    # -- BlockDataset interface ---------------------------------------------

    def __len__(self) -> int:
        return self._dataset_size

    @property
    def dataset_memory_size(self) -> int:
        return self._memory_size

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    def load_block(self, block_id: int) -> list[tuple]:
        from PIL import Image

        start, end = self._block_ranges[block_id]
        return [
            (Image.open(f).convert("RGB"), self.labels[i] if self.labels else None)
            for i, f in enumerate(self.files[start:end], start=start)
        ]

    def get_item(self, block_data: list[tuple], offset: int) -> Any:
        img, lbl = block_data[offset]
        if self.transform:
            img = self.transform(img)
        return img, lbl
