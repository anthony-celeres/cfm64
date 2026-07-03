"""Tests for cfm64.datasets — BlockDataset, TextBlockDataset."""

import os

import pytest

from cfm64.constants import BLOCK_SIZE_BYTES
from cfm64.datasets import BlockDataset, TextBlockDataset

# ---------------------------------------------------------------------------
# BlockDataset ABC
# ---------------------------------------------------------------------------


class _StubBlockDataset(BlockDataset):
    """Minimal concrete implementation for testing the ABC interface."""

    def __init__(self, size: int, memory_bytes: int) -> None:
        self._size = size
        self._memory_bytes = memory_bytes
        self._data = list(range(size))

        # Build block ranges: even split by item count proportional to memory
        items_per_block = max(1, int(size * BLOCK_SIZE_BYTES / max(memory_bytes, 1)))
        self._block_ranges: list[tuple[int, int]] = []
        start = 0
        while start < size:
            end = min(start + items_per_block, size)
            self._block_ranges.append((start, end))
            start = end

    def __len__(self) -> int:
        return self._size

    @property
    def dataset_memory_size(self) -> int:
        return self._memory_bytes

    @property
    def num_blocks(self) -> int:
        return len(self._block_ranges)

    def load_block(self, block_id: int):
        start, end = self._block_ranges[block_id]
        return self._data[start:end]

    def get_item(self, block_data, offset: int):
        return block_data[offset]


class TestBlockDatasetABC:
    """Tests for the abstract base class contract."""

    def test_cannot_instantiate_abc_directly(self):
        with pytest.raises(TypeError):
            BlockDataset()  # type: ignore[abstract]

    def test_stub_basic_properties(self):
        ds = _StubBlockDataset(100, BLOCK_SIZE_BYTES * 2)
        assert len(ds) == 100
        assert ds.dataset_memory_size == BLOCK_SIZE_BYTES * 2
        assert ds.num_blocks >= 1

    def test_load_block_items_convenience(self):
        """load_block_items combines load_block + get_item."""
        ds = _StubBlockDataset(10, BLOCK_SIZE_BYTES)
        items = ds.load_block_items(0)
        assert all(isinstance(x, int) for x in items)

    def test_bijection(self):
        """All items accessed exactly once via load_block + get_item."""
        ds = _StubBlockDataset(25, BLOCK_SIZE_BYTES * 3)
        collected = []
        for bid in range(ds.num_blocks):
            block = ds.load_block(bid)
            for off in range(len(block)):
                collected.append(ds.get_item(block, off))
        assert sorted(collected) == list(range(25))


# ---------------------------------------------------------------------------
# TextBlockDataset
# ---------------------------------------------------------------------------


class TestTextBlockDataset:
    """Tests for raw text / CSV block dataset."""

    @staticmethod
    def _write_lines(path, n: int, *, header: bool = False) -> list[str]:
        """Write *n* lines to *path* and return the expected content lines."""
        lines = []
        with open(path, "w", newline="") as f:
            if header:
                f.write("header_col\n")
            for i in range(n):
                line = f"line_{i}\n"
                f.write(line)
                lines.append(line)
        return lines

    def test_len_matches_lines(self, tmp_path):
        path = str(tmp_path / "data.txt")
        self._write_lines(path, 20)
        ds = TextBlockDataset(path)
        assert len(ds) == 20

    def test_dataset_memory_size(self, tmp_path):
        path = str(tmp_path / "data.txt")
        self._write_lines(path, 20)
        ds = TextBlockDataset(path)
        assert ds.dataset_memory_size > 0
        # Memory size should be close to file size (no header)
        assert ds.dataset_memory_size == os.path.getsize(path)

    def test_num_blocks_for_small_file(self, tmp_path):
        """A small file (< 7MB) should have exactly 1 block."""
        path = str(tmp_path / "data.txt")
        self._write_lines(path, 20)
        ds = TextBlockDataset(path)
        # 20 lines ≈ 140 bytes — way under 7MB
        assert ds.num_blocks == 1

    def test_load_block_returns_correct_lines(self, tmp_path):
        path = str(tmp_path / "data.txt")
        expected = self._write_lines(path, 10)
        ds = TextBlockDataset(path)

        # All lines in one block for small file
        block_0 = ds.load_block(0)
        assert len(block_0) == 10
        for i, raw in enumerate(block_0):
            assert raw.decode().strip() == expected[i].strip()

    def test_get_item_returns_raw_bytes(self, tmp_path):
        path = str(tmp_path / "data.txt")
        self._write_lines(path, 5)
        ds = TextBlockDataset(path)
        block = ds.load_block(0)
        item = ds.get_item(block, 0)
        assert isinstance(item, bytes)

    def test_get_item_with_transform(self, tmp_path):
        path = str(tmp_path / "data.txt")
        self._write_lines(path, 5)
        ds = TextBlockDataset(
            path, transform=lambda b: b.decode().strip().upper()
        )
        block = ds.load_block(0)
        item = ds.get_item(block, 0)
        assert item == "LINE_0"

    def test_has_header_skips_first_line(self, tmp_path):
        path = str(tmp_path / "data.csv")
        self._write_lines(path, 5, header=True)
        ds = TextBlockDataset(path, has_header=True)
        assert len(ds) == 5  # header excluded

    def test_index_file_created_and_reused(self, tmp_path):
        path = str(tmp_path / "data.txt")
        self._write_lines(path, 10)
        idx_path = path + ".idx.txt"

        # First load: index built
        ds1 = TextBlockDataset(path)
        assert os.path.exists(idx_path)

        # Second load: index reused (offsets identical)
        ds2 = TextBlockDataset(path)
        assert ds1.offsets == ds2.offsets

    def test_bijection(self, tmp_path):
        """All items accessed exactly once via load_block + get_item."""
        path = str(tmp_path / "data.txt")
        n = 13
        self._write_lines(path, n)
        ds = TextBlockDataset(path)

        items = []
        for bid in range(ds.num_blocks):
            block = ds.load_block(bid)
            for off in range(len(block)):
                items.append(ds.get_item(block, off))
        assert len(items) == n
        assert len(set(items)) == n  # no duplicates

    def test_load_block_items_convenience(self, tmp_path):
        """load_block_items returns all items ready for buffer."""
        path = str(tmp_path / "data.txt")
        self._write_lines(path, 5)
        ds = TextBlockDataset(path)
        items = ds.load_block_items(0)
        assert len(items) == 5

    def test_empty_file(self, tmp_path):
        path = str(tmp_path / "empty.txt")
        with open(path, "w"):
            pass  # create a zero-byte file
        ds = TextBlockDataset(path)
        assert len(ds) == 0
        assert ds.num_blocks == 0


# ---------------------------------------------------------------------------
# Configurable block size (auto-fio integration)
# ---------------------------------------------------------------------------


class TestConfigurableBlockSize:
    """Block size is overridable per dataset and defaults to BLOCK_SIZE_BYTES."""

    @staticmethod
    def _write_lines(path, n: int) -> None:
        with open(path, "w", newline="") as f:
            for i in range(n):
                f.write(f"line_{i}\n")

    # -- defaults -----------------------------------------------------------

    def test_text_defaults_to_package_constant(self, tmp_path):
        path = str(tmp_path / "data.txt")
        self._write_lines(path, 20)
        ds = TextBlockDataset(path)
        assert ds.block_size_bytes == BLOCK_SIZE_BYTES

    # -- custom value changes blocking --------------------------------------

    def test_text_custom_block_size_is_stored_and_used(self, tmp_path):
        path = str(tmp_path / "data.txt")
        self._write_lines(path, 20)
        default_ds = TextBlockDataset(path)
        # A 1-byte budget forces every line onto its own block (line-aligned,
        # minimum one line per block).
        small_ds = TextBlockDataset(path, block_size_bytes=1)
        assert small_ds.block_size_bytes == 1
        assert default_ds.num_blocks == 1
        assert small_ds.num_blocks == len(small_ds) == 20

    def test_text_custom_block_size_preserves_bijection(self, tmp_path):
        path = str(tmp_path / "data.txt")
        self._write_lines(path, 37)
        ds = TextBlockDataset(path, block_size_bytes=24)  # a few lines per block
        seen = []
        for bid in range(ds.num_blocks):
            block = ds.load_block(bid)
            for off in range(len(block)):
                seen.append(ds.get_item(block, off))
        assert len(seen) == len(ds) == 37  # every line exactly once, in order
        assert seen == [f"line_{i}\n".encode() for i in range(37)]

    # -- validation ---------------------------------------------------------

    @pytest.mark.parametrize("bad", [0, -1, -4096])
    def test_non_positive_block_size_rejected(self, tmp_path, bad):
        path = str(tmp_path / "data.txt")
        self._write_lines(path, 3)
        with pytest.raises(ValueError, match="must be positive"):
            TextBlockDataset(path, block_size_bytes=bad)

    @pytest.mark.parametrize("bad", [1.5, "7000000", True])
    def test_non_int_block_size_rejected(self, tmp_path, bad):
        path = str(tmp_path / "data.txt")
        self._write_lines(path, 3)
        with pytest.raises(TypeError, match="must be an int"):
            TextBlockDataset(path, block_size_bytes=bad)
