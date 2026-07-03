"""Tests for the buffer-free CFM64Loader."""

import pytest

from cfm64.datasets import BlockDataset, TextBlockDataset
from cfm64.loader import CFM64Loader

# torch is an optional dependency; skip this whole module if it is unavailable
# (e.g. on a brand-new Python version before torch ships wheels for it).
torch = pytest.importorskip("torch")


@pytest.fixture
def text_dataset(tmp_path):
    """Create a small text dataset with 200 lines."""
    data_path = tmp_path / "data.txt"
    with open(data_path, "w") as f:
        for i in range(200):
            f.write(f"{i}\n")

    def transform(line):
        val = int(line.strip())
        return torch.tensor([val], dtype=torch.float32), val

    return TextBlockDataset(str(data_path), transform=transform)


class _IntBlockStub(BlockDataset):
    """In-memory integer dataset with a controllable number of blocks.

    Lets loader tests exercise multi-block iteration (including a partial last
    block) without writing multi-megabyte files to disk.
    """

    def __init__(self, n_items: int, items_per_block: int) -> None:
        self._items = list(range(n_items))
        self._ranges = [
            (s, min(s + items_per_block, n_items))
            for s in range(0, n_items, items_per_block)
        ]

    def __len__(self) -> int:
        return len(self._items)

    @property
    def dataset_memory_size(self) -> int:
        return max(len(self._items) * 8, 1)

    @property
    def num_blocks(self) -> int:
        return len(self._ranges)

    def load_block(self, block_id: int):
        start, end = self._ranges[block_id]
        return self._items[start:end]

    def get_item(self, block_data, offset: int):
        return int(block_data[offset])


class TestCFM64Loader:
    """Core loader iteration tests."""

    def test_iterates_all_samples(self, text_dataset):
        """Every item yielded exactly once per epoch (bijection guarantee)."""
        loader = CFM64Loader(text_dataset, batch_size=16, seed=42)
        all_items = []
        for batch_data, batch_labels in loader:
            all_items.extend(batch_data.flatten().tolist())
        assert sorted(all_items) == list(range(200))

    def test_batch_size(self, text_dataset):
        """All but the last batch should have exactly batch_size items."""
        loader = CFM64Loader(text_dataset, batch_size=16, seed=42)
        batches = list(loader)
        for data, labels in batches[:-1]:
            assert len(data) == 16

    def test_len(self, text_dataset):
        """__len__ returns ceil(dataset_size / batch_size)."""
        loader = CFM64Loader(text_dataset, batch_size=16, seed=42)
        # 200 / 16 = 12.5 → ceil = 13
        assert len(loader) == 13

    def test_set_epoch_changes_order(self, text_dataset):
        """Different epochs produce different item orderings."""
        loader = CFM64Loader(text_dataset, batch_size=200, seed=42)
        batch0 = list(loader)[0][0].flatten().tolist()
        loader.set_epoch(1)
        batch1 = list(loader)[0][0].flatten().tolist()
        assert batch0 != batch1

    def test_epoch_determinism(self, text_dataset):
        """Same epoch + seed → identical output across runs."""
        def run_epoch(seed, epoch):
            loader = CFM64Loader(text_dataset, batch_size=200, seed=seed)
            loader.set_epoch(epoch)
            return list(loader)[0][0].flatten().tolist()

        assert run_epoch(42, 3) == run_epoch(42, 3)

    def test_invalid_batch_size(self, text_dataset):
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            CFM64Loader(text_dataset, batch_size=0, seed=42)


class TestCFM64LoaderExactlyOnce:
    """Bijection guarantee holds across a range of block layouts."""

    @pytest.mark.parametrize(
        "n_items, items_per_block",
        [(100, 10), (97, 8), (256, 1), (50, 50), (1, 1), (13, 5)],
    )
    def test_exactly_once(self, n_items, items_per_block):
        ds = _IntBlockStub(n_items, items_per_block)
        loader = CFM64Loader(ds, batch_size=16, seed=42)
        seen = []
        for batch in loader:
            seen.extend(batch.flatten().tolist())
        assert sorted(seen) == list(range(n_items))

    def test_blocks_visited_in_permuted_order(self):
        """Level-1 block shuffle: blocks are not visited sequentially."""
        # 20 single-item blocks → each batch's value is its source block id.
        ds = _IntBlockStub(20, 1)
        loader = CFM64Loader(ds, batch_size=1, seed=42)
        order = [b.flatten().tolist()[0] for b in loader]
        assert sorted(order) == list(range(20))   # all blocks visited once
        assert order != list(range(20))           # but in shuffled order


class TestCFM64LoaderRobustness:
    def test_empty_dataset_yields_nothing(self):
        ds = _IntBlockStub(0, 1)
        loader = CFM64Loader(ds, batch_size=16, seed=42)
        assert list(loader) == []

    def test_from_checkpoint_rejects_dataset_size_mismatch(self, text_dataset):
        loader = CFM64Loader(text_dataset, batch_size=16, seed=42)
        state = loader.get_state()
        state["dataset_size"] = 999_999  # simulate a different dataset
        with pytest.raises(ValueError, match="does not match"):
            CFM64Loader.from_checkpoint(state, text_dataset)


class TestCFM64LoaderCheckpoint:
    """Checkpoint and state management tests."""

    def test_get_state_roundtrip(self, text_dataset):
        loader = CFM64Loader(text_dataset, batch_size=16, seed=42)
        loader.set_epoch(5)
        state = loader.get_state()
        assert state["epoch"] == 5
        assert state["seed"] == 42

    def test_from_checkpoint_restores(self, text_dataset):
        loader = CFM64Loader(text_dataset, batch_size=16, seed=42)
        loader.set_epoch(5)
        state = loader.get_state()

        restored = CFM64Loader.from_checkpoint(state, text_dataset)
        assert restored._epoch == 5
        assert restored.seed == 42
        assert restored.batch_size == 16


class TestCFM64LoaderIOStats:
    """I/O statistics tests."""

    def test_io_stats_keys(self, text_dataset):
        loader = CFM64Loader(text_dataset, batch_size=16, seed=42)
        stats = loader.get_io_stats()
        assert stats["dataset_size"] == 200
        assert stats["num_blocks"] >= 1
        assert stats["block_size_mb"] == 7.0
        assert stats["seek_reduction"] >= 1
        assert stats["seeks_per_epoch"] == stats["num_blocks"]
