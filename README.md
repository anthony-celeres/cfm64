# CFM64 — Celeres-Feistel Mix 64

**Cryptography-Based Index Permutation with Block-Sequential Access for Textual Data Loading Optimization and Its Impact on Text Classification Accuracy**

> Undergraduate thesis · Anthony L. Celeres · Visayas State University (2026)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## The Problem

Training large models requires shuffling billions of samples every epoch.
Traditional shuffling creates two scaling bottlenecks:

| | Fisher-Yates Shuffle | CFM64 |
|---|---|---|
| **Shuffle memory** | O(N) — 8 GB for 1B samples | **O(1) — 24 bytes** |
| **Disk seeks per epoch** | N random seeks | **N / block_size** sequential reads |
| **Epoch initialisation** | O(N) permutation | **O(1) key derivation** |
| **Checkpoint size** | ~8 GB shuffle state | **24 bytes** |
| **Distributed sync** | Broadcast O(N) array | **Zero sync** (same seed) |

## How It Works

CFM64 replaces the O(N) permutation array with a **two-level Feistel network**
that computes shuffled indices on the fly:

```
Level 1 — Block Order               Level 2 — Intra-Block
┌──────────────────────┐            ┌──────────────────────┐
│ Block 0 ──────►  B3  │            │ Offset 0 ──────► O4  │
│ Block 1 ──────►  B0  │            │ Offset 1 ──────► O2  │
│ Block 2 ──────►  B4  │            │ Offset 2 ──────► O0  │
│  ...                 │            │   ...                │
│ Block N ──────►  B1  │            │ Offset B ──────► O3  │
└──────────────────────┘            └──────────────────────┘
  Shuffles WHICH blocks               Shuffles items WITHIN
  to read from disk                    each block (in RAM)
```

The disk reads stay **sequential within each block** (minimising seeks),
while items are shuffled **both across blocks and within each block** every
epoch. This is *block-level* shuffling, not a global permutation — see
[Scope & Status](#scope--status).

---

## Quick Start

### Install

```bash
pip install cfm64
```

### Basic Usage

```python
from cfm64 import CFM64Shuffle

shuffler = CFM64Shuffle(dataset_size=100_000, block_size=1024, seed=42)

for epoch in range(100):
    shuffler.set_epoch(epoch)                    # O(1) — 24-byte state change
    for block_id, offset, global_idx in shuffler:
        sample = dataset[global_idx]
```

### Full I/O-Optimised Loader

```python
from cfm64 import CFM64Loader, BlockDataset

dataset = BlockDataset(raw_dataset, block_size=1024)
loader = CFM64Loader(dataset, batch_size=64, seed=42)

for epoch in range(100):
    loader.set_epoch(epoch)
    for batch in loader:
        train(batch)
```

---

## Scope & Status

CFM64 is an active undergraduate thesis; this repository is the implementation.
Experimental results (throughput, accuracy, memory) are produced separately and
are **not** included in this repository.

**What CFM64 targets:** text data loading on a single machine with NVMe/SSD
storage. Image, network (S3/NFS), distributed, and billion-scale scenarios are
design goals, not part of the current evaluation.

**Shuffle scope.** CFM64 performs *block-level* shuffling: Level-1 randomises
block visit order and Level-2 shuffles within each ~7 MB block. Most batches
therefore draw from a single block, so per-batch diversity is lower than a
global Fisher–Yates shuffle. This is sound when the on-disk data is **not**
ordered by label; a dataset sorted by class should be shuffled once on disk
first.

---

## Architecture

```
src/cfm64/
├── __init__.py       # Public API surface
├── shuffle.py        # Core: SplitMix64, FeistelPermutation, CFM64Shuffle
├── loader.py         # Full I/O-optimised loader
├── datasets.py       # Block-aware dataset adapters
└── ext/              # C++ native extension (pybind11)
    ├── __init__.py
    ├── bindings.cpp
    └── include/cfm64/
        ├── cfm64.hpp       # Umbrella header
        ├── splitmix64.hpp  # SplitMix64 PRNG
        ├── feistel.hpp     # Feistel network permutation
        └── shuffle.hpp     # Two-level block shuffler
```

### Algorithm

```
Input index X
├── Split into (Left, Right) halves
├── For r = 0 to 3:                          (4 Feistel rounds)
│   ├── F = SplitMix64.hash(RoundKey[r], Right) & half_mask
│   ├── new_LEFT  = Right
│   └── new_RIGHT = (Left ⊕ F) & half_mask
├── Combine: result = (Left << half_bits) | Right
└── Cycle walk: if result >= N, re-apply
```

---

## When CFM64 Helps Most

> _Projected impact by scenario; only text + NVMe is evaluated so far (see [Scope & Status](#scope--status))._

| Scenario | Impact |
|---|---|
| GPU + HDD storage | ⭐⭐⭐⭐⭐ Massive speedup |
| GPU + Network (S3, NFS) | ⭐⭐⭐⭐⭐ Massive speedup |
| Memory-constrained (>100M samples) | ⭐⭐⭐⭐⭐ Essential |
| Distributed training | ⭐⭐⭐⭐ Stateless checkpointing |
| GPU + NVMe SSD | ⭐⭐ Moderate benefit |

---

## Citation

```bibtex
@thesis{cfm642026,
  title   = {Cryptography-Based Index Permutation with Block-Sequential
             Access for Textual Data Loading Optimization and Its Impact
             on Text Classification Accuracy},
  author  = {Celeres, Anthony L.},
  year    = {2026},
  school  = {Visayas State University},
}
```

---

## License

MIT — see [LICENSE](LICENSE).

## References

1. Steele, G. L., Lea, D., & Flood, C. H. (2014). *Fast splittable pseudorandom number generators.* OOPSLA 2014.
2. Black, J., & Rogaway, P. (2002). *Ciphers with arbitrary finite domains.* CT-RSA 2002.
3. Morris, B., Rogaway, P., & Stegers, T. (2009). *How to encipher messages on a small domain.* CRYPTO 2009.

