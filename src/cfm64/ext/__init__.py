"""
CFM64 native extension — C++ acceleration.

This module exposes the native C++ implementations of
:class:`FeistelPermutation`, :class:`SplitMix64`, and :class:`IOBlockShuffle`.

Build the extension:
    cd CFM64 && mkdir build && cd build
    cmake .. -DCFM64_BUILD_PYTHON=ON
    cmake --build . --config Release
"""

from __future__ import annotations

from cfm64._cfm64_native import (
    BlockShuffle as NativeBlockShuffle,
)
from cfm64._cfm64_native import (
    FeistelPermutation as NativeFeistelPermutation,
)
from cfm64._cfm64_native import (  # type: ignore[import-not-found]
    SplitMix64 as NativeSplitMix64,
)
from cfm64._cfm64_native import (
    get_all_shuffled_indices,
    get_shuffled_indices,
)

__all__ = [
    "NativeSplitMix64",
    "NativeFeistelPermutation",
    "NativeBlockShuffle",
    "get_shuffled_indices",
    "get_all_shuffled_indices",
]
