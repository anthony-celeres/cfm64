/**
 * CFM64 Python Bindings (pybind11)
 *
 * Exposes the C++ core as `_cfm64_native` for use by the Python package.
 * The Python layer in `cfm64.shuffle` auto-detects and prefers these
 * native implementations when available.
 *
 * Build:
 *   pip install pybind11
 *   cd CFM64 && mkdir build && cd build
 *   cmake .. -DCFM64_BUILD_PYTHON=ON
 *   cmake --build .
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cfm64/cfm64.hpp>

namespace py = pybind11;



// ============================================================================
// Python Module Definition
// ============================================================================

PYBIND11_MODULE(_cfm64_native, m) {
    m.doc() = R"doc(
CFM64 — Celeres-Feistel Mix 64 (Native C++ Extension)

High-performance C++ core for O(1)-memory shuffling.
This module is auto-detected by `cfm64.shuffle` when available.

Do not import directly — use `from cfm64 import CFM64Shuffle` instead.
)doc";

    // Version info
    m.attr("__version__") = CFM64_VERSION_STRING;



    // ========================================================================
    // SplitMix64 — Core PRNG
    // ========================================================================
    py::class_<cfm64::SplitMix64>(m, "SplitMix64",
        "Fast PRNG based on SplitMix64 algorithm")
        .def(py::init<uint64_t>(), py::arg("seed"))
        .def("next", &cfm64::SplitMix64::next, "Generate next random number")
        .def_static("hash", &cfm64::SplitMix64::hash,
            py::arg("key"), py::arg("data"),
            "Stateless hash function")
        .def_static("derive_key", &cfm64::SplitMix64::derive_key,
            py::arg("base_key"), py::arg("data"),
            "Derive a new key from base key and data");

    // ========================================================================
    // FeistelPermutation — Core permutation
    // ========================================================================
    py::class_<cfm64::FeistelPermutation>(m, "FeistelPermutation",
        R"doc(
Feistel Network Permutation.

Provides bijective (1-to-1) mapping over arbitrary domain sizes.
O(1) memory, O(1) per-index computation.

Args:
    seed: Permutation key
    domain_size: Size of the permutation domain
    rounds: Number of Feistel rounds (default 4)
)doc")
        .def(py::init<uint64_t, size_t, size_t>(),
            py::arg("seed"),
            py::arg("domain_size"),
            py::arg("rounds") = 4)
        .def("permute", &cfm64::FeistelPermutation::permute, py::arg("index"))
        .def("permute_batch", [](const cfm64::FeistelPermutation& self, py::array_t<size_t> indices) {
            auto buf = indices.unchecked<1>();
            auto result = py::array_t<size_t>(buf.shape(0));
            auto res_buf = result.mutable_unchecked<1>();
            for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
                res_buf(i) = self.permute(buf(i));
            }
            return result;
        }, py::arg("indices"))
        .def("inverse_permute", &cfm64::FeistelPermutation::inverse_permute, py::arg("index"))
        .def("size", &cfm64::FeistelPermutation::size)
        .def("key", &cfm64::FeistelPermutation::key)
        .def("__len__", &cfm64::FeistelPermutation::size)
        .def("__getitem__", &cfm64::FeistelPermutation::permute);

    // ========================================================================
    // BlockShuffle — Two-level shuffler
    // ========================================================================
    py::class_<cfm64::BlockShuffle>(m, "BlockShuffle",
        R"doc(
I/O-Aware Block Shuffler.

Level 1: Shuffles which BLOCKS to load from disk (sequential I/O).
Level 2: Shuffles items AFTER block is loaded into RAM.

Args:
    dataset_size: Total number of samples
    block_size: Samples per block
    seed: Epoch seed
    rounds: Feistel rounds (default 4)
)doc")
        .def(py::init<size_t, size_t, uint64_t, size_t>(),
            py::arg("dataset_size"),
            py::arg("block_size"),
            py::arg("seed"),
            py::arg("rounds") = 4)
        // Level 1
        .def("get_block_to_load", &cfm64::BlockShuffle::get_block_to_load, py::arg("position"))
        .def("get_block_order", &cfm64::BlockShuffle::get_block_order)
        .def("get_block_size", &cfm64::BlockShuffle::get_block_size, py::arg("block_id"))
        .def("get_block_start", &cfm64::BlockShuffle::get_block_start, py::arg("block_id"))
        // Level 2
        .def("get_intra_block_order", &cfm64::BlockShuffle::get_intra_block_order, py::arg("block_id"))
        .def("get_item_offset", &cfm64::BlockShuffle::get_item_offset,
            py::arg("block_id"), py::arg("position"))
        .def("expected_locality_factor", &cfm64::BlockShuffle::expected_locality_factor)
        // Accessors
        .def("dataset_size", &cfm64::BlockShuffle::dataset_size)
        .def("block_size", &cfm64::BlockShuffle::block_size)
        .def("num_blocks", &cfm64::BlockShuffle::num_blocks)
        .def("seed", &cfm64::BlockShuffle::seed)
        .def("__len__", &cfm64::BlockShuffle::dataset_size);



}
