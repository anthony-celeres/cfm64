/**
 * CFM64 — Celeres-Feistel Mix 64
 *
 * I/O-Aware Block Shuffle
 *
 * Implements the core two-level shuffle that makes CFM64's data loading
 * I/O-efficient while maintaining full permutation quality.
 *
 * Level 1 (Inter-block): Shuffles which blocks to load from disk.
 *   → Determines the ORDER blocks are read (sequential I/O within blocks).
 *
 * Level 2 (Intra-block): Shuffles items within a loaded block.
 *   → Applied AFTER the block is in memory (no I/O penalty).
 */

#ifndef CFM64_SHUFFLE_HPP
#define CFM64_SHUFFLE_HPP

#include <cstdint>
#include <cstddef>
#include <vector>
#include <algorithm>
#include "feistel.hpp"

namespace cfm64 {

/**
 * BlockShuffle — Two-level I/O-aware shuffler
 */
class BlockShuffle {
private:
    size_t dataset_size_;
    size_t block_size_;
    size_t num_blocks_;
    uint64_t seed_;
    size_t rounds_;
    FeistelPermutation block_order_;   // Level 1: inter-block shuffle

public:
    BlockShuffle(size_t dataset_size, size_t block_size, uint64_t seed, size_t rounds = 4)
        : dataset_size_(dataset_size)
        , block_size_(block_size)
        , num_blocks_((dataset_size + block_size - 1) / block_size)
        , seed_(seed)
        , rounds_(rounds)
        , block_order_(SplitMix64::derive_key(seed, 0), num_blocks_, rounds)
    {}

    // ========================================================================
    // Level 1 — Block loading order
    // ========================================================================

    /** Which block to load at position `pos` in the epoch */
    size_t get_block_to_load(size_t position) const {
        return block_order_.permute(position);
    }

    /** Full shuffled block order */
    std::vector<size_t> get_block_order() const {
        std::vector<size_t> order(num_blocks_);
        for (size_t i = 0; i < num_blocks_; ++i)
            order[i] = block_order_.permute(i);
        return order;
    }

    /** Actual size of block (last block may be smaller) */
    size_t get_block_size(size_t block_id) const {
        size_t start = block_id * block_size_;
        return std::min(block_size_, dataset_size_ - std::min(start, dataset_size_));
    }

    /** Starting global index of a block */
    size_t get_block_start(size_t block_id) const {
        return block_id * block_size_;
    }

    // ========================================================================
    // Level 2 — Intra-block shuffle (applied in memory)
    // ========================================================================

    /** Shuffled offsets for items within a block */
    std::vector<size_t> get_intra_block_order(size_t block_id) const {
        size_t actual = get_block_size(block_id);
        if (actual == 0) return {};

        uint64_t item_key = SplitMix64::derive_key(seed_, block_id + 1);
        FeistelPermutation item_perm(item_key, actual, rounds_);

        std::vector<size_t> order(actual);
        for (size_t i = 0; i < actual; ++i)
            order[i] = item_perm.permute(i);
        return order;
    }

    /** Single item offset within a block */
    size_t get_item_offset(size_t block_id, size_t position) const {
        size_t actual = get_block_size(block_id);
        uint64_t item_key = SplitMix64::derive_key(seed_, block_id + 1);
        FeistelPermutation item_perm(item_key, actual, rounds_);
        return item_perm.permute(position);
    }

    /** Shuffle a data array in-place using the intra-block permutation */
    template<typename T>
    void shuffle_block_inplace(size_t block_id, T* data, size_t count) const {
        auto order = get_intra_block_order(block_id);
        size_t n = std::min(count, order.size());

        std::vector<T> temp(data, data + n);
        for (size_t i = 0; i < n; ++i) {
            data[i] = temp[order[i]];
        }
    }

    // ========================================================================
    // Full iteration order (for testing/debugging)
    // ========================================================================

    /** Get complete iteration order: for each block in shuffled order, yield shuffled items */
    std::vector<size_t> get_full_iteration_order() const {
        std::vector<size_t> result;
        result.reserve(dataset_size_);

        for (size_t pos = 0; pos < num_blocks_; ++pos) {
            size_t block_id = block_order_.permute(pos);
            size_t start = block_id * block_size_;
            auto intra = get_intra_block_order(block_id);

            for (size_t offset : intra) {
                size_t global = start + offset;
                if (global < dataset_size_)
                    result.push_back(global);
            }
        }
        return result;
    }

    /** Expected locality factor */
    double expected_locality_factor() const {
        return static_cast<double>(block_size_) / dataset_size_;
    }

    // Accessors
    size_t dataset_size() const { return dataset_size_; }
    size_t block_size() const { return block_size_; }
    size_t num_blocks() const { return num_blocks_; }
    uint64_t seed() const { return seed_; }
};

} // namespace cfm64

#endif // CFM64_SHUFFLE_HPP
