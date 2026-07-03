/**
 * CFM64 — Celeres-Feistel Mix 64
 *
 * Feistel Network Implementation
 *
 * A balanced Feistel network that provides bijective (1-to-1) permutation
 * over arbitrary domain sizes using cycle-walking.
 *
 * Properties:
 * - Bijective: Every input maps to exactly one unique output
 * - Invertible: Can compute reverse permutation
 * - Stateless: Output depends only on (seed, index)
 * - O(1) memory: No arrays needed
 */

#ifndef CFM64_FEISTEL_HPP
#define CFM64_FEISTEL_HPP

#include <cstdint>
#include <cstddef>
#include "splitmix64.hpp"

namespace cfm64 {

/**
 * Feistel Network Permutation
 *
 * Implements a balanced Feistel cipher structure for index permutation.
 * Uses cycle-walking to handle arbitrary domain sizes (not just powers of 2).
 */
class FeistelPermutation {
private:
    uint64_t key_;          // The permutation seed/key
    size_t domain_size_;    // N - total number of items
    size_t half_bits_;      // Bits for each half of the structure
    uint64_t half_mask_;    // Mask for extracting half
    size_t rounds_;         // Number of Feistel rounds

    /**
     * Round function using SplitMix64
     */
    uint64_t round_function(uint64_t round_key, uint64_t input) const noexcept {
        return SplitMix64::hash(round_key, input);
    }

public:
    /**
     * Construct a Feistel permutation
     *
     * @param seed The permutation key (e.g., derived from epoch)
     * @param domain_size Total number of items (N)
     * @param rounds Number of Feistel rounds (default 4)
     */
    FeistelPermutation(uint64_t seed, size_t domain_size, size_t rounds = 4) noexcept
        : key_(seed)
        , domain_size_(domain_size)
        , half_bits_(0)
        , half_mask_(0)
        , rounds_(rounds)
    {
        // total_bits = ceil(log2(domain_size)), computed with integer ops.
        // std::log2 on a double loses precision near/above 2^53, which would
        // make total_bits one too small and silently break the bijection at
        // very large N (cycle-walking could never reach the high indices).
        // Integer bit-width is exact across the full 64-bit domain.
        size_t total_bits;
        if (domain_size <= 1) {
            total_bits = 1;
        } else {
            total_bits = 0;
            size_t v = domain_size - 1;
            while (v > 0) { ++total_bits; v >>= 1; }
            if (total_bits < 1) total_bits = 1;
        }
        half_bits_ = (total_bits + 1) / 2;
        half_mask_ = (1ULL << half_bits_) - 1;
    }

    /**
     * Permute a single index using the Feistel Network
     *
     * Uses cycle-walking to handle arbitrary N (not just powers of 2).
     * Guaranteed to return a unique value in [0, N-1] for each input in [0, N-1].
     *
     * @param index Input index in range [0, N-1]
     * @return Permuted index in range [0, N-1]
     */
    size_t permute(size_t index) const noexcept {
        if (domain_size_ <= 1) return 0;

        size_t result = index;

        do {
            uint64_t left = result >> half_bits_;
            uint64_t right = result & half_mask_;

            for (size_t r = 0; r < rounds_; ++r) {
                uint64_t round_key = SplitMix64::hash(key_, r);
                uint64_t f_out = round_function(round_key, right) & half_mask_;

                uint64_t new_left = right;
                uint64_t new_right = (left ^ f_out) & half_mask_;

                left = new_left;
                right = new_right;
            }

            result = (left << half_bits_) | right;

        } while (result >= domain_size_);

        return result;
    }

    /**
     * Inverse permutation
     *
     * Given a permuted index, recover the original index.
     * inverse_permute(permute(i)) == i for all i in [0, N-1]
     *
     * @param index Permuted index in range [0, N-1]
     * @return Original index in range [0, N-1]
     */
    size_t inverse_permute(size_t index) const noexcept {
        if (domain_size_ <= 1) return 0;

        size_t result = index;

        do {
            uint64_t left = result >> half_bits_;
            uint64_t right = result & half_mask_;

            // Apply Feistel rounds in REVERSE order
            for (size_t r = rounds_; r > 0; --r) {
                uint64_t round_key = SplitMix64::hash(key_, r - 1);
                uint64_t f_out = round_function(round_key, left) & half_mask_;

                uint64_t new_right = left;
                uint64_t new_left = (right ^ f_out) & half_mask_;

                left = new_left;
                right = new_right;
            }

            result = (left << half_bits_) | right;

        } while (result >= domain_size_);

        return result;
    }

    // Accessors
    size_t size() const noexcept { return domain_size_; }
    uint64_t key() const noexcept { return key_; }
    size_t rounds() const noexcept { return rounds_; }
};

} // namespace cfm64

#endif // CFM64_FEISTEL_HPP
