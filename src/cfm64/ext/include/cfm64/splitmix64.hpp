/**
 * CFM64 — Celeres-Feistel Mix 64
 *
 * SplitMix64 PRNG Implementation
 *
 * A fast, high-quality 64-bit pseudo-random number generator.
 * Reference: Steele, Lea, Flood (2014) "Fast splittable pseudorandom number generators"
 *
 * Properties:
 * - Period: 2^64
 * - State: 64 bits (8 bytes)
 * - Output: 64 bits
 * - Passes BigCrush statistical tests
 */

#ifndef CFM64_SPLITMIX64_HPP
#define CFM64_SPLITMIX64_HPP

#include <cstdint>

namespace cfm64 {

/**
 * SplitMix64 PRNG class
 *
 * Used as the round function for Feistel networks.
 * Provides excellent mixing properties for non-cryptographic applications.
 */
class SplitMix64 {
private:
    uint64_t state_;

public:
    /**
     * Construct with initial seed
     * @param seed Initial state value
     */
    explicit constexpr SplitMix64(uint64_t seed) noexcept : state_(seed) {}

    /**
     * Generate next random value (advances state)
     * @return 64-bit pseudo-random value
     */
    constexpr uint64_t next() noexcept {
        uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    /**
     * Stateless hash function: combines key and data
     *
     * This is the core mixing function used in Feistel rounds.
     * Given the same (key, data) pair, always returns the same output.
     *
     * @param key The round key or seed
     * @param data The input data to hash
     * @return Mixed 64-bit output
     */
    static constexpr uint64_t hash(uint64_t key, uint64_t data) noexcept {
        uint64_t z = key ^ data;
        z += 0x9e3779b97f4a7c15ULL;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    /**
     * Derive a new key from base key and index
     * Useful for generating round keys or epoch-specific seeds
     */
    static constexpr uint64_t derive_key(uint64_t base_key, uint64_t index) noexcept {
        return hash(base_key, index);
    }
};

} // namespace cfm64

#endif // CFM64_SPLITMIX64_HPP
