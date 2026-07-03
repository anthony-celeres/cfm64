/**
 * CFM64 — Celeres-Feistel Mix 64
 *
 * Main umbrella header that includes all core CFM64 C++ components.
 *
 * Usage:
 *   #include <cfm64/cfm64.hpp>
 */

#ifndef CFM64_CFM64_HPP
#define CFM64_CFM64_HPP

// Version
#define CFM64_VERSION_MAJOR 0
#define CFM64_VERSION_MINOR 1
#define CFM64_VERSION_PATCH 0
#define CFM64_VERSION_STRING "0.1.0"

// Core components
#include "shuffle.hpp" // IWYU pragma: export

#endif // CFM64_CFM64_HPP
