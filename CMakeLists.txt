cmake_minimum_required(VERSION 3.14)

project(CFM64
    VERSION 0.1.0
    DESCRIPTION "Celeres-Feistel Mix 64 — O(1)-memory shuffling for deep learning"
    LANGUAGES CXX
)

# ============================================================================
# Build Options
# ============================================================================
option(CFM64_BUILD_TESTS "Build C++ unit tests" OFF)
option(CFM64_BUILD_PYTHON "Build Python bindings (requires pybind11)" OFF)

# ============================================================================
# C++ Standard and Compiler Flags
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Position-independent code (required for shared libraries)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compiler-specific flags
if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# ============================================================================
# CFM64 Header-Only Library
# ============================================================================
add_library(cfm64 INTERFACE)
target_include_directories(cfm64 INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/cfm64/ext/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(cfm64 INTERFACE cxx_std_17)

# ============================================================================
# Python Bindings (pybind11)
# ============================================================================
if(CFM64_BUILD_PYTHON)
    find_package(pybind11 CONFIG QUIET)

    if(NOT pybind11_FOUND)
        # Fall back to pip-installed pybind11
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
            OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(PYBIND11_CMAKE_DIR)
            list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
            find_package(pybind11 CONFIG REQUIRED)
        else()
            # Last resort: use pip to find pybind11
            execute_process(
                COMMAND pip show pybind11 --files
                OUTPUT_VARIABLE PYBIND11_INFO
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
            execute_process(
                COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
                OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
            find_package(pybind11 CONFIG REQUIRED)
        endif()
    endif()

    pybind11_add_module(_cfm64_native
        src/cfm64/ext/bindings.cpp
    )
    target_link_libraries(_cfm64_native PRIVATE cfm64)
    target_compile_definitions(_cfm64_native PRIVATE CFM64_VERSION_STRING="${PROJECT_VERSION}")

    # Install the Python extension INTO the cfm64 package so it is importable
    # as `cfm64._cfm64_native`. DESTINATION with no artifact keyword covers the
    # module on every platform (Linux/macOS .so, Windows .pyd); the previous
    # `DESTINATION .` placed it at the wheel root, breaking the import.
    install(TARGETS _cfm64_native DESTINATION cfm64)
endif()

# ============================================================================
# C++ Tests
# ============================================================================
if(CFM64_BUILD_TESTS)
    add_executable(cfm64_test_bijection tests/cpp/test_bijection.cpp)
    target_link_libraries(cfm64_test_bijection PRIVATE cfm64)
    enable_testing()
    add_test(NAME bijection_test COMMAND cfm64_test_bijection)
endif()

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "CFM64 Configuration Summary")
message(STATUS "=========================")
message(STATUS "Version:       ${PROJECT_VERSION}")
message(STATUS "Build Tests:   ${CFM64_BUILD_TESTS}")
message(STATUS "Build Python:  ${CFM64_BUILD_PYTHON}")
message(STATUS "")
