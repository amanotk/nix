// -*- C++ -*-
#ifndef _COMMON_H_
#define _COMMON_H_

///
/// common includes and definitions
///
#include "debug.hpp"

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <sys/time.h>
#include <vector>

#include <nlohmann/json.hpp>

// integer types
using int32  = int32_t;
using int64  = int64_t;
using uint32 = uint32_t;
using uint64 = uint64_t;

// floating point number types
using float32 = float;
using float64 = double;
using real    = float64;

//
// SIMD width (512 bit for 64bit float by default)
//
#ifndef NIX_SIMD_WIDTH
#define NIX_SIMD_WIDTH 8
#endif

constexpr int nix_simd_width = NIX_SIMD_WIDTH;


namespace common
{
using json = nlohmann::ordered_json;

// mathematical constants
const float64 pi  = M_PI;     ///< pi
const float64 pi2 = 2 * M_PI; ///< 2 pi
const float64 pi4 = 4 * M_PI; ///< 4 pi

// binary mode
const std::ios::openmode binary_write  = std::ios::binary | std::ios::out | std::ios::trunc;
const std::ios::openmode binary_append = std::ios::binary | std::ios::out | std::ios::app;
const std::ios::openmode binary_read   = std::ios::binary | std::ios::in;

// text mode
const std::ios::openmode text_write  = std::ios::out | std::ios::trunc;
const std::ios::openmode text_append = std::ios::out | std::ios::app;
const std::ios::openmode text_read   = std::ios::in;

// send/recv mode
enum SendRecvMode {
  SendMode = 0b01000000000000, // 4096
  RecvMode = 0b10000000000000, // 8192
};

///
/// @brief return elapsed time
/// @return elapsed time in second
///
inline double etime()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + (double)tv.tv_usec * 1.0e-6;
}

///
/// @brief return flag to determine endian of the system
/// @return 1 on little endian and 16777216 on big endian
///
inline int32_t get_endian_flag()
{
  union {
    int32_t flag;
    uint8_t byte[4] = {1, 0, 0, 0};
  } endian_flag;

  return endian_flag.flag;
}

///
/// @brief convinient wrapper for std::memcpy
/// @param dst destination buffer pointer
/// @param src source buffer pointer
/// @param count number of bytes to be copied
/// @param dstaddr offset from dst pointer
/// @param srcaddr offset from src pointer
/// @return number of bytes copied
///
/// If both `dst` or `src` are not `nullptr`, then data copy will be performed and it return the
/// number of bytes that are copied from `src` to `dst`. Otherwise, it will be in a query mode, in
/// which the copy operation will not be performed but the number of bytes to be copied will be
/// returned.
/// For coninience, `dstaddr` and `srcaddr` can be specified as offsets to the buffer pointers.
///
inline size_t memcpy_count(void *dst, void *src, size_t count, size_t dstaddr, size_t srcaddr)
{
  if (dst != nullptr && src != nullptr) {
    uint8_t *dstptr = &static_cast<uint8_t *>(dst)[dstaddr];
    uint8_t *srcptr = &static_cast<uint8_t *>(src)[srcaddr];
    std::memcpy(dstptr, srcptr, count);
  }
  return count;
}

} // namespace common

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
