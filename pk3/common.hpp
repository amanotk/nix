// -*- C++ -*-
#ifndef _COMMON_H_
#define _COMMON_H_

///
/// Common Module
///
/// Author: Takanobu AMANO <amano@eps.s.u-tokyo.ac.jp>
/// $Id: common.hpp,v d8e00b23eb4a 2015/06/23 06:33:27 amano $
///
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <vector>
#include <list>
#include <set>
#include <algorithm>
#include <memory>
#include <bitset>
#include <random>
#include <cassert>
#include <sys/time.h>
#include "config.hpp"
#include "debug.hpp"
#include "tinyformat.hpp"

namespace common
{
// common variables
//@{
// mathematical constants
const float64 pi  = M_PI;   ///< pi
const float64 pi2 = 2*M_PI; ///< 2 pi
const float64 pi4 = 4*M_PI; ///< 4 pi

// utility constants
const float64 HUGEVAL   = HUGE_VAL;
const float64 TOLERANCE = 1.0e-08; ///< default tolerance
const float64 EPSILON   = 1.0e-15; ///< machine epsilon
const float64 NORMMIN   = 1.0e-12; ///< minimum norm (for matrix solvers)

// binary mode
const std::ios::openmode binary_write =
  std::ios::binary | std::ios::out | std::ios::trunc;
const std::ios::openmode binary_append =
  std::ios::binary | std::ios::out | std::ios::app;
const std::ios::openmode binary_read =
  std::ios::binary | std::ios::in;

// text mode
const std::ios::openmode text_write  = std::ios::out | std::ios::trunc;
const std::ios::openmode text_append = std::ios::out | std::ios::app;
const std::ios::openmode text_read   = std::ios::in;

// common functions
//@{

//// wrapper of std::memcpy
inline size_t memcpy_count(void *dst, const void *src, size_t count, bool count_only = false)
{
  if (!count_only) {
    std::memcpy(dst, src, count);
  }
  return count;
}

//// return 1 on little endian and 16777216 on big endian
inline int32_t get_endian_flag()
{
  union {
    int  flag;
    char byte[4] = {1, 0, 0, 0};
  } endian_flag;

  return endian_flag.flag;
}

/// return elapsed time in second
inline double etime()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + (double)tv.tv_usec*1.0e-6;
}

/// return absolute value
template <class T>
inline T absolute(T x)
{
  return (x > 0) ? x : -x;
}

/// return maximum
template <class T>
inline T maximum(T a, T b)
{
  return (a > b) ? a : b;
}

/// return minimum
template <class T>
inline T minimum(T a, T b)
{
  return (a < b) ? a : b;
}

/// minmod limiter
template <class T>
inline T minmod(T a, T b)
{
  return 0.5*(copysign(1.0, a) + copysign(1.0, b)) * minimum(abs(a), abs(b));
}
//@}

}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
