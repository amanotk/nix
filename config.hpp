// -*- C++ -*-
#ifndef _CONFIG_HPP_
#define _CONFIG_HPP_

///
/// System Dependent Configuration
///
/// This header file tries to absorb system dependence as much as possible for portability.
/// Most of the system specific configurations are probably no longer necessary as of 2022.
///
#include <cstdarg>
#include <cstddef>
#include <cstdint>

//----------------------------------------------------------
#if defined(_SX)
//
// NEC SX Series : assume long as 64bit integer
//
typedef int           int32_t;
typedef long          int64_t;
typedef unsigned int  uint32_t;
typedef unsigned long uint64_t;
#define PRId8 "d"
#define PRId16 "d"
#define PRId32 "d"
#define PRId64 "ld"

#define INLINE inline
#define RESTRICT restrict

#elif defined(__FCC_VERSION) && defined(__sparcv9)
//
// Fujitsu C++ Compiler on SPARC V9 Processor (64bit)
//
// intXX_t seems to be already defined.
//
#define PRId8 "d"
#define PRId16 "d"
#define PRId32 "d"
#define PRId64 "ld"

#define INLINE inline
#define RESTRICT

#elif defined(__xlC__) && defined(_POWER)
//
// IBM XL C/C++ Compiler on Power Processor
//
#if defined(__64BIT__)
typedef int           int32_t;
typedef long          int64_t;
typedef unsigned int  uint32_t;
typedef unsigned long uint64_t;
#define PRId8 "d"
#define PRId16 "d"
#define PRId32 "d"
#define PRId64 "ld"
#else // 32bit
typedef int                int32_t;
typedef long long          int64_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
#define PRId8 "d"
#define PRId16 "d"
#define PRId32 "d"
#define PRId64 "lld"
#endif

#define INLINE inline
#define RESTRICT __restrict__

#elif defined(__GNUC__)
//
// GNU Compiler and Comparitble
//
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <stdint.h>

#define INLINE inline
#define RESTRICT __restrict__

#else
// report error
#error "Error ! You should define some macros for your system"

#endif
//----------------------------------------------------------

// integer type definition : require stdint.h
typedef int32_t  int32;
typedef int64_t  int64;
typedef uint32_t uint32;
typedef uint64_t uint64;
// real type definition
typedef float  float32;
typedef double float64;

// definition of default floating point numbers
typedef float64 real;

//
// SIMD width
//
#if defined(_SIMD_WIDTH)
// user-provided value
#define CONFIG_SIMD_WIDTH _SIMD_WIDTH
#else
// 512 bit for 64bit float by default
#define CONFIG_SIMD_WIDTH 8
#endif

constexpr int config_simd_width = CONFIG_SIMD_WIDTH;

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
