// -*- C++ -*-
#ifndef _CONFIG_HPP_
#define _CONFIG_HPP_

///
/// System Dependent Configuration
///
/// The data size for 'long' depends on your systems: might be 32bit on 32bit
/// machines and 64bit on 64bit machines, but this may change, for example,
/// according to compilers flags. The C99 standard introduces a safe way to use
/// 32bit and 64bit integers, however, some C++ compilers do not provide full
/// C99 support. This dependency can be virtually removed by adding some
/// configuration macros for your system. You should at least be able to use
/// (u)int32_t, (u)int64_t and their correspoinding format macros. You may also
/// define macros which may not be supported yet, i.e., 'inline' and 'restrict'.
///
/// Author: Takanobu AMANO <amano@eps.s.u-tokyo.ac.jp>
/// $Id: config.hpp,v b3eb68886539 2012/08/06 08:56:55 amano $
///
#include <cstddef>
#include <cstdint>
#include <cstdarg>

//----------------------------------------------------------
#if   defined (_SX)
//
// NEC SX Series : assume long as 64bit integer
//
typedef int           int32_t;
typedef long          int64_t;
typedef unsigned int  uint32_t;
typedef unsigned long uint64_t;
#define PRId8         "d"
#define PRId16        "d"
#define PRId32        "d"
#define PRId64        "ld"

#define INLINE   inline
#define RESTRICT restrict

#elif defined(__FCC_VERSION) && defined(__sparcv9)
//
// Fujitsu C++ Compiler on SPARC V9 Processor (64bit)
//
// intXX_t seems to be already defined.
//
#define PRId8         "d"
#define PRId16        "d"
#define PRId32        "d"
#define PRId64        "ld"

#define INLINE   inline
#define RESTRICT

//
// Following MPI type specifications with explict data size should be defined
// accroding to the MPI standard:
// - MPI::INTEGER4
// - MPI::INTEGER8
// - MPI::REAL4
// - MPI::REAL8
// However, these are NOT defined in Fujitsu MPI.
// Following macro definitions avoid this problem.
//
#define INTEGER4 INT
#define INTEGER8 LONG
#define REAL4    FLOAT
#define REAL8    DOUBLE

#elif defined(__xlC__) && defined(_POWER)
//
// IBM XL C/C++ Compiler on Power Processor
//
#if   defined(__64BIT__)
typedef int           int32_t;
typedef long          int64_t;
typedef unsigned int  uint32_t;
typedef unsigned long uint64_t;
#define PRId8         "d"
#define PRId16        "d"
#define PRId32        "d"
#define PRId64        "ld"
#else // 32bit
typedef int                int32_t;
typedef long long          int64_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
#define PRId8         "d"
#define PRId16        "d"
#define PRId32        "d"
#define PRId64        "lld"
#endif

#define INLINE   inline
#define RESTRICT __restrict__

#elif defined (__GNUC__)
//
// GNU Compiler and Comparitble
//
#define __STDC_FORMAT_MACROS
#include <stdint.h>
#include <inttypes.h>

#define INLINE   inline
#define RESTRICT __restrict__

#else
// report error
#error "Error ! You should define some macros for your system"

#endif
//----------------------------------------------------------

// integer type definition : require stdint.h
typedef int32_t        int32;
typedef int64_t        int64;
typedef uint32_t       uint32;
typedef uint64_t       uint64;
// real type definition
typedef float          float32;
typedef double         float64;

// definition of default floating point numbers
typedef float64        real;

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
