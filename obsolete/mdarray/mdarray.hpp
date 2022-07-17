// -*- C++ -*-
#ifndef _MDARRAY_HPP_
#define _MDARRAY_HPP_

///
/// Multidimensional Array Wrapper
///
/// $Id$
///

#if  (__cplusplus >= 201402L) || defined(_USE_XTENSOR)
//
// xtensor; requires c++14 or later
//
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"

namespace xt
{

/// wrapper for resize
template <class X, class S>
void resize(X&& x, S&& s)
{
    x.resize(s);
}

/// wrapper for resize with {}
template <class X, class T>
void resize(X&& x, std::initializer_list<T> s)
{
  resize(x, std::vector<T>(s));
}

}

#elif (__cplusplus == 201103L) || defined(_USE_BLITZ)
///
/// blitz; requires c++11
///
/// Only up to 7 dimensional arrays are supported.
///
#include "blitz/array.h"

namespace xt
{
/// alias for blitz::Array
template <class T, int N>
using xtensor = blitz::Array<T,N>;

/// wrapper for resize
template <class X, class S>
void resize(X&& x, S&& s)
{
    if( s.size() == 0 ) {
        std::cerr << "Error in reshape" << std::endl;
    } else if ( s.size() == 1 ) {
        x.resize(s[0]);
    } else if ( s.size() == 2 ) {
        x.resize(s[0], s[1]);
    } else if ( s.size() == 3 ) {
        x.resize(s[0], s[1], s[2]);
    } else if ( s.size() == 4 ) {
        x.resize(s[0], s[1], s[2], s[3]);
    } else if ( s.size() == 5 ) {
        x.resize(s[0], s[1], s[2], s[3], s[4]);
    } else if ( s.size() == 6 ) {
        x.resize(s[0], s[1], s[2], s[3], s[4], s[5]);
    } else if ( s.size() == 7 ) {
        x.resize(s[0], s[1], s[2], s[3], s[4], s[5], s[6]);
    }
}

/// wrapper for resize with {}
template <class X, class T>
void resize(X&& x, std::initializer_list<T> s)
{
  resize(x, std::vector<T>(s));
}


/// Range::all
blitz::Range all()
{
  return blitz::Range::all();
}

/// Range
blitz::Range range(const int begin, const int end, const int stride=1)
{
  return blitz::Range(begin, end-stride, stride);
}

/// view wrapper (1D)
template <class T, class R>
xtensor<T,1> view(xtensor<T,1>& x, R r1)
{
  return x(r1);
}

/// view wrapper (2D)
template <class T, class R>
xtensor<T,2> view(xtensor<T,2>& x, R r1, R r2)
{
  return x(r1, r2);
}

/// view wrapper (3D)
template <class T, class R>
xtensor<T,3> view(xtensor<T,3>& x, R r1, R r2, R r3)
{
  return x(r1, r2, r3);
}

/// view wrapper (4D)
template <class T, class R>
xtensor<T,4> view(xtensor<T,4>& x, R r1, R r2, R r3, R r4)
{
  return x(r1, r2, r3, r4);
}

/// view wrapper (5D)
template <class T, class R>
xtensor<T,5> view(xtensor<T,5>& x, R r1, R r2, R r3, R r4, R r5)
{
  return x(r1, r2, r3, r4, r5);
}

/// view wrapper (6D)
template <class T, class R>
xtensor<T,6> view(xtensor<T,6>& x, R r1, R r2, R r3, R r4, R r5, R r6)
{
  return x(r1, r2, r3, r4, r5, r6);
}

/// view wrapper (7D)
template <class T, class R>
xtensor<T,7> view(xtensor<T,7>& x, R r1, R r2, R r3, R r4, R r5, R r6, R r7)
{
  return x(r1, r2, r3, r4, r5, r6, r7);
}

///
/// The following code is taken and modified from blitz/array-impl.cc.
/// These allows view() to take any combination of int and Range arguments.
///
using nilArraySection = blitz::nilArraySection;

template<typename T_numtype, typename T1, typename T2>
typename blitz::SliceInfo<T_numtype,T1,T2>::T_slice
view(xtensor<T_numtype,2>& x, T1 r1, T2 r2)
{
  typedef typename blitz::SliceInfo<T_numtype,T1,T2>::T_slice slice;
  return slice(x.noConst(), r1, r2,
               nilArraySection(), nilArraySection(), nilArraySection(),
               nilArraySection(), nilArraySection(), nilArraySection(),
               nilArraySection(), nilArraySection(), nilArraySection());
}

template<typename T_numtype, typename T1, typename T2, typename T3>
typename blitz::SliceInfo<T_numtype,T1,T2,T3>::T_slice
view(xtensor<T_numtype,3>& x, T1 r1, T2 r2, T3 r3)
{
  typedef typename blitz::SliceInfo<T_numtype,T1,T2,T3>::T_slice slice;
  return slice(x.noConst(), r1, r2, r3,
               nilArraySection(), nilArraySection(), nilArraySection(),
               nilArraySection(), nilArraySection(), nilArraySection(),
               nilArraySection(), nilArraySection());
}

template<typename T_numtype, typename T1, typename T2, typename T3,
         typename T4>
typename blitz::SliceInfo<T_numtype,T1,T2,T3,T4>::T_slice
view(xtensor<T_numtype,4>& x, T1 r1, T2 r2, T3 r3, T4 r4)
{
  typedef typename blitz::SliceInfo<T_numtype,T1,T2,T3,T4>::T_slice slice;
  return slice(x.noConst(), r1, r2, r3, r4,
               nilArraySection(), nilArraySection(), nilArraySection(),
               nilArraySection(), nilArraySection(), nilArraySection(),
               nilArraySection());
}

template<typename T_numtype, typename T1, typename T2, typename T3,
         typename T4, typename T5>
typename blitz::SliceInfo<T_numtype,T1,T2,T3,T4,T5>::T_slice
view(xtensor<T_numtype,5>& x, T1 r1, T2 r2, T3 r3, T4 r4, T5 r5)
{
  typedef typename blitz::SliceInfo<T_numtype,T1,T2,T3,T4,T5>::T_slice slice;
  return slice(x.noConst(), r1, r2, r3, r4, r5,
               nilArraySection(), nilArraySection(), nilArraySection(),
               nilArraySection(), nilArraySection(), nilArraySection());
}

template<typename T_numtype, typename T1, typename T2, typename T3,
         typename T4, typename T5, typename T6>
typename blitz::SliceInfo<T_numtype,T1,T2,T3,T4,T5,T6>::T_slice
view(xtensor<T_numtype,6>& x, T1 r1, T2 r2, T3 r3, T4 r4, T5 r5, T6 r6)
{
  typedef typename blitz::SliceInfo<T_numtype,T1,T2,T3,T4,T5,T6>::T_slice slice;
  return slice(x.noConst(), r1, r2, r3, r4, r5, r6,
               nilArraySection(), nilArraySection(), nilArraySection(),
               nilArraySection(), nilArraySection());
}

template<typename T_numtype, typename T1, typename T2, typename T3,
         typename T4, typename T5, typename T6, typename T7>
typename blitz::SliceInfo<T_numtype,T1,T2,T3,T4,T5,T6,T7>::T_slice
view(xtensor<T_numtype,7>& x, T1 r1, T2 r2, T3 r3, T4 r4, T5 r5, T6 r6, T7 r7)
{
  typedef typename blitz::SliceInfo<T_numtype,T1,T2,T3,T4,T5,T6,T7>::T_slice slice;
  return slice(x.noConst(), r1, r2, r3, r4, r5, r6, r7,
               nilArraySection(), nilArraySection(), nilArraySection(),
               nilArraySection());
}

}

#else
/// error
#error "No multidimensional array class defined"

#endif

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
