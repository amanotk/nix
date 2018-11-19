// -*- C++ -*-
#ifndef _BUFFER_HPP_
#define _BUFFER_HPP_

///
/// Buffer
///
/// $Id$
///
#include "base/common.hpp"

///
/// Buffer for MPI
///
struct Buffer
{
  typedef std::unique_ptr<char[]> Pointer;

  int     size;
  Pointer data = nullptr;

  /// constructor
  Buffer(const int s=0)
    : size(s)
  {
    data.reset(new char[size]);
  }

  /// get raw pointer
  char* get(const int pos=0)
  {
    return data.get() + pos;
  }

  /// resize
  bool resize(const int s)
  {
    if( s > size ) {
      // allocate new memory and copy contents
      Pointer p(new char[s]);
      std::memcpy(data.get(), p.get(), size);

      // move
      data = std::move(p);
      size = s;
    }
  }
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
