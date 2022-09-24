// -*- C++ -*-
#ifndef _BUFFER_HPP_
#define _BUFFER_HPP_

#include "nix.hpp"

///
/// @brief Buffer for MPI
///
struct Buffer {
  using Pointer = std::unique_ptr<uint8_t[]>;

  int     size; ///< size of buffer in byte
  Pointer data; ///< data pointer

  ///
  /// @brief Constructor
  /// @param s size of buffer
  ///
  Buffer(const int s = 0) : size(s)
  {
    data = std::make_unique<uint8_t[]>(size);
  }

  ///
  /// @brief get raw pointer
  /// @param pos position in byte from the beginning of pointer
  /// @return return pointer
  ///
  uint8_t *get(const int pos = 0)
  {
    return data.get() + pos;
  }

  ///
  /// @brief resize the buffer if 's' is larger than the current (otherwise do nothing)
  /// @param s new size for resize
  ///
  void resize(const int s)
  {
    if (s > size) {
      // allocate new memory and copy contents
      Pointer p = std::make_unique<uint8_t[]>(s);
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
