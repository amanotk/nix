// -*- C++ -*-
#ifndef _HALO3D_HPP_
#define _HALO3D_HPP_

#include "nix.hpp"
#include "xtensorall.hpp"

NIX_NAMESPACE_BEGIN

///
/// @brief Halo3D
///
template <class T_data, class T_chunk>
class Halo3D
{
public:
  T_data*  data;
  T_chunk* chunk;
  void*    send_buffer;
  void*    recv_buffer;
  int32_t  send_count;
  int32_t  recv_count;

  ///
  /// @brief constructor
  ///
  Halo3D(T_data& data, T_chunk& chunk)
  {
    this->data  = &data;
    this->chunk = &chunk;
  }

  ///
  /// @brief pre-processing for packing
  ///
  template <class T_buffer>
  void pre_pack(T_buffer& mpibuf)
  {
  }

  ///
  /// @brief post-processing for packing
  ///
  template <class T_buffer>
  void post_pack(T_buffer& mpibuf)
  {
  }

  ///
  /// @brief pre-processing for unpacking
  ///
  template <class T_buffer>
  void pre_unpack(T_buffer& mpibuf)
  {
  }

  ///
  /// @brief post-processing for unpacking
  ///
  template <class T_buffer>
  void post_unpack(T_buffer& mpibuf)
  {
  }

  ///
  /// @brief perform packing; return false if send/recv is not required
  /// @param mpibuf pointer to MpiBuffer
  /// @param iz direction in z (either 0, 1, 2)
  /// @param iy direction in y (either 0, 1, 2)
  /// @param ix direction in x (either 0, 1, 2)
  /// @param send_bound send lower- and upper- bounds
  /// @param recv_bound recv lower- and upper- bounds
  ///
  template <class T_buffer>
  bool pack(T_buffer& mpibuf, int iz, int iy, int ix, int send_bound[3][2], int recv_bound[3][2]);

  ///
  /// @brief perform unpacking; return false if send/recv is not required
  /// @param mpibuf pointer to MpiBuffer
  /// @param iz direction in z (either 0, 1, 2)
  /// @param iy direction in y (either 0, 1, 2)
  /// @param ix direction in x (either 0, 1, 2)
  /// @param send_bound send lower- and upper- bounds
  /// @param recv_bound recv lower- and upper- bounds
  ///
  template <class T_buffer>
  bool unpack(T_buffer& mpibuf, int iz, int iy, int ix, int send_bound[3][2], int recv_bound[3][2]);
};

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
