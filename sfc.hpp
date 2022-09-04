// -*- C++ -*-
#ifndef _SFC_HPP_
#define _SFC_HPP_

#include "common.hpp"
#include "xtensorall.hpp"

#include "thirdparty/tinyformat.hpp"

///
/// Space-Filling Curve
///
/// The implementation of Generalized Hilbert Curve (gilbert) for pseudo Hilbert
/// curve of arbitrary sizes is according to the python code available at:
/// https://github.com/jakubcerveny/gilbert
///

namespace sfc
{
using array1d = xt::xtensor<int, 1>;
using array2d = xt::xtensor<int, 2>;
using array3d = xt::xtensor<int, 3>;

void get_map1d(size_t Nx, array1d &index, array2d &coord);

void get_map2d(size_t Ny, size_t Nx, array2d &index, array2d &coord);

void get_map3d(size_t Nz, size_t Ny, size_t Nx, array3d &index, array2d &coord);

template <class T>
bool check_index(T &index);

bool check_locality2d(array2d &coord, const int distmin = 1);

bool check_locality3d(array2d &coord, const int distmin = 1);

void gilbert2d(array2d &index, int &id, int &x, int &y, int ax, int ay, int bx, int by);

void gilbert3d(array3d &index, int &id, int &x, int &y, int &z, int ax, int ay, int az, int bx,
               int by, int bz, int cx, int cy, int cz);

} // namespace sfc

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
