// -*- C++ -*-

///
/// Implementation of Space-Filing Curve
///
#include "sfc.hpp"

namespace sfc
{
inline int sign(const int x)
{
  return x == 0 ? 0 : (x > 0 ? +1 : -1);
}

inline void push2d(array2d &index, int &id, int &x, int &y, int dx, int dy)
{
  id++;
  x += dx;
  y += dy;
  // std::cout << tfm::format("%8d, %4d, %4d\n", id, x, y);
  index.at(y, x) = id;
}

inline void push3d(array3d &index, int &id, int &x, int &y, int &z, int dx, int dy, int dz)
{
  id++;
  x += dx;
  y += dy;
  z += dz;
  // std::cout << tfm::format("%8d, %4d, %4d, %4d\n", id, x, y, z);
  index.at(z, y, x) = id;
}

void get_map1d(size_t Nx, array1d &index, array2d &coord)
{
  for (int ix = 0; ix < Nx; ix++) {
    index(ix) = ix;
    coord(ix, 0) = ix;
  }
}

void get_map2d(size_t Ny, size_t Nx, array2d &index, array2d &coord)
{
  // dimensions must be even
  assert(Nx % 2 == 0);
  assert(Ny % 2 == 0);

  // calculate coordiante to ID map
  int id = -1;
  int x = 0;
  int y = 0;
  push2d(index, id, x, y, 0, 0);

  if (Nx >= Ny) {
    gilbert2d(index, id, x, y, Nx, 0, 0, Ny);
  } else {
    gilbert2d(index, id, x, y, 0, Ny, Nx, 0);
  }

  // calculate ID to coordiante map
  for (int iy = 0; iy < Ny; iy++) {
    for (int ix = 0; ix < Nx; ix++) {
      int id = index(iy, ix);
      coord.at(id, 0) = ix;
      coord.at(id, 1) = iy;
    }
  }
}

void get_map3d(size_t Nz, size_t Ny, size_t Nx, array3d &index, array2d &coord)
{
  // dimensions must be even
  assert(Nx % 2 == 0);
  assert(Ny % 2 == 0);
  assert(Nz % 2 == 0);

  // calculate coordiante to ID map
  int id = -1;
  int x = 0;
  int y = 0;
  int z = 0;
  push3d(index, id, x, y, z, 0, 0, 0);

  if (Nx >= Ny && Nx >= Nz) {
    gilbert3d(index, id, x, y, z, Nx, 0, 0, 0, Ny, 0, 0, 0, Nz);
  } else if (Ny >= Nx && Ny >= Nz) {
    gilbert3d(index, id, x, y, z, 0, Ny, 0, Nx, 0, 0, 0, 0, Nz);
  } else if (Nz >= Nx && Nz >= Ny) {
    gilbert3d(index, id, x, y, z, 0, 0, Nz, Nx, 0, 0, 0, Ny, 0);
  }

  // calculate ID to coordinate map
  for (int iz = 0; iz < Nz; iz++) {
    for (int iy = 0; iy < Ny; iy++) {
      for (int ix = 0; ix < Nx; ix++) {
        int id = index.at(iz, iy, ix);
        coord.at(id, 0) = ix;
        coord.at(id, 1) = iy;
        coord.at(id, 2) = iz;
      }
    }
  }
}

template <class T>
bool check_index(T &index)
{
  bool status = true;

  auto flatindex = xt::sort(xt::flatten(index));
  for (int id = 0; id < flatindex.size(); id++) {
    status = status & (id == flatindex(id));
  }

  return status;
}

bool check_locality2d(array2d &coord, const int distmin)
{
  bool status = true;

  int dx, dy;
  dx = coord(0, 0);
  dy = coord(0, 1);
  for (int id = 1; id < coord.shape(0); id++) {
    int ix = coord(id, 0);
    int iy = coord(id, 1);
    dx = dx - ix;
    dy = dy - iy;
    status = status & (dx * dx + dy * dy <= distmin * distmin);
    dx = ix;
    dy = iy;
  }

  return status;
}

bool check_locality3d(array2d &coord, const int distmin)
{
  bool status = true;

  int dx, dy, dz;
  dx = coord(0, 0);
  dy = coord(0, 1);
  dz = coord(0, 2);
  for (int id = 1; id < coord.shape(0); id++) {
    int ix = coord(id, 0);
    int iy = coord(id, 1);
    int iz = coord(id, 2);
    dx = dx - ix;
    dy = dy - iy;
    dz = dz - iz;
    status = status & (dx * dx + dy * dy + dz * dz <= distmin * distmin);
    dx = ix;
    dy = iy;
    dz = iz;
  }

  return status;
}

void gilbert2d(array2d &index, int &id, int &x, int &y, int ax, int ay, int bx, int by)
{
  int ww = abs(ax + ay);
  int hh = abs(bx + by);
  int dax = sign(ax);
  int day = sign(ay);
  int dbx = sign(bx);
  int dby = sign(by);

  //
  // trivial path
  //
  {
    // 2x2 square
    if (hh == 2 && ww == 2) {
      push2d(index, id, x, y, +dbx, +dby);
      push2d(index, id, x, y, +dax, +day);
      push2d(index, id, x, y, -dbx, -dby);
      return;
    }

    // straight segment in x
    if (hh == 1) {
      for (int i = 1; i < ww; i++) {
        push2d(index, id, x, y, dax, day);
      }
      return;
    }

    // straight segment in y
    if (ww == 1) {
      for (int i = 1; i < hh; i++) {
        push2d(index, id, x, y, dbx, dby);
      }
      return;
    }
  }

  //
  // recursive call
  //
  {
    int ax1 = ax / 2;
    int ay1 = ay / 2;
    int bx1 = bx / 2;
    int by1 = by / 2;
    int ax2 = ax - ax1;
    int ay2 = ay - ay1;
    int bx2 = bx - bx1;
    int by2 = by - by1;
    int ww1 = abs(ax1 + ay1);
    int hh1 = abs(bx1 + by1);

    if (ww1 % 2 == 1 && ww > 2) {
      ax1 = ax1 + dax;
      ay1 = ay1 + day;
      ax2 = ax - ax1;
      ay2 = ay - ay1;
    }

    if (hh1 % 2 == 1 && hh > 2) {
      bx1 = bx1 + dbx;
      by1 = by1 + dby;
      bx2 = bx - bx1;
      by2 = by - by1;
    }

    if (2 * ww > 3 * hh) {
      // split only in x
      gilbert2d(index, id, x, y, +ax1, +ay1, +bx, +by);
      push2d(index, id, x, y, +dax, +day);
      gilbert2d(index, id, x, y, +ax2, +ay2, +bx, +by);
    } else {
      gilbert2d(index, id, x, y, +bx1, +by1, +ax1, +ay1);
      push2d(index, id, x, y, +dbx, +dby);
      gilbert2d(index, id, x, y, +ax, +ay, +bx2, +by2);
      push2d(index, id, x, y, -dbx, -dby);
      gilbert2d(index, id, x, y, -bx1, -by1, -ax2, -ay2);
    }
  }
}

void gilbert3d(array3d &index, int &id, int &x, int &y, int &z, int ax, int ay, int az, int bx,
               int by, int bz, int cx, int cy, int cz)
{
  int ww = abs(ax + ay + az);
  int hh = abs(bx + by + bz);
  int dd = abs(cx + cy + cz);
  int dax = sign(ax);
  int day = sign(ay);
  int daz = sign(az);
  int dbx = sign(bx);
  int dby = sign(by);
  int dbz = sign(bz);
  int dcx = sign(cx);
  int dcy = sign(cy);
  int dcz = sign(cz);

  //
  // trivial path
  //
  {
    // 2x2x2 cube
    if (ww == 2 && hh == 2 && dd == 2) {
      push3d(index, id, x, y, z, +dbx, +dby, +dbz);
      push3d(index, id, x, y, z, +dcx, +dcy, +dcz);
      push3d(index, id, x, y, z, -dbx, -dby, -dbz);
      push3d(index, id, x, y, z, +dax, +day, +daz);
      push3d(index, id, x, y, z, +dbx, +dby, +dbz);
      push3d(index, id, x, y, z, -dcx, -dcy, -dcz);
      push3d(index, id, x, y, z, -dbx, -dby, -dbz);
      return;
    }

    // straight segment in x
    if (hh == 1 && dd == 1) {
      for (int i = 1; i < ww; i++) {
        push3d(index, id, x, y, z, dax, day, daz);
      }
      return;
    }

    // straight segment in y
    if (dd == 1 && ww == 1) {
      for (int i = 1; i < hh; i++) {
        push3d(index, id, x, y, z, dbx, dby, dbz);
      }
      return;
    }

    // straight segment in z
    if (ww == 1 && hh == 1) {
      for (int i = 1; i < dd; i++) {
        push3d(index, id, x, y, z, dcx, dcy, dcz);
      }
      return;
    }
  }

  //
  // recursive call
  //
  {
    int ax1 = ax / 2;
    int ay1 = ay / 2;
    int az1 = az / 2;
    int bx1 = bx / 2;
    int by1 = by / 2;
    int bz1 = bz / 2;
    int cx1 = cx / 2;
    int cy1 = cy / 2;
    int cz1 = cz / 2;
    int ax2 = ax - ax1;
    int ay2 = ay - ay1;
    int az2 = az - az1;
    int bx2 = bx - bx1;
    int by2 = by - by1;
    int bz2 = bz - bz1;
    int cx2 = cx - cx1;
    int cy2 = cy - cy1;
    int cz2 = cz - cz1;
    int ww1 = abs(ax1 + ay1 + az1);
    int hh1 = abs(bx1 + by1 + bz1);
    int dd1 = abs(cx1 + cy1 + cz1);

    if ((ww1 % 2 == 1) && (ww > 2)) {
      ax1 = ax1 + dax;
      ay1 = ay1 + day;
      az1 = az1 + daz;
      ax2 = ax - ax1;
      ay2 = ay - ay1;
      az2 = az - az1;
    }

    if ((hh1 % 2 == 1) && (hh > 2)) {
      bx1 = bx1 + dbx;
      by1 = by1 + dby;
      bz1 = bz1 + dbz;
      bx2 = bx - bx1;
      by2 = by - by1;
      bz2 = bz - bz1;
    }

    if ((dd1 % 2 == 1) && (dd > 2)) {
      cx1 = cx1 + dcx;
      cy1 = cy1 + dcy;
      cz1 = cz1 + dcz;
      cx2 = cx - cx1;
      cy2 = cy - cy1;
      cz2 = cz - cz1;
    }

    if ((2 * ww > 3 * hh) && (2 * ww > 3 * dd)) {
      // split only in x
      gilbert3d(index, id, x, y, z, +ax1, +ay1, +az1, +bx, +by, +bz, +cx, +cy, +cz);
      push3d(index, id, x, y, z, +dax, +day, +daz);
      gilbert3d(index, id, x, y, z, +ax2, +ay2, +az2, +bx, +by, +bz, +cx, +cy, +cz);
    } else if (3 * hh > 4 * dd) {
      // split only in x-y
      gilbert3d(index, id, x, y, z, +bx1, +by1, +bz1, +cx, +cy, +cz, +ax1, +ay1, +az1);
      push3d(index, id, x, y, z, +dbx, +dby, +dbz);
      gilbert3d(index, id, x, y, z, +ax, +ay, +az, +bx2, +by2, +bz2, +cx, +cy, +cz);
      push3d(index, id, x, y, z, -dbx, -dby, -dbz);
      gilbert3d(index, id, x, y, z, -bx1, -by1, -bz1, +cx, +cy, +cz, -ax2, -ay2, -az2);
    } else if (3 * dd > 4 * hh) {
      // split only in x-z
      gilbert3d(index, id, x, y, z, +cx1, +cy1, +cz1, +ax1, +ay1, +az1, +bx, +by, +bz);
      push3d(index, id, x, y, z, +dcx, +dcy, +dcz);
      gilbert3d(index, id, x, y, z, +ax, +ay, +az, +bx, +by, +bz, +cx2, +cy2, +cz2);
      push3d(index, id, x, y, z, -dcx, -dcy, -dcz);
      gilbert3d(index, id, x, y, z, -cx1, -cy1, -cz1, -ax2, -ay2, -az2, +bx, +by, +bz);
    } else {
      // full split
      gilbert3d(index, id, x, y, z, +bx1, +by1, +bz1, +cx1, +cy1, +cz1, +ax1, +ay1, +az1);
      push3d(index, id, x, y, z, dbx, dby, dbz);
      gilbert3d(index, id, x, y, z, +cx, +cy, +cz, +ax1, +ay1, +az1, +bx2, +by2, +bz2);
      push3d(index, id, x, y, z, -dbx, -dby, -dbz);
      gilbert3d(index, id, x, y, z, +ax, +ay, +az, -bx1, -by1, -bz1, -cx2, -cy2, -cz2);
      push3d(index, id, x, y, z, +dbx, +dby, +dbz);
      gilbert3d(index, id, x, y, z, -cx, -cy, -cz, -ax2, -ay2, -az2, +bx2, +by2, +bz2);
      push3d(index, id, x, y, z, -dbx, -dby, -dbz);
      gilbert3d(index, id, x, y, z, -bx1, -by1, -bz1, +cx1, +cy1, +cz1, -ax2, -ay2, -az2);
    }
  }
}

template bool check_index(array1d &index);
template bool check_index(array2d &index);
template bool check_index(array3d &index);

} // namespace sfc

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
