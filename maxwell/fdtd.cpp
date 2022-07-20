// -*- C++ -*-
#include "fdtd.hpp"

#define DEFINE_MEMBER(type, name) type FDTD::name

DEFINE_MEMBER(, FDTD)(const int dims[3], const int id) : BaseChunk<3>(dims, id)
{
  size_t Nz = shape[0] + 2 * Nb;
  size_t Ny = shape[1] + 2 * Nb;
  size_t Nx = shape[2] + 2 * Nb;

  // lower and upper bound
  Lbz = Nb;
  Ubz = shape[0] + Nb - 1;
  Lby = Nb;
  Uby = shape[1] + Nb - 1;
  Lbx = Nb;
  Ubx = shape[2] + Nb - 1;

  // memory allocation
  zc.resize({Nz});
  yc.resize({Ny});
  xc.resize({Nx});
  uf.resize({Nz, Ny, Nx, 6});
  zc.fill(0);
  yc.fill(0);
  xc.fill(0);
  uf.fill(0);
}

DEFINE_MEMBER(, ~FDTD)()
{
  zc.resize({0});
  yc.resize({0});
  xc.resize({0});
  uf.resize({0});
}

DEFINE_MEMBER(void, initialize_load)()
{
  load = 0;
}

DEFINE_MEMBER(float64, get_load)()
{
  return load;
}

DEFINE_MEMBER(int, pack)(const int mode, void *buffer)
{
  using common::memcpy_count;

  int   count = 0;
  char *ptr   = static_cast<char *>(buffer);

  switch (mode) {
  case PackAll:
    count += BaseChunk<3>::pack(BaseChunk<3>::PackAll, &ptr[count]);
    count += memcpy_count(&ptr[count], xc.data(), xc.size() * sizeof(float64), false);
    count += memcpy_count(&ptr[count], yc.data(), yc.size() * sizeof(float64), false);
    count += memcpy_count(&ptr[count], zc.data(), zc.size() * sizeof(float64), false);
    count += memcpy_count(&ptr[count], uf.data(), uf.size() * sizeof(float64), false);
    count += memcpy_count(&ptr[count], &cc, sizeof(float64), false);
    count += memcpy_count(&ptr[count], &delh, sizeof(float64), false);
    count += memcpy_count(&ptr[count], xlim, 3 * sizeof(float64), false);
    count += memcpy_count(&ptr[count], ylim, 3 * sizeof(float64), false);
    count += memcpy_count(&ptr[count], zlim, 3 * sizeof(float64), false);
    break;
  case PackAllQuery:
    count += BaseChunk<3>::pack(BaseChunk<3>::PackAllQuery, &ptr[count]);
    count += memcpy_count(&ptr[count], xc.data(), xc.size() * sizeof(float64), true);
    count += memcpy_count(&ptr[count], yc.data(), yc.size() * sizeof(float64), true);
    count += memcpy_count(&ptr[count], zc.data(), zc.size() * sizeof(float64), true);
    count += memcpy_count(&ptr[count], uf.data(), uf.size() * sizeof(float64), true);
    count += memcpy_count(&ptr[count], &cc, sizeof(float64), true);
    count += memcpy_count(&ptr[count], &delh, sizeof(float64), true);
    count += memcpy_count(&ptr[count], xlim, 3 * sizeof(float64), true);
    count += memcpy_count(&ptr[count], ylim, 3 * sizeof(float64), true);
    count += memcpy_count(&ptr[count], zlim, 3 * sizeof(float64), true);
    break;
  case PackEmf:
    count = pack_diagnostic(false, buffer);
    break;
  case PackEmfQuery:
    count = pack_diagnostic(true, buffer);
    break;
  default:
    break;
  }

  return count;
}

DEFINE_MEMBER(int, unpack)(const int mode, void *buffer)
{
  using common::memcpy_count;

  int   count = 0;
  char *ptr   = static_cast<char *>(buffer);

  switch (mode) {
  case PackAll:
    count += BaseChunk<3>::unpack(BaseChunk<3>::PackAll, &ptr[count]);
    count += memcpy_count(xc.data(), &ptr[count], xc.size() * sizeof(float64), false);
    count += memcpy_count(yc.data(), &ptr[count], yc.size() * sizeof(float64), false);
    count += memcpy_count(zc.data(), &ptr[count], zc.size() * sizeof(float64), false);
    count += memcpy_count(uf.data(), &ptr[count], uf.size() * sizeof(float64), false);
    count += memcpy_count(&cc, &ptr[count], sizeof(float64), false);
    count += memcpy_count(&delh, &ptr[count], sizeof(float64), false);
    count += memcpy_count(xlim, &ptr[count], 3 * sizeof(float64), false);
    count += memcpy_count(ylim, &ptr[count], 3 * sizeof(float64), false);
    count += memcpy_count(zlim, &ptr[count], 3 * sizeof(float64), false);
    break;
  case PackAllQuery:
    count += BaseChunk<3>::unpack(BaseChunk<3>::PackAllQuery, &ptr[count]);
    count += memcpy_count(xc.data(), &ptr[count], xc.size() * sizeof(float64), true);
    count += memcpy_count(yc.data(), &ptr[count], yc.size() * sizeof(float64), true);
    count += memcpy_count(zc.data(), &ptr[count], zc.size() * sizeof(float64), true);
    count += memcpy_count(uf.data(), &ptr[count], uf.size() * sizeof(float64), true);
    count += memcpy_count(&cc, &ptr[count], sizeof(float64), true);
    count += memcpy_count(&delh, &ptr[count], sizeof(float64), true);
    count += memcpy_count(xlim, &ptr[count], 3 * sizeof(float64), true);
    count += memcpy_count(ylim, &ptr[count], 3 * sizeof(float64), true);
    count += memcpy_count(zlim, &ptr[count], 3 * sizeof(float64), true);
    break;
  case PackEmf:
    ERRORPRINT("Not implemented yet");
    break;
  case PackEmfQuery:
    ERRORPRINT("Not implemented yet");
    break;
  default:
    break;
  }

  return count;
}

DEFINE_MEMBER(void, setup)
(const float64 cc, const float64 delh, const int offset[3], T_function initializer)
{
  this->cc        = cc;
  this->delh      = delh;
  this->offset[0] = offset[0];
  this->offset[1] = offset[1];
  this->offset[2] = offset[2];

  zlim[0] = offset[0] * delh;
  zlim[1] = offset[0] * delh + shape[0] * delh;
  zlim[2] = zlim[1] - zlim[0];
  ylim[0] = offset[1] * delh;
  ylim[1] = offset[1] * delh + shape[1] * delh;
  ylim[2] = ylim[1] - ylim[0];
  xlim[0] = offset[2] * delh;
  xlim[1] = offset[2] * delh + shape[2] * delh;
  xlim[2] = xlim[1] - xlim[0];

  // set coordinate
  for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
    zc(iz) = zlim[0] + (iz + 0.5) * delh;
  }

  for (int iy = Lby - Nb; iy <= Uby + Nb; iy++) {
    yc(iy) = ylim[0] + (iy + 0.5) * delh;
  }

  for (int ix = Lbx - Nb; ix <= Ubx + Nb; ix++) {
    xc(ix) = xlim[0] + (ix + 0.5) * delh;
  }

  // set initial condition
  for (int iz = Lbz; iz <= Ubz; iz++) {
    for (int iy = Lby; iy <= Uby; iy++) {
      for (int ix = Lbx; ix <= Ubx; ix++) {
        initializer(zc(iz), yc(iy), xc(ix), &uf(iz, iy, ix, 0));
      }
    }
  }

  // set boundary condition
  set_boundary();
}

DEFINE_MEMBER(void, push)(const float64 delt)
{
  const float64 cfl = cc * delt / delh;

  float64 etime = common::etime();

  // advance E-field
  for (int iz = Lbz - 1; iz <= Ubz; iz++) {
    for (int iy = Lby - 1; iy <= Uby; iy++) {
      for (int ix = Lbx - 1; ix <= Ubx; ix++) {
        uf(iz, iy, ix, 0) += (+cfl) * (uf(iz, iy + 1, ix, 5) - uf(iz, iy, ix, 5)) +
                             (-cfl) * (uf(iz + 1, iy, ix, 4) - uf(iz, iy, ix, 4));
        uf(iz, iy, ix, 1) += (+cfl) * (uf(iz + 1, iy, ix, 3) - uf(iz, iy, ix, 3)) +
                             (-cfl) * (uf(iz, iy, ix + 1, 5) - uf(iz, iy, ix, 5));
        uf(iz, iy, ix, 2) += (+cfl) * (uf(iz, iy, ix + 1, 4) - uf(iz, iy, ix, 4)) +
                             (-cfl) * (uf(iz, iy + 1, ix, 3) - uf(iz, iy, ix, 3));
      }
    }
  }

  // advance B-field
  for (int iz = Lbz; iz <= Ubz; iz++) {
    for (int iy = Lby; iy <= Uby; iy++) {
      for (int ix = Lbx; ix <= Ubx; ix++) {
        uf(iz, iy, ix, 3) += (-cfl) * (uf(iz, iy, ix, 2) - uf(iz, iy - 1, ix, 2)) +
                             (+cfl) * (uf(iz, iy, ix, 1) - uf(iz - 1, iy, ix, 1));
        uf(iz, iy, ix, 4) += (-cfl) * (uf(iz, iy, ix, 0) - uf(iz - 1, iy, ix, 0)) +
                             (+cfl) * (uf(iz, iy, ix, 2) - uf(iz, iy, ix - 1, 2));
        uf(iz, iy, ix, 5) += (-cfl) * (uf(iz, iy, ix, 1) - uf(iz, iy, ix - 1, 1)) +
                             (+cfl) * (uf(iz, iy, ix, 0) - uf(iz, iy - 1, ix, 0));
      }
    }
  }

  // set boundary condition
  set_boundary();

  // store computation time
  load += common::etime() - etime;
}

DEFINE_MEMBER(void, set_boundary)()
{
  auto I = xt::all();

  // z dir
  xt::view(uf, Lbz - 1, I, I, I) = xt::view(uf, Ubz, I, I, I);
  xt::view(uf, Ubz + 1, I, I, I) = xt::view(uf, Lbz, I, I, I);
  // y dir
  xt::view(uf, I, Lby - 1, I, I) = xt::view(uf, I, Uby, I, I);
  xt::view(uf, I, Uby + 1, I, I) = xt::view(uf, I, Lby, I, I);
  // x dir
  xt::view(uf, I, I, Lbx - 1, I) = xt::view(uf, I, I, Ubx, I);
  xt::view(uf, I, I, Ubx + 1, I) = xt::view(uf, I, I, Lbx, I);
}

DEFINE_MEMBER(int, pack_diagnostic)(const bool query, void *buffer)
{
  size_t   size = shape[2] * shape[1] * shape[0] * 6;
  float64 *buf  = static_cast<float64 *>(buffer);

  if (query) {
    return sizeof(float64) * size;
  }

  std::vector<size_t> shape = {size};

  auto iz = xt::range(Lbz, Ubz + 1);
  auto iy = xt::range(Lby, Uby + 1);
  auto ix = xt::range(Lbx, Ubx + 1);
  auto uu = xt::view(uf, iz, iy, ix, xt::all());
  auto vv = xt::adapt(buf, size, xt::no_ownership(), shape);

  // packing
  vv = uu;

  return sizeof(float64) * size;
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
