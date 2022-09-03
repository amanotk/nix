// -*- C++ -*-
#include "fdtd.hpp"

#define DEFINE_MEMBER(type, name) type FDTD::name

const int FDTD::Nb;

DEFINE_MEMBER(, FDTD)(const int dims[3], const int id) : Chunk(dims, id)
{
  size_t Nz = this->dims[0] + 2 * Nb;
  size_t Ny = this->dims[1] + 2 * Nb;
  size_t Nx = this->dims[2] + 2 * Nb;

  // memory allocation
  uf.resize({Nz, Ny, Nx, 6});
  uf.fill(0);

  // initialize MPI buffer
  mpibufvec.push_back(std::make_unique<MpiBuffer>());
  set_mpi_buffer(mpibufvec[0], 0, sizeof(float64) * 6);
}

DEFINE_MEMBER(, ~FDTD)()
{
  uf.resize({0});
}

DEFINE_MEMBER(int, pack)(const int mode, void *buffer)
{
  using common::memcpy_count;

  int   count = 0;
  char *ptr   = static_cast<char *>(buffer);

  switch (mode) {
  case PackAllQuery:
    count += Chunk::pack(Chunk::PackAllQuery, &ptr[count]);
    count += memcpy_count(&ptr[count], uf.data(), uf.size() * sizeof(float64), true);
    count += memcpy_count(&ptr[count], &cc, sizeof(float64), true);
    break;
  case PackAll:
    count += Chunk::pack(Chunk::PackAll, &ptr[count]);
    count += memcpy_count(&ptr[count], uf.data(), uf.size() * sizeof(float64), false);
    count += memcpy_count(&ptr[count], &cc, sizeof(float64), false);
    break;
  case PackEmf:
    count = pack_diagnostic(buffer, false);
    break;
  case PackEmfQuery:
    count = pack_diagnostic(buffer, true);
    break;
  default:
    ERRORPRINT("No such packing mode");
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
    count += Chunk::unpack(Chunk::PackAll, &ptr[count]);
    count += memcpy_count(uf.data(), &ptr[count], uf.size() * sizeof(float64), false);
    count += memcpy_count(&cc, &ptr[count], sizeof(float64), false);
    break;
  case PackEmf:
    ERRORPRINT("Not implemented yet");
    break;
  case PackEmfQuery:
    ERRORPRINT("Not implemented yet");
    break;
  default:
    ERRORPRINT("No such unpacking mode");
    break;
  }

  return count;
}

DEFINE_MEMBER(void, setup)
(const float64 cc, const float64 delh, const int *offset, const int *ndims, T_function initializer)
{
  this->cc   = cc;
  this->delh = delh;

  set_global_context(offset, ndims);

  // set initial condition
  for (int iz = Lbz; iz <= Ubz; iz++) {
    for (int iy = Lby; iy <= Uby; iy++) {
      for (int ix = Lbx; ix <= Ubx; ix++) {
        initializer(zc(iz), yc(iy), xc(ix), &uf(iz, iy, ix, 0));
      }
    }
  }
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

  // store computation time
  load += common::etime() - etime;
}

DEFINE_MEMBER(int, pack_diagnostic)(void *buffer, const bool query)
{
  size_t   size = dims[2] * dims[1] * dims[0] * 6;
  float64 *buf  = static_cast<float64 *>(buffer);

  if (query) {
    return sizeof(float64) * size;
  }

  auto Iz = xt::range(Lbz, Ubz + 1);
  auto Iy = xt::range(Lby, Uby + 1);
  auto Ix = xt::range(Lbx, Ubx + 1);
  auto uu = xt::view(uf, Iz, Iy, Ix, xt::all());

  // packing
  std::copy(uu.begin(), uu.end(), buf);

  return sizeof(float64) * size;
}

DEFINE_MEMBER(void, set_boundary_begin)(const int mode)
{
  auto Ia = xt::all();

  // physical boundary
  set_boundary_physical();

  begin_bc_exchange(mpibufvec[0], uf);
}

DEFINE_MEMBER(void, set_boundary_end)(const int mode)
{
  auto Ia = xt::all();

  end_bc_exchange(mpibufvec[0], uf, false);
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
