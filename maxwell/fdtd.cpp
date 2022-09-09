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

DEFINE_MEMBER(int, pack)(void *buffer, const int address)
{
  using common::memcpy_count;

  int count = address;

  count += Chunk::pack(buffer, address);
  count += memcpy_count(buffer, uf.data(), uf.size() * sizeof(float64), count, 0);
  count += memcpy_count(buffer, &cc, sizeof(float64), count, 0);

  return count;
}

DEFINE_MEMBER(int, unpack)(void *buffer, const int address)
{
  using common::memcpy_count;

  int count = address;

  count += Chunk::unpack(buffer, address);
  count += memcpy_count(uf.data(), buffer, uf.size() * sizeof(float64), 0, count);
  count += memcpy_count(&cc, buffer, sizeof(float64), 0, count);

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

DEFINE_MEMBER(int, pack_diagnostic)(void *buffer, const int address)
{
  size_t size = dims[2] * dims[1] * dims[0] * 6;

  if (buffer == nullptr) {
    return sizeof(float64) * size;
  }

  auto Iz = xt::range(Lbz, Ubz + 1);
  auto Iy = xt::range(Lby, Uby + 1);
  auto Ix = xt::range(Lbx, Ubx + 1);
  auto uu = xt::view(uf, Iz, Iy, Ix, xt::all());

  // packing
  char *ptr = &static_cast<char *>(buffer)[address];
  std::copy(uu.begin(), uu.end(), ptr);

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
