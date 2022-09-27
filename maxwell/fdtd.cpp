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
}

DEFINE_MEMBER(int, pack)(void *buffer, const int address)
{
  int count = address;

  count += Chunk::pack(buffer, address);
  count += memcpy_count(buffer, uf.data(), uf.size() * sizeof(float64), count, 0);
  count += memcpy_count(buffer, &cc, sizeof(float64), count, 0);

  return count;
}

DEFINE_MEMBER(int, unpack)(void *buffer, const int address)
{
  int count = address;

  count += Chunk::unpack(buffer, address);
  count += memcpy_count(uf.data(), buffer, uf.size() * sizeof(float64), 0, count);
  count += memcpy_count(&cc, buffer, sizeof(float64), 0, count);

  return count;
}

DEFINE_MEMBER(void, setup)(json &config)
{
  float64 delh = config["delh"].get<float64>();

  cc   = config["cc"].get<float64>();
  delx = delh;
  dely = delh;
  delz = delh;

  int     kdir    = config["kdir"].get<int>();
  int     efd[3]  = {0};
  int     bfd[3]  = {0};
  float64 kvec[3] = {0};

  switch (kdir) {
  case 0: {
    // propagation in z dir
    kvec[0] = 0;
    kvec[1] = 0;
    kvec[2] = math::pi2 / zlim[2];
    efd[0]  = 0;
    efd[1]  = 1;
    efd[2]  = 2;
    bfd[0]  = 4;
    bfd[1]  = 3;
    bfd[2]  = 5;
  } break;
  case 1: {
    // propagation in y dir
    kvec[0] = 0;
    kvec[1] = math::pi2 / ylim[2];
    kvec[2] = 0;
    efd[0]  = 2;
    efd[1]  = 0;
    efd[2]  = 1;
    bfd[0]  = 3;
    bfd[1]  = 5;
    bfd[2]  = 4;
  } break;
  case 2: {
    // propagation in x dir
    kvec[0] = math::pi2 / xlim[2];
    kvec[1] = 0;
    kvec[2] = 0;
    efd[0]  = 1;
    efd[1]  = 2;
    efd[2]  = 0;
    bfd[0]  = 5;
    bfd[1]  = 4;
    bfd[2]  = 3;
  } break;
  default:
    break;
  }

  // set initial condition
  for (int iz = Lbz; iz <= Ubz; iz++) {
    for (int iy = Lby; iy <= Uby; iy++) {
      for (int ix = Lbx; ix <= Ubx; ix++) {
        float64 ff = cos(kvec[0] * xc(ix) + kvec[1] * yc(iy) + kvec[2] * zc(iz));
        float64 gg = cos(kvec[0] * xc(ix) + kvec[1] * yc(iy) + kvec[2] * zc(iz));

        uf(iz, iy, ix, efd[0]) = ff;
        uf(iz, iy, ix, efd[1]) = gg;
        uf(iz, iy, ix, efd[2]) = 0;
        uf(iz, iy, ix, bfd[0]) = ff;
        uf(iz, iy, ix, bfd[1]) = gg;
        uf(iz, iy, ix, bfd[2]) = 0;
      }
    }
  }

  // initialize MPI buffer (one buffer object)
  mpibufvec.resize(1);
  mpibufvec[0] = std::make_unique<MpiBuffer>();
  set_mpi_buffer(mpibufvec[0], 0, sizeof(float64) * 6);
}

DEFINE_MEMBER(void, push)(const float64 delt)
{
  const float64 cflx = cc * delt / delx;
  const float64 cfly = cc * delt / dely;
  const float64 cflz = cc * delt / delz;

  float64 tzero = wall_clock();

  // advance E-field
  for (int iz = Lbz - 1; iz <= Ubz; iz++) {
    for (int iy = Lby - 1; iy <= Uby; iy++) {
      for (int ix = Lbx - 1; ix <= Ubx; ix++) {
        uf(iz, iy, ix, 0) += (+cfly) * (uf(iz, iy + 1, ix, 5) - uf(iz, iy, ix, 5)) +
                             (-cflz) * (uf(iz + 1, iy, ix, 4) - uf(iz, iy, ix, 4));
        uf(iz, iy, ix, 1) += (+cflz) * (uf(iz + 1, iy, ix, 3) - uf(iz, iy, ix, 3)) +
                             (-cflx) * (uf(iz, iy, ix + 1, 5) - uf(iz, iy, ix, 5));
        uf(iz, iy, ix, 2) += (+cflx) * (uf(iz, iy, ix + 1, 4) - uf(iz, iy, ix, 4)) +
                             (-cfly) * (uf(iz, iy + 1, ix, 3) - uf(iz, iy, ix, 3));
      }
    }
  }

  // advance B-field
  for (int iz = Lbz; iz <= Ubz; iz++) {
    for (int iy = Lby; iy <= Uby; iy++) {
      for (int ix = Lbx; ix <= Ubx; ix++) {
        uf(iz, iy, ix, 3) += (-cfly) * (uf(iz, iy, ix, 2) - uf(iz, iy - 1, ix, 2)) +
                             (+cflz) * (uf(iz, iy, ix, 1) - uf(iz - 1, iy, ix, 1));
        uf(iz, iy, ix, 4) += (-cflz) * (uf(iz, iy, ix, 0) - uf(iz - 1, iy, ix, 0)) +
                             (+cflx) * (uf(iz, iy, ix, 2) - uf(iz, iy, ix - 1, 2));
        uf(iz, iy, ix, 5) += (-cflx) * (uf(iz, iy, ix, 1) - uf(iz, iy, ix - 1, 1)) +
                             (+cfly) * (uf(iz, iy, ix, 0) - uf(iz, iy - 1, ix, 0));
      }
    }
  }

  // store computation time
  load[0] += wall_clock() - tzero;
}

DEFINE_MEMBER(int, pack_diagnostic)(int mode, void *buffer, const int address)
{
  return this->pack_diagnostic_field(buffer, address, uf);
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
