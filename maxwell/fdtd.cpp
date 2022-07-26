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
  case PackAll:
    count += Chunk::pack(Chunk::PackAll, &ptr[count]);
    count += memcpy_count(&ptr[count], uf.data(), uf.size() * sizeof(float64), false);
    count += memcpy_count(&ptr[count], &cc, sizeof(float64), false);
    break;
  case PackAllQuery:
    count += Chunk::pack(Chunk::PackAllQuery, &ptr[count]);
    count += memcpy_count(&ptr[count], uf.data(), uf.size() * sizeof(float64), true);
    count += memcpy_count(&ptr[count], &cc, sizeof(float64), true);
    break;
  case PackEmf:
    count = pack_diagnostic(buffer, false);
    break;
  case PackEmfQuery:
    count = pack_diagnostic(buffer, true);
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
    count += Chunk::unpack(Chunk::PackAll, &ptr[count]);
    count += memcpy_count(uf.data(), &ptr[count], uf.size() * sizeof(float64), false);
    count += memcpy_count(&cc, &ptr[count], sizeof(float64), false);
    break;
  case PackAllQuery:
    count += Chunk::unpack(Chunk::PackAllQuery, &ptr[count]);
    count += memcpy_count(uf.data(), &ptr[count], uf.size() * sizeof(float64), true);
    count += memcpy_count(&cc, &ptr[count], sizeof(float64), true);
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
  set_coordinate(delh, offset);

  // speed of light
  this->cc = cc;

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

DEFINE_MEMBER(void, set_boundary_begin)()
{
  auto Ia = xt::all();

  // physical boundary
  set_boundary_physical(0);
  set_boundary_physical(1);
  set_boundary_physical(2);

  for (int dirz = -1, iz = 0; dirz <= +1; dirz++, iz++) {
    for (int diry = -1, iy = 0; diry <= +1; diry++, iy++) {
      for (int dirx = -1, ix = 0; dirx <= +1; dirx++, ix++) {
        // skip send/recv to itself
        if (dirz == 0 && diry == 0 && dirx == 0) {
          sendreq[iz][iy][ix] = MPI_REQUEST_NULL;
          recvreq[iz][iy][ix] = MPI_REQUEST_NULL;
          continue;
        }

        // index range
        auto Iz = xt::range(sendlb[0][iz], sendub[0][iz] + 1);
        auto Iy = xt::range(sendlb[1][iy], sendub[1][iy] + 1);
        auto Ix = xt::range(sendlb[2][ix], sendub[2][ix] + 1);

        // MPI
        auto  view   = xt::view(uf, Iz, Iy, Ix, Ia);
        int   byte   = view.size() * sizeof(float64);
        int   nbrank = get_nb_rank(dirz, diry, dirx);
        int   sndtag = get_sndtag(dirz, diry, dirx);
        int   rcvtag = get_rcvtag(dirz, diry, dirx);
        int   dsize  = sizeof(float64) * 6;
        void *sndptr = sendbuf.get(bufaddr(iz, iy, ix) * dsize);
        void *rcvptr = recvbuf.get(bufaddr(iz, iy, ix) * dsize);

        // pack
        std::copy(view.begin(), view.end(), static_cast<float64 *>(sndptr));

        // send/recv calls
        MPI_Isend(sndptr, byte, MPI_BYTE, nbrank, sndtag, MPI_COMM_WORLD, &sendreq[iz][iy][ix]);
        MPI_Irecv(rcvptr, byte, MPI_BYTE, nbrank, rcvtag, MPI_COMM_WORLD, &recvreq[iz][iy][ix]);
      }
    }
  }
}

DEFINE_MEMBER(void, set_boundary_end)()
{
  auto Ia = xt::all();

  //
  // wait for MPI calls to complete
  //
  MPI_Waitall(27, &sendreq[0][0][0], MPI_STATUS_IGNORE);
  MPI_Waitall(27, &recvreq[0][0][0], MPI_STATUS_IGNORE);

  //
  // unpack recv buffer
  //
  for (int dirz = -1, iz = 0; dirz <= +1; dirz++, iz++) {
    for (int diry = -1, iy = 0; diry <= +1; diry++, iy++) {
      for (int dirx = -1, ix = 0; dirx <= +1; dirx++, ix++) {
        // skip send/recv to itself
        if (dirz == 0 && diry == 0 && dirx == 0) {
          continue;
        }

        // skip physical boundary
        if (get_nb_rank(dirz, diry, dirx) == MPI_PROC_NULL) {
          continue;
        }

        // index range
        auto Iz = xt::range(recvlb[0][iz], recvub[0][iz] + 1);
        auto Iy = xt::range(recvlb[1][iy], recvub[1][iy] + 1);
        auto Ix = xt::range(recvlb[2][ix], recvub[2][ix] + 1);

        // unpack
        auto     view   = xt::view(uf, Iz, Iy, Ix, Ia);
        int      dsize  = sizeof(float64) * 6;
        void *   rcvptr = recvbuf.get(bufaddr(iz, iy, ix) * dsize);
        float64 *ptr    = static_cast<float64 *>(rcvptr);
        std::copy(ptr, ptr + view.size(), view.begin());
      }
    }
  }
}

DEFINE_MEMBER(bool, set_boundary_query)(const int mode)
{
  int flag = 0;

  switch (mode) {
  case +1: // receive
    MPI_Testall(27, &recvreq[0][0][0], &flag, MPI_STATUS_IGNORE);
    break;
  case -1: // send
    MPI_Testall(27, &sendreq[0][0][0], &flag, MPI_STATUS_IGNORE);
    break;
  case 0: // both send and receive
    MPI_Testall(27, &sendreq[0][0][0], &flag, MPI_STATUS_IGNORE);
    MPI_Testall(27, &recvreq[0][0][0], &flag, MPI_STATUS_IGNORE);
    break;
  deafult:
    ERRORPRINT("No such mode is available");
  }

  return !(flag == 0);
}

DEFINE_MEMBER(void, set_boundary_physical)(const int dir)
{
  switch (dir) {
  case 0: // z direction
    // lower boundary in z
    if (get_nb_rank(-1, 0, 0) == MPI_PROC_NULL) {
      ERRORPRINT("Non-periodic boundary condition has not been implemented!\n");
    }

    // upper boundary in z
    if (get_nb_rank(+1, 0, 0) == MPI_PROC_NULL) {
      ERRORPRINT("Non-periodic boundary condition has not been implemented!\n");
    }
    break;

  case 1: // y direction
    // lower boundary in y
    if (get_nb_rank(0, -1, 0) == MPI_PROC_NULL) {
      ERRORPRINT("Non-periodic boundary condition has not been implemented!\n");
    }

    // upper boundary in y
    if (get_nb_rank(0, +1, 0) == MPI_PROC_NULL) {
      ERRORPRINT("Non-periodic boundary condition has not been implemented!\n");
    }
    break;

  case 2: // x direction
    // lower boundary in x
    if (get_nb_rank(0, 0, -1) == MPI_PROC_NULL) {
      ERRORPRINT("Non-periodic boundary condition has not been implemented!\n");
    }

    // upper boundary in x
    if (get_nb_rank(0, 0, +1) == MPI_PROC_NULL) {
      ERRORPRINT("Non-periodic boundary condition has not been implemented!\n");
    }
    break;
  }
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
