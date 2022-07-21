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

  bufsize[0] = 6 * Nx * Ny * Nb * sizeof(float64);
  bufsize[1] = 6 * Nz * Nx * Nb * sizeof(float64);
  bufsize[2] = 6 * Ny * Nz * Nb * sizeof(float64);
  bufsize[3] = bufsize[0] + bufsize[1] + bufsize[2];
  sendbuf.resize(2 * bufsize[3]);
  recvbuf.resize(2 * bufsize[3]);
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

  // set boundary condition
  set_boundary();

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

DEFINE_MEMBER(int, pack_diagnostic)(void *buffer, const bool query)
{
  size_t   size = shape[2] * shape[1] * shape[0] * 6;
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
  const size_t Sz = bufsize[0];
  const size_t Sy = bufsize[1];
  const size_t Sx = bufsize[2];

  auto Ia = xt::all();
  auto Iz = xt::range(Lbz, Ubz + 1);
  auto Iy = xt::range(Lby, Uby + 1);
  auto Ix = xt::range(Lbx, Ubx + 1);

  // calculate buffer positions
  xt::xarray<size_t> bufpos = {Sz, Sz, Sy, Sy, Sx, Sx};

  bufpos = xt::cumsum(bufpos) - Sz;
  bufpos.reshape({3, 2});

  // physical boundary
  set_boundary_physical(0);
  set_boundary_physical(1);
  set_boundary_physical(2);

  //
  // issue send/recv calls in z direction
  //
  {
    int   nbrank[2] = {get_nb_rank(-1, 0, 0), get_nb_rank(+1, 0, 0)};
    int   sndtag[2] = {get_sndtag(-1, 0, 0), get_sndtag(+1, 0, 0)};
    int   rcvtag[2] = {get_rcvtag(-1, 0, 0), get_rcvtag(+1, 0, 0)};
    void *sndpos[2] = {sendbuf.get(bufpos(0, 0)), sendbuf.get(bufpos(0, 1))};
    void *rcvpos[2] = {recvbuf.get(bufpos(0, 0)), recvbuf.get(bufpos(0, 1))};

    // lower bound
    {
      auto     view   = xt::view(uf, Lbz, Iy, Ix, Ia);
      float64 *buffer = static_cast<float64 *>(sndpos[0]);
      int      byte   = view.size() * sizeof(float64);
      std::copy(view.begin(), view.end(), buffer);
      MPI_Isend(sndpos[0], byte, MPI_BYTE, nbrank[0], sndtag[0], MPI_COMM_WORLD, &sendreq[0][0]);
      MPI_Irecv(rcvpos[0], byte, MPI_BYTE, nbrank[0], rcvtag[0], MPI_COMM_WORLD, &recvreq[0][0]);
    }

    // upper bound
    {
      auto     view   = xt::view(uf, Ubz, Iy, Ix, Ia);
      float64 *buffer = static_cast<float64 *>(sndpos[1]);
      int      byte   = view.size() * sizeof(float64);
      std::copy(view.begin(), view.end(), buffer);
      MPI_Isend(sndpos[1], byte, MPI_BYTE, nbrank[1], sndtag[1], MPI_COMM_WORLD, &sendreq[0][1]);
      MPI_Irecv(rcvpos[1], byte, MPI_BYTE, nbrank[1], rcvtag[1], MPI_COMM_WORLD, &recvreq[0][1]);
    }
  }

  //
  // issue send/recv calls in y direction
  //
  {
    int   nbrank[2] = {get_nb_rank(0, -1, 0), get_nb_rank(0, +1, 0)};
    int   sndtag[2] = {get_sndtag(0, -1, 0), get_sndtag(0, +1, 0)};
    int   rcvtag[2] = {get_rcvtag(0, -1, 0), get_rcvtag(0, +1, 0)};
    void *sndpos[2] = {sendbuf.get(bufpos(1, 0)), sendbuf.get(bufpos(1, 1))};
    void *rcvpos[2] = {recvbuf.get(bufpos(1, 0)), recvbuf.get(bufpos(1, 1))};

    // lower bound
    {
      auto     view   = xt::view(uf, Iz, Lby, Ix, Ia);
      float64 *buffer = static_cast<float64 *>(sndpos[0]);
      int      byte   = view.size() * sizeof(float64);
      std::copy(view.begin(), view.end(), buffer);
      MPI_Isend(sndpos[0], byte, MPI_BYTE, nbrank[0], sndtag[0], MPI_COMM_WORLD, &sendreq[1][0]);
      MPI_Irecv(rcvpos[0], byte, MPI_BYTE, nbrank[0], rcvtag[0], MPI_COMM_WORLD, &recvreq[1][0]);
    }

    // upper bound
    {
      auto     view   = xt::view(uf, Iz, Uby, Ix, Ia);
      float64 *buffer = static_cast<float64 *>(sndpos[1]);
      int      byte   = view.size() * sizeof(float64);
      std::copy(view.begin(), view.end(), buffer);
      MPI_Isend(sndpos[1], byte, MPI_BYTE, nbrank[1], sndtag[1], MPI_COMM_WORLD, &sendreq[1][1]);
      MPI_Irecv(rcvpos[1], byte, MPI_BYTE, nbrank[1], rcvtag[1], MPI_COMM_WORLD, &recvreq[1][1]);
    }
  }

  //
  // issue send/recv calls in x direction
  //
  {
    int   nbrank[2] = {get_nb_rank(0, 0, -1), get_nb_rank(0, 0, +1)};
    int   sndtag[2] = {get_sndtag(0, 0, -1), get_sndtag(0, 0, +1)};
    int   rcvtag[2] = {get_rcvtag(0, 0, -1), get_rcvtag(0, 0, +1)};
    void *sndpos[2] = {sendbuf.get(bufpos(2, 0)), sendbuf.get(bufpos(2, 1))};
    void *rcvpos[2] = {recvbuf.get(bufpos(2, 0)), recvbuf.get(bufpos(2, 1))};

    // lower bound
    {
      auto     view   = xt::view(uf, Iz, Iy, Lbx, Ia);
      float64 *buffer = static_cast<float64 *>(sndpos[0]);
      int      byte   = view.size() * sizeof(float64);
      std::copy(view.begin(), view.end(), buffer);
      MPI_Isend(sndpos[0], byte, MPI_BYTE, nbrank[0], sndtag[0], MPI_COMM_WORLD, &sendreq[2][0]);
      MPI_Irecv(rcvpos[0], byte, MPI_BYTE, nbrank[0], rcvtag[0], MPI_COMM_WORLD, &recvreq[2][0]);
    }

    // upper bound
    {
      auto     view   = xt::view(uf, Iz, Iy, Ubx, Ia);
      float64 *buffer = static_cast<float64 *>(sndpos[1]);
      int      byte   = view.size() * sizeof(float64);
      std::copy(view.begin(), view.end(), buffer);
      MPI_Isend(sndpos[1], byte, MPI_BYTE, nbrank[1], sndtag[1], MPI_COMM_WORLD, &sendreq[2][1]);
      MPI_Irecv(rcvpos[1], byte, MPI_BYTE, nbrank[1], rcvtag[1], MPI_COMM_WORLD, &recvreq[2][1]);
    }
  }
}

DEFINE_MEMBER(void, set_boundary_end)()
{
  const size_t Sz = bufsize[0];
  const size_t Sy = bufsize[1];
  const size_t Sx = bufsize[2];

  auto Ia = xt::all();
  auto Iz = xt::range(Lbz, Ubz + 1);
  auto Iy = xt::range(Lby, Uby + 1);
  auto Ix = xt::range(Lbx, Ubx + 1);

  // calculate buffer positions
  xt::xarray<size_t> bufpos = {Sz, Sz, Sy, Sy, Sx, Sx};

  bufpos = xt::cumsum(bufpos) - Sz;
  bufpos.reshape({3, 2});

  //
  // unpack recv buffer in z direction
  //
  {
    void *rcvpos[2] = {recvbuf.get(bufpos(0, 0)), recvbuf.get(bufpos(0, 1))};

    // wait for receive
    MPI_Waitall(2, &recvreq[0][0], MPI_STATUS_IGNORE);

    // lower bound
    if (get_nb_rank(-1, 0, 0) != MPI_PROC_NULL) {
      auto     view   = xt::view(uf, Lbz - 1, Iy, Ix, Ia);
      float64 *buffer = static_cast<float64 *>(rcvpos[0]);
      std::copy(buffer, buffer + view.size(), view.begin());
    }

    // upper bound
    if (get_nb_rank(+1, 0, 0) != MPI_PROC_NULL) {
      auto     view   = xt::view(uf, Ubz + 1, Iy, Ix, Ia);
      float64 *buffer = static_cast<float64 *>(rcvpos[1]);
      std::copy(buffer, buffer + view.size(), view.begin());
    }

    // wait for send
    MPI_Waitall(2, &sendreq[0][0], MPI_STATUS_IGNORE);
  }

  //
  // unpack recv buffer in y direction
  //
  {
    void *rcvpos[2] = {recvbuf.get(bufpos(1, 0)), recvbuf.get(bufpos(1, 1))};

    // wait for receive
    MPI_Waitall(2, &recvreq[1][0], MPI_STATUS_IGNORE);

    // lower bound
    if (get_nb_rank(0, -1, 0) != MPI_PROC_NULL) {
      auto     view   = xt::view(uf, Iz, Lby - 1, Ix, Ia);
      float64 *buffer = static_cast<float64 *>(rcvpos[0]);
      std::copy(buffer, buffer + view.size(), view.begin());
    }

    // upper bound
    if (get_nb_rank(0, +1, 0) != MPI_PROC_NULL) {
      auto     view   = xt::view(uf, Iz, Uby + 1, Ix, Ia);
      float64 *buffer = static_cast<float64 *>(rcvpos[1]);
      std::copy(buffer, buffer + view.size(), view.begin());
    }

    // wait for send
    MPI_Waitall(2, &sendreq[1][0], MPI_STATUS_IGNORE);
  }

  //
  // unpack recv buffer in x direction
  //
  {
    void *rcvpos[2] = {recvbuf.get(bufpos(2, 0)), recvbuf.get(bufpos(2, 1))};

    // wait for receive
    MPI_Waitall(2, &recvreq[1][0], MPI_STATUS_IGNORE);

    // lower bound
    if (get_nb_rank(0, 0, -1) != MPI_PROC_NULL) {
      auto     view   = xt::view(uf, Iz, Iy, Lbx - 1, Ia);
      float64 *buffer = static_cast<float64 *>(rcvpos[0]);
      std::copy(buffer, buffer + view.size(), view.begin());
    }

    // upper bound
    if (get_nb_rank(0, 0, +1) != MPI_PROC_NULL) {
      auto     view   = xt::view(uf, Iz, Iy, Ubx + 1, Ia);
      float64 *buffer = static_cast<float64 *>(rcvpos[1]);
      std::copy(buffer, buffer + view.size(), view.begin());
    }

    // wait for send
    MPI_Waitall(2, &sendreq[1][0], MPI_STATUS_IGNORE);
  }
}

DEFINE_MEMBER(bool, set_boundary_query)(const int mode)
{
  int flag = 0;

  switch (mode) {
  case +1: // receive
    MPI_Testall(6, &recvreq[0][0], &flag, MPI_STATUS_IGNORE);
    break;
  case -1: // send
    MPI_Testall(6, &sendreq[0][0], &flag, MPI_STATUS_IGNORE);
    break;
  case 0: // both send and receive
    MPI_Testall(6, &recvreq[0][0], &flag, MPI_STATUS_IGNORE);
    MPI_Testall(6, &sendreq[0][0], &flag, MPI_STATUS_IGNORE);
    break;
  deafult:
    ERRORPRINT("No such mode is available");
  }

  return !(flag == 0);
}

DEFINE_MEMBER(void, set_boundary_physical)(const int dir)
{
  // lower boundary in z
  if (get_nb_rank(-1, 0, 0) == MPI_PROC_NULL) {
    ERRORPRINT("Non-periodic boundary condition has not been implemented!");
  }

  // upper boundary in z
  if (get_nb_rank(+1, 0, 0) == MPI_PROC_NULL) {
    ERRORPRINT("Non-periodic boundary condition has not been implemented!");
  }

  // lower boundary in y
  if (get_nb_rank(0, -1, 0) == MPI_PROC_NULL) {
    ERRORPRINT("Non-periodic boundary condition has not been implemented!");
  }

  // upper boundary in y
  if (get_nb_rank(0, +1, 0) == MPI_PROC_NULL) {
    ERRORPRINT("Non-periodic boundary condition has not been implemented!");
  }

  // lower boundary in x
  if (get_nb_rank(0, 0, -1) == MPI_PROC_NULL) {
    ERRORPRINT("Non-periodic boundary condition has not been implemented!");
  }

  // upper boundary in x
  if (get_nb_rank(0, 0, +1) == MPI_PROC_NULL) {
    ERRORPRINT("Non-periodic boundary condition has not been implemented!");
  }
}

DEFINE_MEMBER(void, set_boundary)()
{
  set_boundary_begin();
  set_boundary_end();
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
