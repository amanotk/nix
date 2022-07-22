// -*- C++ -*-
#include "fdtd.hpp"

#define DEFINE_MEMBER(type, name) type FDTD::name

DEFINE_MEMBER(, FDTD)(const int dims[3], const int id) : BaseChunk<3>(dims, id)
{
  size_t Nz = this->dims[0] + 2 * Nb;
  size_t Ny = this->dims[1] + 2 * Nb;
  size_t Nx = this->dims[2] + 2 * Nb;

  // lower and upper bound
  Lbz = Nb;
  Ubz = this->dims[0] + Nb - 1;
  Lby = Nb;
  Uby = this->dims[1] + Nb - 1;
  Lbx = Nb;
  Ubx = this->dims[2] + Nb - 1;

  // memory allocation
  zc.resize({Nz});
  yc.resize({Ny});
  xc.resize({Nx});
  uf.resize({Nz, Ny, Nx, 6});
  zc.fill(0);
  yc.fill(0);
  xc.fill(0);
  uf.fill(0);

  set_buffer_position();
  sendbuf.resize(bufsize);
  recvbuf.resize(bufsize);
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
  zlim[1] = offset[0] * delh + dims[0] * delh;
  zlim[2] = zlim[1] - zlim[0];
  ylim[0] = offset[1] * delh;
  ylim[1] = offset[1] * delh + dims[1] * delh;
  ylim[2] = ylim[1] - ylim[0];
  xlim[0] = offset[2] * delh;
  xlim[1] = offset[2] * delh + dims[2] * delh;
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

DEFINE_MEMBER(void, set_buffer_position) ()
{
  const std::vector<size_t> shape = {3, 3};

  auto I = xt::all();
  auto J = xt::newaxis();

  {
    //
    // lower/upper bounds for MPI send
    //
    // * z direction
    sendlb[0][0] = Lbz;
    sendlb[0][1] = Lbz;
    sendlb[0][2] = Ubz - Nb + 1;
    sendub[0][0] = Lbz + Nb - 1;
    sendub[0][1] = Ubz;
    sendub[0][2] = Ubz;
    // * y direction
    sendlb[1][0] = Lby;
    sendlb[1][1] = Lby;
    sendlb[1][2] = Uby - Nb + 1;
    sendub[1][0] = Lby + Nb - 1;
    sendub[1][1] = Uby;
    sendub[1][2] = Uby;
    // * x direction
    sendlb[2][0] = Lbx;
    sendlb[2][1] = Lbx;
    sendlb[2][2] = Ubx - Nb + 1;
    sendub[2][0] = Lbx + Nb - 1;
    sendub[2][1] = Ubx;
    sendub[2][2] = Ubx;

    //
    // lower/upper bounds for MPI recv
    //
    // * z direction
    recvlb[0][0] = Lbz - Nb;
    recvlb[0][1] = Lbz;
    recvlb[0][2] = Ubz + 1;
    recvub[0][0] = Lbz - 1;
    recvub[0][1] = Ubz;
    recvub[0][2] = Ubz + Nb;
    // * y direction
    recvlb[1][0] = Lby - Nb;
    recvlb[1][1] = Lby;
    recvlb[1][2] = Uby + 1;
    recvub[1][0] = Lby - 1;
    recvub[1][1] = Uby;
    recvub[1][2] = Uby + Nb;
    // * x direction
    recvlb[2][0] = Lbx - Nb;
    recvlb[2][1] = Lbx;
    recvlb[2][2] = Ubx + 1;
    recvub[2][0] = Lbx - 1;
    recvub[2][1] = Ubx;
    recvub[2][2] = Ubx + Nb;
  }

  auto xlb = xt::adapt(&recvlb[0][0], 9, xt::no_ownership(), shape);
  auto xub = xt::adapt(&recvub[0][0], 9, xt::no_ownership(), shape);
  auto xss = xub - xlb + 1;
  auto pos =
      xt::eval(xt::view(xss, 0, I, J, J) * xt::view(xss, 1, J, I, J) * xt::view(xss, 2, J, J, I));

  // no send/recv with itself
  pos(1, 1, 1) = 0;

  // calculate buffer positions and size
  pos     = xt::cumsum(pos);
  pos     = xt::roll(pos, 1);
  bufsize = pos(0) * sizeof(float64) * 6;
  pos(0)  = 0;

  // reshpe and store
  pos.reshape({3, 3, 3});
  bufaddr = pos;
}

DEFINE_MEMBER(void, set_boundary_begin)()
{
  auto Ia = xt::all();

  // physical boundary
  set_boundary_physical(0);
  set_boundary_physical(1);
  set_boundary_physical(2);

  for (int dirz = -1, iz=0; dirz <= +1; dirz++, iz++) {
    for (int diry = -1, iy=0; diry <= +1; diry++, iy++) {
      for (int dirx = -1, ix=0; dirx <= +1; dirx++, ix++) {
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
        void *sndpos = sendbuf.get(bufaddr(iz, iy, ix) * dsize);
        void *rcvpos = recvbuf.get(bufaddr(iz, iy, ix) * dsize);

        // pack
        std::copy(view.begin(), view.end(), static_cast<float64 *>(sndpos));

        // send/recv calls
        MPI_Isend(sndpos, byte, MPI_BYTE, nbrank, sndtag, MPI_COMM_WORLD, &sendreq[iz][iy][ix]);
        MPI_Irecv(rcvpos, byte, MPI_BYTE, nbrank, rcvtag, MPI_COMM_WORLD, &recvreq[iz][iy][ix]);
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
  for (int dirz = -1, iz=0; dirz <= +1; dirz++, iz++) {
    for (int diry = -1, iy=0; diry <= +1; diry++, iy++) {
      for (int dirx = -1, ix=0; dirx <= +1; dirx++, ix++) {
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
        void *   rcvpos = recvbuf.get(bufaddr(iz, iy, ix) * dsize);
        float64 *ptr    = static_cast<float64 *>(rcvpos);
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
    MPI_Waitall(27, &recvreq[0][0][0], MPI_STATUS_IGNORE);
    break;
  case -1: // send
    MPI_Waitall(27, &sendreq[0][0][0], MPI_STATUS_IGNORE);
    break;
  case 0: // both send and receive
    MPI_Waitall(27, &sendreq[0][0][0], MPI_STATUS_IGNORE);
    MPI_Waitall(27, &recvreq[0][0][0], MPI_STATUS_IGNORE);
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
    ERRORPRINT("Non-periodic boundary condition has not been implemented!\n");
  }

  // upper boundary in z
  if (get_nb_rank(+1, 0, 0) == MPI_PROC_NULL) {
    ERRORPRINT("Non-periodic boundary condition has not been implemented!\n");
  }

  // lower boundary in y
  if (get_nb_rank(0, -1, 0) == MPI_PROC_NULL) {
    ERRORPRINT("Non-periodic boundary condition has not been implemented!\n");
  }

  // upper boundary in y
  if (get_nb_rank(0, +1, 0) == MPI_PROC_NULL) {
    ERRORPRINT("Non-periodic boundary condition has not been implemented!\n");
  }

  // lower boundary in x
  if (get_nb_rank(0, 0, -1) == MPI_PROC_NULL) {
    ERRORPRINT("Non-periodic boundary condition has not been implemented!\n");
  }

  // upper boundary in x
  if (get_nb_rank(0, 0, +1) == MPI_PROC_NULL) {
    ERRORPRINT("Non-periodic boundary condition has not been implemented!\n");
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
