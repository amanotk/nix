// -*- C++ -*-
#include "chunk3d.hpp"

#define DEFINE_MEMBER(type, name)                                                                  \
  template <int Nb>                                                                                \
  type Chunk3D<Nb>::name

DEFINE_MEMBER(, Chunk3D)(const int dims[3], const int id) : Chunk<3>(dims, id)
{
  size_t Nz = this->dims[0] + 2 * Nb;
  size_t Ny = this->dims[1] + 2 * Nb;
  size_t Nx = this->dims[2] + 2 * Nb;

  //
  // lower and upper bound
  //
  Lbz = Nb;
  Ubz = this->dims[0] + Nb - 1;
  Lby = Nb;
  Uby = this->dims[1] + Nb - 1;
  Lbx = Nb;
  Ubx = this->dims[2] + Nb - 1;

  // * z direction for MPI send
  sendlb[0][0] = Lbz;
  sendlb[0][1] = Lbz;
  sendlb[0][2] = Ubz - Nb + 1;
  sendub[0][0] = Lbz + Nb - 1;
  sendub[0][1] = Ubz;
  sendub[0][2] = Ubz;
  // * y direction for MPI send
  sendlb[1][0] = Lby;
  sendlb[1][1] = Lby;
  sendlb[1][2] = Uby - Nb + 1;
  sendub[1][0] = Lby + Nb - 1;
  sendub[1][1] = Uby;
  sendub[1][2] = Uby;
  // * x direction for MPI send
  sendlb[2][0] = Lbx;
  sendlb[2][1] = Lbx;
  sendlb[2][2] = Ubx - Nb + 1;
  sendub[2][0] = Lbx + Nb - 1;
  sendub[2][1] = Ubx;
  sendub[2][2] = Ubx;
  // * z direction for MPI recv
  recvlb[0][0] = Lbz - Nb;
  recvlb[0][1] = Lbz;
  recvlb[0][2] = Ubz + 1;
  recvub[0][0] = Lbz - 1;
  recvub[0][1] = Ubz;
  recvub[0][2] = Ubz + Nb;
  // * y direction for MPI recv
  recvlb[1][0] = Lby - Nb;
  recvlb[1][1] = Lby;
  recvlb[1][2] = Uby + 1;
  recvub[1][0] = Lby - 1;
  recvub[1][1] = Uby;
  recvub[1][2] = Uby + Nb;
  // * x direction for MPI recv
  recvlb[2][0] = Lbx - Nb;
  recvlb[2][1] = Lbx;
  recvlb[2][2] = Ubx + 1;
  recvub[2][0] = Lbx - 1;
  recvub[2][1] = Ubx;
  recvub[2][2] = Ubx + Nb;

  // memory allocation
  zc.resize({Nz});
  yc.resize({Ny});
  xc.resize({Nx});
  zc.fill(0);
  yc.fill(0);
  xc.fill(0);
}

DEFINE_MEMBER(, ~Chunk3D)()
{
  zc.resize({0});
  yc.resize({0});
  xc.resize({0});
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
  case PackAllQuery:
    count += Chunk<3>::pack(Chunk<3>::PackAllQuery, &ptr[count]);
    count += memcpy_count(&ptr[count], xc.data(), xc.size() * sizeof(float64), true);
    count += memcpy_count(&ptr[count], yc.data(), yc.size() * sizeof(float64), true);
    count += memcpy_count(&ptr[count], zc.data(), zc.size() * sizeof(float64), true);
    count += memcpy_count(&ptr[count], &delh, sizeof(float64), true);
    count += memcpy_count(&ptr[count], xlim, 3 * sizeof(float64), true);
    count += memcpy_count(&ptr[count], ylim, 3 * sizeof(float64), true);
    count += memcpy_count(&ptr[count], zlim, 3 * sizeof(float64), true);
    break;
  case PackAll:
    count += Chunk<3>::pack(Chunk<3>::PackAll, &ptr[count]);
    count += memcpy_count(&ptr[count], xc.data(), xc.size() * sizeof(float64), false);
    count += memcpy_count(&ptr[count], yc.data(), yc.size() * sizeof(float64), false);
    count += memcpy_count(&ptr[count], zc.data(), zc.size() * sizeof(float64), false);
    count += memcpy_count(&ptr[count], &delh, sizeof(float64), false);
    count += memcpy_count(&ptr[count], xlim, 3 * sizeof(float64), false);
    count += memcpy_count(&ptr[count], ylim, 3 * sizeof(float64), false);
    count += memcpy_count(&ptr[count], zlim, 3 * sizeof(float64), false);
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
    count += Chunk<3>::unpack(Chunk<3>::PackAll, &ptr[count]);
    count += memcpy_count(xc.data(), &ptr[count], xc.size() * sizeof(float64), false);
    count += memcpy_count(yc.data(), &ptr[count], yc.size() * sizeof(float64), false);
    count += memcpy_count(zc.data(), &ptr[count], zc.size() * sizeof(float64), false);
    count += memcpy_count(&delh, &ptr[count], sizeof(float64), false);
    count += memcpy_count(xlim, &ptr[count], 3 * sizeof(float64), false);
    count += memcpy_count(ylim, &ptr[count], 3 * sizeof(float64), false);
    count += memcpy_count(zlim, &ptr[count], 3 * sizeof(float64), false);
    break;
  default:
    ERRORPRINT("No such unpacking mode");
    break;
  }

  return count;
}

DEFINE_MEMBER(void, set_coordinate)(const float64 delh, const int offset[3])
{
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
  zc = zlim[0] + delh * (xt::arange<float64>(Lbz - Nb, Ubz + Nb + 1) - Lbz + 0.5);
  yc = ylim[0] + delh * (xt::arange<float64>(Lby - Nb, Uby + Nb + 1) - Lby + 0.5);
  xc = xlim[0] + delh * (xt::arange<float64>(Lbx - Nb, Ubx + Nb + 1) - Lbx + 0.5);
}

DEFINE_MEMBER(void, begin_bc_exchange)(MpiBuffer *mpibuf, xt::xtensor<float64, 4> &array)
{
  auto Ia = xt::all();

  for (int dirz = -1, iz = 0; dirz <= +1; dirz++, iz++) {
    for (int diry = -1, iy = 0; diry <= +1; diry++, iy++) {
      for (int dirx = -1, ix = 0; dirx <= +1; dirx++, ix++) {
        // skip send/recv to itself
        if (dirz == 0 && diry == 0 && dirx == 0) {
          mpibuf->sendreq(iz, iy, ix) = MPI_REQUEST_NULL;
          mpibuf->recvreq(iz, iy, ix) = MPI_REQUEST_NULL;
          continue;
        }

        // index range
        auto Iz = xt::range(this->sendlb[0][iz], this->sendub[0][iz] + 1);
        auto Iy = xt::range(this->sendlb[1][iy], this->sendub[1][iy] + 1);
        auto Ix = xt::range(this->sendlb[2][ix], this->sendub[2][ix] + 1);

        // MPI
        auto  view   = xt::view(array, Iz, Iy, Ix, Ia);
        int   byte   = view.size() * sizeof(float64);
        int   nbrank = this->get_nb_rank(dirz, diry, dirx);
        int   sndtag = this->get_sndtag(dirz, diry, dirx);
        int   rcvtag = this->get_rcvtag(dirz, diry, dirx);
        void *sndptr = mpibuf->sendbuf.get(mpibuf->bufaddr(iz, iy, ix));
        void *rcvptr = mpibuf->recvbuf.get(mpibuf->bufaddr(iz, iy, ix));

        // pack
        std::copy(view.begin(), view.end(), static_cast<float64 *>(sndptr));

        // send/recv calls
        MPI_Isend(sndptr, byte, MPI_BYTE, nbrank, sndtag, mpibuf->comm,
                  &mpibuf->sendreq(iz, iy, ix));
        MPI_Irecv(rcvptr, byte, MPI_BYTE, nbrank, rcvtag, mpibuf->comm,
                  &mpibuf->recvreq(iz, iy, ix));
      }
    }
  }
}

DEFINE_MEMBER(void, count_particle)(ParticleList &particle, int *Lbp, int *Ubp, bool reset)
{
  int     stride[3] = {0};
  int     xrange[2] = {0};
  int     yrange[2] = {0};
  int     zrange[2] = {0};
  float64 rdh[3]    = {0};

  if (require_sort) {
    //
    // full sorting
    //
    stride[0] = dims[2] * dims[1];
    stride[1] = dims[2];
    stride[2] = 1;
    zrange[0] = 0;
    zrange[1] = dims[0] - 1;
    yrange[0] = 0;
    yrange[1] = dims[1] - 1;
    xrange[0] = 0;
    xrange[1] = dims[2] - 1;
    rdh[0]    = 1 / delh;
    rdh[1]    = 1 / delh;
    rdh[2]    = 1 / delh;
  } else {
    //
    // no sorting (assume only a single cell in the chunk)
    //
    stride[0] = 1;
    stride[1] = 1;
    stride[2] = 1;
    zrange[0] = 0;
    zrange[1] = 0;
    yrange[0] = 0;
    yrange[1] = 0;
    xrange[0] = 0;
    xrange[1] = 0;
    rdh[0]    = 1 / zlim[2];
    rdh[1]    = 1 / ylim[2];
    rdh[2]    = 1 / xlim[2];
  }

  // reset count
  if (reset) {
    for (int is = 0; is < particle.size(); is++) {
      particle[is]->reset_count();
    }
  }

  //
  // count particles
  //
  for (int is = 0; is < particle.size(); is++) {
    const int out_of_bounds = particle[is]->Ng;
    float64  *xu            = particle[is]->xu.data();

    // loop over particles
    for (int ip = Lbp[is]; ip <= Ubp[is]; ip++) {
      int iz = Particle::digitize(xu[Particle::Nc * ip + 2], zlim[0], rdh[0]);
      int iy = Particle::digitize(xu[Particle::Nc * ip + 1], ylim[0], rdh[1]);
      int ix = Particle::digitize(xu[Particle::Nc * ip + 0], xlim[0], rdh[2]);
      int ii = iz * stride[0] + iy * stride[1] + ix * stride[2];

      // take care out-of-bounds particles
      ii = (iz < zrange[0] || iz > zrange[1]) ? out_of_bounds : ii;
      ii = (iy < yrange[0] || iy > yrange[1]) ? out_of_bounds : ii;
      ii = (ix < xrange[0] || ix > xrange[1]) ? out_of_bounds : ii;

      particle[is]->increment(ip, ii);
    }
  }
}

DEFINE_MEMBER(void, begin_bc_exchange)(MpiBuffer *mpibuf, ParticleList &particle)
{
  const size_t header_size = sizeof(int);
  const size_t data_size   = sizeof(float64) * Particle::Nc;
  const size_t Ns          = particle.size();

  xt::xtensor<int, 4> send_count = xt::zeros<int>({Ns, 3ul, 3ul, 3ul});

  //
  // count particles
  //
  {
    int Lbp[Ns];
    int Ubp[Ns];
    for (int is = 0; is < Ns; is++) {
      Lbp[is] = 0;
      Ubp[is] = particle[is]->Np - 1;
    }

    count_particle(particle, Lbp, Ubp, true);
  }

  //
  // pack out-of-bounds particles
  //
  for (int is = 0; is < Ns; is++) {
    float64 *xu = particle[is]->xu.data();

    // loop over particles
    for (int ip = 0; ip < particle[is]->Np; ip++) {
      float64 *ptcl = &xu[Particle::Nc * ip];
      int      dirz = (ptcl[2] > zlim[1]) - (ptcl[2] < zlim[0]);
      int      diry = (ptcl[1] > ylim[1]) - (ptcl[1] < ylim[0]);
      int      dirx = (ptcl[0] > xlim[1]) - (ptcl[0] < xlim[0]);

      if (dirx == 0 && diry == 0 && dirz == 0) {
        continue;
      }

      int iz  = dirz + 1;
      int iy  = diry + 1;
      int ix  = dirx + 1;
      int pos = mpibuf->bufaddr(iz, iy, ix) + data_size * send_count(is, iz, iy, ix) +
                header_size * (is + 1);
      std::memcpy(mpibuf->sendbuf.get(pos), ptcl, data_size);
      send_count(is, iz, iy, ix)++;
    }
  }

  //
  // begin exchange particles
  //
  for (int dirz = -1, iz = 0; dirz <= +1; dirz++, iz++) {
    for (int diry = -1, iy = 0; diry <= +1; diry++, iy++) {
      for (int dirx = -1, ix = 0; dirx <= +1; dirx++, ix++) {
        // skip send/recv to itself
        if (dirz == 0 && diry == 0 && dirx == 0) {
          mpibuf->sendreq(iz, iy, ix) = MPI_REQUEST_NULL;
          mpibuf->recvreq(iz, iy, ix) = MPI_REQUEST_NULL;
          continue;
        }

        // add header for each species
        int addr = mpibuf->bufaddr(iz, iy, ix);
        int byte = 0;
        for (int is = 0; is < Ns; is++) {
          std::memcpy(mpibuf->sendbuf.get(addr + byte), &send_count(is, iz, iy, ix), header_size);
          byte += header_size + data_size * send_count(is, iz, iy, ix);
        }

        int   nbrank = this->get_nb_rank(dirz, diry, dirx);
        int   sndtag = this->get_sndtag(dirz, diry, dirx);
        int   rcvtag = this->get_rcvtag(dirz, diry, dirx);
        void *sndptr = mpibuf->sendbuf.get(mpibuf->bufaddr(iz, iy, ix));
        void *rcvptr = mpibuf->recvbuf.get(mpibuf->bufaddr(iz, iy, ix));

        // send/recv calls
        MPI_Isend(sndptr, byte, MPI_BYTE, nbrank, sndtag, mpibuf->comm,
                  &mpibuf->sendreq(iz, iy, ix));
        MPI_Irecv(rcvptr, byte, MPI_BYTE, nbrank, rcvtag, mpibuf->comm,
                  &mpibuf->recvreq(iz, iy, ix));
      }
    }
  }
}

DEFINE_MEMBER(void, end_bc_exchange)(MpiBuffer *mpibuf, xt::xtensor<float64, 4> &array, bool append)
{
  auto Ia = xt::all();

  //
  // wait for MPI recv calls to complete
  //
  MPI_Waitall(27, mpibuf->recvreq.data(), MPI_STATUS_IGNORE);

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
        if (this->get_nb_rank(dirz, diry, dirx) == MPI_PROC_NULL) {
          continue;
        }

        // index range
        auto Iz = xt::range(this->recvlb[0][iz], this->recvub[0][iz] + 1);
        auto Iy = xt::range(this->recvlb[1][iy], this->recvub[1][iy] + 1);
        auto Ix = xt::range(this->recvlb[2][ix], this->recvub[2][ix] + 1);

        // unpack
        auto     view   = xt::view(array, Iz, Iy, Ix, Ia);
        void    *rcvptr = mpibuf->recvbuf.get(mpibuf->bufaddr(iz, iy, ix));
        float64 *ptr    = static_cast<float64 *>(rcvptr);

        // copy or append
        if (append) {
          std::transform(ptr, ptr + view.size(), view.begin(), view.begin(), std::plus<float64>());
        } else {
          std::copy(ptr, ptr + view.size(), view.begin());
        }
      }
    }
  }

  //
  // wait for MPI send calls to complete (this is optional)
  //
  MPI_Waitall(27, mpibuf->sendreq.data(), MPI_STATUS_IGNORE);
}

DEFINE_MEMBER(void, end_bc_exchange)(MpiBuffer *mpibuf, ParticleList &particle)
{
  const size_t header_size = sizeof(int);
  const size_t data_size   = sizeof(float64) * Particle::Nc;
  const size_t Ns          = particle.size();

  // array to store total number of particles
  std::vector<int> num_particle(Ns);
  for (int is = 0; is < Ns; is++) {
    num_particle[is] = particle[is]->Np;
  }

  //
  // wait for MPI recv calls to complete
  //
  MPI_Waitall(27, mpibuf->recvreq.data(), MPI_STATUS_IGNORE);

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
        if (this->get_nb_rank(dirz, diry, dirx) == MPI_PROC_NULL) {
          continue;
        }

        // copy to the end of particle array
        char *recvptr = mpibuf->recvbuf.get(mpibuf->bufaddr(iz, iy, ix));
        for (int is = 0; is < Ns; is++) {
          // header
          int cnt;
          std::memcpy(&cnt, recvptr, header_size);
          recvptr += header_size;

          // particles
          float64 *ptcl = &particle[is]->xu(num_particle[is], 0);
          std::memcpy(ptcl, recvptr, data_size * cnt);
          recvptr += data_size * cnt;

          // increment number of particles
          num_particle[is] += cnt;
        }
      }
    }
  }

  //
  // count received particles
  //
  {
    int Lbp[Ns];
    int Ubp[Ns];
    for (int is = 0; is < Ns; is++) {
      Lbp[is] = particle[is]->Np;
      Ubp[is] = num_particle[is] - 1;
    }

    count_particle(particle, Lbp, Ubp, false);

    // now update number of particles
    for (int is = 0; is < Ns; is++) {
      particle[is]->Np = num_particle[is];
    }
  }

  //
  // sort (or rearrage) particle array
  //
  for (int is = 0; is < Ns; is++) {
    particle[is]->sort();
  }

  //
  // wait for MPI send calls to complete (this is optional)
  //
  MPI_Waitall(27, mpibuf->sendreq.data(), MPI_STATUS_IGNORE);
}

DEFINE_MEMBER(bool, set_boundary_query)(const int mode)
{
  int  flag   = 0;
  int  bcmode = mode;
  bool send   = (mode & SendMode) == SendMode; // send flag
  bool recv   = (mode & RecvMode) == RecvMode; // recv flag

  // remove send/recv bits
  bcmode &= ~SendMode;
  bcmode &= ~RecvMode;

  // MPI buffer
  MpiBuffer *mpibuf = this->mpibufvec[bcmode].get();

  if (send == true && recv == true) {
    // both send/recv
    MPI_Testall(27, mpibuf->sendreq.data(), &flag, MPI_STATUS_IGNORE);
    MPI_Testall(27, mpibuf->recvreq.data(), &flag, MPI_STATUS_IGNORE);
  } else if (send == true) {
    // send
    MPI_Testall(27, mpibuf->sendreq.data(), &flag, MPI_STATUS_IGNORE);
  } else if (recv == true) {
    // recv
    MPI_Testall(27, mpibuf->recvreq.data(), &flag, MPI_STATUS_IGNORE);
  }

  return !(flag == 0);
}

DEFINE_MEMBER(void, set_boundary_physical)(const int mode)
{
  // lower boundary in z
  if (this->get_nb_rank(-1, 0, 0) == MPI_PROC_NULL) {
    ERRORPRINT("Non-periodic boundary condition has not been implemented!\n");
  }

  // upper boundary in z
  if (this->get_nb_rank(+1, 0, 0) == MPI_PROC_NULL) {
    ERRORPRINT("Non-periodic boundary condition has not been implemented!\n");
  }

  // lower boundary in y
  if (this->get_nb_rank(0, -1, 0) == MPI_PROC_NULL) {
    ERRORPRINT("Non-periodic boundary condition has not been implemented!\n");
  }

  // upper boundary in y
  if (this->get_nb_rank(0, +1, 0) == MPI_PROC_NULL) {
    ERRORPRINT("Non-periodic boundary condition has not been implemented!\n");
  }

  // lower boundary in x
  if (this->get_nb_rank(0, 0, -1) == MPI_PROC_NULL) {
    ERRORPRINT("Non-periodic boundary condition has not been implemented!\n");
  }

  // upper boundary in x
  if (this->get_nb_rank(0, 0, +1) == MPI_PROC_NULL) {
    ERRORPRINT("Non-periodic boundary condition has not been implemented!\n");
  }
}

template class Chunk3D<1>;
template class Chunk3D<2>;
template class Chunk3D<3>;
template class Chunk3D<4>;

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
