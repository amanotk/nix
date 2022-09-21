// -*- C++ -*-
#include "chunk3d.hpp"

#define DEFINE_MEMBER(type, name)                                                                  \
  template <int Nb>                                                                                \
  type Chunk3D<Nb>::name

DEFINE_MEMBER(, Chunk3D)
(const int dims[3], const int id) : Chunk<3>(dims, id), delh(1.0), require_sort(true)
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

  // reset load
  reset_load();
}

DEFINE_MEMBER(, ~Chunk3D)()
{
  zc.resize({0});
  yc.resize({0});
  xc.resize({0});
}

DEFINE_MEMBER(int, pack)(void *buffer, const int address)
{
  using common::memcpy_count;

  int count = address;

  count += Chunk<3>::pack(buffer, count);
  count += memcpy_count(buffer, xc.data(), xc.size() * sizeof(float64), count, 0);
  count += memcpy_count(buffer, yc.data(), yc.size() * sizeof(float64), count, 0);
  count += memcpy_count(buffer, zc.data(), zc.size() * sizeof(float64), count, 0);
  count += memcpy_count(buffer, &delh, sizeof(float64), count, 0);
  count += memcpy_count(buffer, xlim, 3 * sizeof(float64), count, 0);
  count += memcpy_count(buffer, ylim, 3 * sizeof(float64), count, 0);
  count += memcpy_count(buffer, zlim, 3 * sizeof(float64), count, 0);
  // MPI buffer (NOTE: MPI communicator is NOT packed)
  {
    int nmode = mpibufvec.size();
    count += memcpy_count(buffer, &nmode, sizeof(int), count, 0);

    for (int mode = 0; mode < nmode; mode++) {
      count += mpibufvec[mode]->pack(buffer, count);
    }
  }

  return count;
}

DEFINE_MEMBER(int, unpack)(void *buffer, const int address)
{
  using common::memcpy_count;

  int count = address;

  count += Chunk<3>::unpack(buffer, count);
  count += memcpy_count(xc.data(), buffer, xc.size() * sizeof(float64), 0, count);
  count += memcpy_count(yc.data(), buffer, yc.size() * sizeof(float64), 0, count);
  count += memcpy_count(zc.data(), buffer, zc.size() * sizeof(float64), 0, count);
  count += memcpy_count(&delh, buffer, sizeof(float64), 0, count);
  count += memcpy_count(xlim, buffer, 3 * sizeof(float64), 0, count);
  count += memcpy_count(ylim, buffer, 3 * sizeof(float64), 0, count);
  count += memcpy_count(zlim, buffer, 3 * sizeof(float64), 0, count);
  // MPI buffer (NOTE: MPI communicator is NOT unpacked)
  {
    int nmode = 0;
    count += memcpy_count(&nmode, buffer, sizeof(int), 0, count);
    mpibufvec.resize(nmode);

    for (int mode = 0; mode < nmode; mode++) {
      mpibufvec[mode] = std::make_shared<MpiBuffer>();
      count += mpibufvec[mode]->unpack(buffer, count);
    }
  }

  return count;
}

DEFINE_MEMBER(void, set_global_context)(const int *offset, const int *ndims)
{
  this->ndims[0]  = ndims[0];
  this->ndims[1]  = ndims[1];
  this->ndims[2]  = ndims[2];
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

DEFINE_MEMBER(void, sort_particle)(ParticleVec &particle)
{
  for (int is = 0; is < particle.size(); is++) {
    count_particle(particle[is], 0, particle[is]->Np - 1, true);
    particle[is]->sort();
  }
}

DEFINE_MEMBER(void, set_mpi_communicator)(const int mode, MPI_Comm &comm)
{
  if (mode >= 0 && mode < mpibufvec.size()) {
    mpibufvec[mode]->comm = comm;
  } else {
    ERRORPRINT("invalid index %d for mpibufvec\n", mode);
  }
}

DEFINE_MEMBER(void, count_particle)(PtrParticle particle, const int Lbp, const int Ubp, bool reset)
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
    particle->reset_count();
  }

  //
  // count particles
  //
  const int out_of_bounds = particle->Ng;
  float64 * xu            = particle->xu.data();

  // loop over particles
  for (int ip = Lbp; ip <= Ubp; ip++) {
    int iz = Particle::digitize(xu[Particle::Nc * ip + 2], zlim[0], rdh[0]);
    int iy = Particle::digitize(xu[Particle::Nc * ip + 1], ylim[0], rdh[1]);
    int ix = Particle::digitize(xu[Particle::Nc * ip + 0], xlim[0], rdh[2]);
    int ii = iz * stride[0] + iy * stride[1] + ix * stride[2];

    // take care out-of-bounds particles
    ii = (iz < zrange[0] || iz > zrange[1]) ? out_of_bounds : ii;
    ii = (iy < yrange[0] || iy > yrange[1]) ? out_of_bounds : ii;
    ii = (ix < xrange[0] || ix > xrange[1]) ? out_of_bounds : ii;

    particle->increment(ip, ii);
  }
}

DEFINE_MEMBER(int, pack_diagnostic_load)(void *buffer, const int address)
{
  int count = sizeof(float64) * load.size() + address;

  if (buffer == nullptr) {
    return count;
  }

  std::copy(load.begin(), load.end(),
            reinterpret_cast<float64 *>(static_cast<char *>(buffer) + address));

  return count;
}

DEFINE_MEMBER(int, pack_diagnostic_coord)(void *buffer, const int address, const int dir)
{
  size_t size  = dims[dir];
  int    count = sizeof(float64) * size + address;

  if (buffer == nullptr) {
    return count;
  }

  float64 *ptr = reinterpret_cast<float64 *>(static_cast<char *>(buffer) + address);

  switch (dir) {
  case 0: {
    auto zz = xt::view(zc, xt::range(Lbz, Ubz + 1));
    std::copy(zz.begin(), zz.end(), ptr);
  } break;
  case 1: {
    auto yy = xt::view(yc, xt::range(Lby, Uby + 1));
    std::copy(yy.begin(), yy.end(), ptr);
  } break;
  case 2: {
    auto xx = xt::view(xc, xt::range(Lbx, Ubx + 1));
    std::copy(xx.begin(), xx.end(), ptr);
  } break;
  default:
    break;
  }

  return count;
}

DEFINE_MEMBER(int, pack_diagnostic_field)
(void *buffer, const int address, xt::xtensor<float64, 4> &u)
{
  size_t size  = dims[2] * dims[1] * dims[0] * u.shape(3);
  int    count = sizeof(float64) * size + address;

  if (buffer == nullptr) {
    return count;
  }

  auto Iz = xt::range(Lbz, Ubz + 1);
  auto Iy = xt::range(Lby, Uby + 1);
  auto Ix = xt::range(Lbx, Ubx + 1);
  auto vv = xt::view(u, Iz, Iy, Ix, xt::all());

  // packing
  char *ptr = &static_cast<char *>(buffer)[address];
  std::copy(vv.begin(), vv.end(), reinterpret_cast<float64 *>(ptr));

  return count;
}

DEFINE_MEMBER(int, pack_diagnostic_particle)
(void *buffer, const int address, PtrParticle p)
{
  using common::memcpy_count;

  int count = address;

  count += memcpy_count(buffer, p->xu.data(), p->Np * Particle::Nc * sizeof(float64), count, 0);

  return count;
}

DEFINE_MEMBER(void, begin_bc_exchange)(PtrMpiBuffer mpibuf, ParticleVec &particle)
{
  const size_t header_size = sizeof(int);
  const size_t data_size   = sizeof(float64) * Particle::Nc;
  const size_t Ns          = particle.size();

  xt::xtensor<int, 4> snd_count = xt::zeros<int>({Ns + 1, 3ul, 3ul, 3ul});

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
      int pos = mpibuf->bufaddr(iz, iy, ix) + data_size * snd_count(Ns, iz, iy, ix) +
                header_size * (is + 1);
      std::memcpy(mpibuf->sendbuf.get(pos), ptcl, data_size);
      snd_count(is, iz, iy, ix)++;
      snd_count(Ns, iz, iy, ix)++; // total number of send particles
    }
  }

  // check buffer size and reallocate if needed
  {
    auto I    = xt::all();
    bool safe = xt::all(
        xt::greater_equal(mpibuf->bufsize, 2 * data_size * xt::view(snd_count, Ns, I, I, I)));

    if (safe == false) {
      int elembyte = 2 * data_size * xt::amax(xt::view(snd_count, Ns, I, I, I))();
      set_mpi_buffer(mpibuf, header_size, elembyte);
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
        int addr     = mpibuf->bufaddr(iz, iy, ix);
        int rcv_byte = mpibuf->bufsize(iz, iy, ix);
        int snd_byte = 0;
        for (int is = 0; is < Ns; is++) {
          std::memcpy(mpibuf->sendbuf.get(addr + snd_byte), &snd_count(is, iz, iy, ix),
                      header_size);
          snd_byte += header_size + data_size * snd_count(is, iz, iy, ix);
        }

        int   nbrank = get_nb_rank(dirz, diry, dirx);
        int   sndtag = get_sndtag(dirz, diry, dirx);
        int   rcvtag = get_rcvtag(dirz, diry, dirx);
        void *sndptr = mpibuf->sendbuf.get(mpibuf->bufaddr(iz, iy, ix));
        void *rcvptr = mpibuf->recvbuf.get(mpibuf->bufaddr(iz, iy, ix));

        // send/recv calls
        MPI_Isend(sndptr, snd_byte, MPI_BYTE, nbrank, sndtag, mpibuf->comm,
                  &mpibuf->sendreq(iz, iy, ix));
        MPI_Irecv(rcvptr, rcv_byte, MPI_BYTE, nbrank, rcvtag, mpibuf->comm,
                  &mpibuf->recvreq(iz, iy, ix));
      }
    }
  }
}

DEFINE_MEMBER(void, end_bc_exchange)(PtrMpiBuffer mpibuf, ParticleVec &particle)
{
  const size_t header_size = sizeof(int);
  const size_t data_size   = sizeof(float64) * Particle::Nc;
  const size_t Ns          = particle.size();

  // wait for MPI recv calls to complete
  MPI_Waitall(27, mpibuf->recvreq.data(), MPI_STATUS_IGNORE);

  // array to store total number of particles
  std::vector<int> num_particle(Ns);
  for (int is = 0; is < Ns; is++) {
    num_particle[is] = particle[is]->Np;
  }

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

        // copy to the end of particle array
        char *recvptr = mpibuf->recvbuf.get(mpibuf->bufaddr(iz, iy, ix));
        for (int is = 0; is < Ns; is++) {
          // header
          int cnt;
          std::memcpy(&cnt, recvptr, header_size);
          recvptr += header_size;

          if (num_particle[is] + cnt > particle[is]->Np_total) {
            // run out of particle buffer and try to reallocate twice the original
            particle[is]->resize(2 * particle[is]->Np_total);
          }

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
  // set boundary condition and append count for received particles
  //
  for (int is = 0; is < Ns; is++) {
    set_boundary_particle(particle[is], particle[is]->Np, num_particle[is] - 1);
    count_particle(particle[is], particle[is]->Np, num_particle[is] - 1, false);
    // now update number of particles
    particle[is]->Np = num_particle[is];
  }

  //
  // sort (or rearrage) particle array
  //
  for (int is = 0; is < Ns; is++) {
    particle[is]->sort();
  }

  // wait for MPI send calls to complete (optional)
  MPI_Waitall(27, mpibuf->sendreq.data(), MPI_STATUS_IGNORE);

  //
  // automatically resize particle buffer
  //
  for (int is = 0; is < Ns; is++) {
    const float64 fraction1 = 0.8; // increase when > 80% is used
    const float64 fraction2 = 0.2; // decrease when < 20% is used

    int new_np = particle[is]->Np_total;

    // increase particle buffer
    if (particle[is]->Np > fraction1 * particle[is]->Np_total) {
      new_np = 2.0 * particle[is]->Np_total;
    }

    // decrease particle buffer
    if (particle[is]->Np < fraction2 * particle[is]->Np_total) {
      new_np = 0.5 * particle[is]->Np_total;
    }

    // resize if needed
    particle[is]->resize(new_np);
  }
}

DEFINE_MEMBER(void, begin_bc_exchange)
(PtrMpiBuffer mpibuf, xt::xtensor<float64, 4> &array, bool moment)
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

        int   byte   = 0;
        int   nbrank = get_nb_rank(dirz, diry, dirx);
        int   sndtag = get_sndtag(dirz, diry, dirx);
        int   rcvtag = get_rcvtag(dirz, diry, dirx);
        void *sndptr = mpibuf->sendbuf.get(mpibuf->bufaddr(iz, iy, ix));
        void *rcvptr = mpibuf->recvbuf.get(mpibuf->bufaddr(iz, iy, ix));

        if (moment == true) {
          // index range
          auto Iz   = xt::range(recvlb[0][iz], recvub[0][iz] + 1);
          auto Iy   = xt::range(recvlb[1][iy], recvub[1][iy] + 1);
          auto Ix   = xt::range(recvlb[2][ix], recvub[2][ix] + 1);
          auto view = xt::view(array, Iz, Iy, Ix, Ia);

          // pack
          byte = view.size() * sizeof(float64);
          std::copy(view.begin(), view.end(), static_cast<float64 *>(sndptr));
        } else {
          // index range
          auto Iz   = xt::range(sendlb[0][iz], sendub[0][iz] + 1);
          auto Iy   = xt::range(sendlb[1][iy], sendub[1][iy] + 1);
          auto Ix   = xt::range(sendlb[2][ix], sendub[2][ix] + 1);
          auto view = xt::view(array, Iz, Iy, Ix, Ia);

          // pack
          byte = view.size() * sizeof(float64);
          std::copy(view.begin(), view.end(), static_cast<float64 *>(sndptr));
        }

        // send/recv calls
        MPI_Isend(sndptr, byte, MPI_BYTE, nbrank, sndtag, mpibuf->comm,
                  &mpibuf->sendreq(iz, iy, ix));
        MPI_Irecv(rcvptr, byte, MPI_BYTE, nbrank, rcvtag, mpibuf->comm,
                  &mpibuf->recvreq(iz, iy, ix));
      }
    }
  }
}

DEFINE_MEMBER(void, end_bc_exchange)
(PtrMpiBuffer mpibuf, xt::xtensor<float64, 4> &array, bool moment)
{
  auto Ia = xt::all();

  // wait for MPI recv calls to complete
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
        if (get_nb_rank(dirz, diry, dirx) == MPI_PROC_NULL) {
          continue;
        }

        if (moment == true) {
          // index range
          auto Iz   = xt::range(sendlb[0][iz], sendub[0][iz] + 1);
          auto Iy   = xt::range(sendlb[1][iy], sendub[1][iy] + 1);
          auto Ix   = xt::range(sendlb[2][ix], sendub[2][ix] + 1);
          auto view = xt::view(array, Iz, Iy, Ix, Ia);

          // unpack
          void *   rcvptr = mpibuf->recvbuf.get(mpibuf->bufaddr(iz, iy, ix));
          float64 *ptr    = static_cast<float64 *>(rcvptr);

          // accumulate
          std::transform(ptr, ptr + view.size(), view.begin(), view.begin(), std::plus<float64>());
        } else {
          // index range
          auto Iz   = xt::range(recvlb[0][iz], recvub[0][iz] + 1);
          auto Iy   = xt::range(recvlb[1][iy], recvub[1][iy] + 1);
          auto Ix   = xt::range(recvlb[2][ix], recvub[2][ix] + 1);
          auto view = xt::view(array, Iz, Iy, Ix, Ia);

          // unpack
          void *   rcvptr = mpibuf->recvbuf.get(mpibuf->bufaddr(iz, iy, ix));
          float64 *ptr    = static_cast<float64 *>(rcvptr);

          std::copy(ptr, ptr + view.size(), view.begin());
        }
      }
    }
  }

  // wait for MPI send calls to complete (optional)
  MPI_Waitall(27, mpibuf->sendreq.data(), MPI_STATUS_IGNORE);
}

DEFINE_MEMBER(bool, set_boundary_query)(const int mode)
{
  int  flag   = 0;
  int  bcmode = mode;
  bool send   = (mode & common::SendMode) == common::SendMode; // send flag
  bool recv   = (mode & common::RecvMode) == common::RecvMode; // recv flag

  // remove send/recv bits
  bcmode &= ~common::SendMode;
  bcmode &= ~common::RecvMode;

  // MPI buffer
  PtrMpiBuffer mpibuf = mpibufvec[bcmode];

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

DEFINE_MEMBER(void, set_boundary_particle)(PtrParticle particle, int Lbp, int Ubp)
{
  float64 xyzmin[3] = {0.0, 0.0, 0.0};
  float64 xyzmax[3] = {ndims[2] * delh, ndims[1] * delh, ndims[0] * delh};
  float64 xyzlen[3] = {ndims[2] * delh, ndims[1] * delh, ndims[0] * delh};

  // push particle position
  for (int ip = Lbp; ip <= Ubp; ip++) {
    float64 *xu = &particle->xu(ip, 0);

    // apply periodic boundary condition
    xu[0] += (xu[0] < xyzmin[0]) * xyzlen[0] - (xu[0] >= xyzmax[0]) * xyzlen[0];
    xu[1] += (xu[1] < xyzmin[1]) * xyzlen[1] - (xu[1] >= xyzmax[1]) * xyzlen[1];
    xu[2] += (xu[2] < xyzmin[2]) * xyzlen[2] - (xu[2] >= xyzmax[2]) * xyzlen[2];
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
