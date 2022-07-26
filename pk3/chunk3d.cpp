// -*- C++ -*-
#include "chunk3d.hpp"

#define DEFINE_MEMBER(type, name)                                                                  \
  template <int Nb>                                                                                \
  type Chunk3D<Nb>::name

DEFINE_MEMBER(, Chunk3D)(const int dims[3], const int id) : BaseChunk<3>(dims, id)
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
  zc.fill(0);
  yc.fill(0);
  xc.fill(0);

  set_buffer_address();
  sendbuf.resize(bufsize);
  recvbuf.resize(bufsize);
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
  case PackAll:
    count += BaseChunk<3>::pack(BaseChunk<3>::PackAll, &ptr[count]);
    count += memcpy_count(&ptr[count], xc.data(), xc.size() * sizeof(float64), false);
    count += memcpy_count(&ptr[count], yc.data(), yc.size() * sizeof(float64), false);
    count += memcpy_count(&ptr[count], zc.data(), zc.size() * sizeof(float64), false);
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
    count += memcpy_count(&ptr[count], &delh, sizeof(float64), true);
    count += memcpy_count(&ptr[count], xlim, 3 * sizeof(float64), true);
    count += memcpy_count(&ptr[count], ylim, 3 * sizeof(float64), true);
    count += memcpy_count(&ptr[count], zlim, 3 * sizeof(float64), true);
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
    count += memcpy_count(&delh, &ptr[count], sizeof(float64), true);
    count += memcpy_count(xlim, &ptr[count], 3 * sizeof(float64), true);
    count += memcpy_count(ylim, &ptr[count], 3 * sizeof(float64), true);
    count += memcpy_count(zlim, &ptr[count], 3 * sizeof(float64), true);
    break;
  default:
    break;
  }

  return count;
}

DEFINE_MEMBER(void, set_buffer_address)()
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

  // calculate buffer address and size
  pos     = xt::cumsum(pos);
  pos     = xt::roll(pos, 1);
  bufsize = pos(0) * sizeof(float64) * 6;
  pos(0)  = 0;

  // reshape and store
  pos.reshape({3, 3, 3});
  bufaddr = pos;
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
  for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
    zc(iz) = zlim[0] + (iz + 0.5) * delh;
  }

  for (int iy = Lby - Nb; iy <= Uby + Nb; iy++) {
    yc(iy) = ylim[0] + (iy + 0.5) * delh;
  }

  for (int ix = Lbx - Nb; ix <= Ubx + Nb; ix++) {
    xc(ix) = xlim[0] + (ix + 0.5) * delh;
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
