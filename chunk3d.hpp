// -*- C++ -*-
#ifndef _CHUNK3D_HPP_
#define _CHUNK3D_HPP_

#include "buffer.hpp"
#include "chunk.hpp"
#include "debug.hpp"
#include "jsonio.hpp"
#include "nix.hpp"
#include "particle.hpp"
#include "xtensorall.hpp"

NIX_NAMESPACE_BEGIN

///
/// @brief Base class for 3D Chunk
/// @tparam Nb number of boundary margins
///
template <int Nb>
class Chunk3D : public Chunk<3>
{
public:
  using T_array3d  = xt::xtensor_fixed<int, xt::xshape<3, 3, 3>>;
  using T_request  = xt::xtensor_fixed<MPI_Request, xt::xshape<3, 3, 3>>;
  using T_datatype = xt::xtensor_fixed<MPI_Datatype, xt::xshape<3, 3, 3>>;

  ///
  /// @brief MPI buffer
  ///
  struct MpiBuffer {
    MPI_Comm   comm;
    Buffer     sendbuf;
    Buffer     recvbuf;
    T_array3d  bufsize;
    T_array3d  bufaddr;
    T_request  sendreq;
    T_request  recvreq;
    T_datatype sendtype;
    T_datatype recvtype;

    ///
    /// constructor
    ///
    MpiBuffer() : comm(MPI_COMM_WORLD)
    {
    }

    ///
    /// @brief return send buffer for given direction
    /// @param iz z direction index
    /// @param iy y direction index
    /// @param ix x direction index
    /// @return buffer pointer
    ///
    void* get_send_buffer(int iz, int iy, int ix)
    {
      return sendbuf.get(bufaddr(iz, iy, ix));
    }

    ///
    /// @brief return recv buffer for given direction
    /// @param iz z direction index
    /// @param iy y direction index
    /// @param ix x direction index
    /// @return buffer pointer
    ///
    void* get_recv_buffer(int iz, int iy, int ix)
    {
      return recvbuf.get(bufaddr(iz, iy, ix));
    }

    ///
    /// @brief pack the content into given `buffer`
    /// @param buffer pointer to buffer to pack
    /// @param address first address of buffer to which the data will be packed
    /// @return `address` + (number of bytes packed as a result)
    ///
    int pack(void* buffer, int address)
    {
      int count = address;
      int ssize = sendbuf.size;
      int rsize = recvbuf.size;
      int asize = bufsize.size() * sizeof(int);

      count += memcpy_count(buffer, &ssize, sizeof(int), count, 0);
      count += memcpy_count(buffer, &rsize, sizeof(int), count, 0);
      count += memcpy_count(buffer, bufsize.data(), asize, count, 0);
      count += memcpy_count(buffer, bufaddr.data(), asize, count, 0);

      return count;
    }

    ///
    /// @brief unpack the content from given `buffer`
    /// @param buffer pointer to buffer from unpack
    /// @param address first address of buffer to which the data will be packed
    /// @return `address` + (number of bytes packed as a result)
    ///
    int unpack(void* buffer, int address)
    {
      int count = address;
      int ssize = 0;
      int rsize = 0;
      int asize = bufsize.size() * sizeof(int);

      count += memcpy_count(&ssize, buffer, sizeof(int), 0, count);
      count += memcpy_count(&rsize, buffer, sizeof(int), 0, count);
      count += memcpy_count(bufsize.data(), buffer, asize, 0, count);
      count += memcpy_count(bufaddr.data(), buffer, asize, 0, count);

      // memory allocation
      sendbuf.resize(ssize);
      recvbuf.resize(rsize);

      return count;
    }
  };
  using PtrMpiBuffer = std::shared_ptr<MpiBuffer>;
  using MpiBufferVec = std::vector<PtrMpiBuffer>;

  /// boundary margin
  static const int boundary_margin = Nb;

protected:
  bool require_sort; ///< sort flag
  int  gdims[3];     ///< global number of grids
  int  offset[3];    ///< global index offset
  int  Lbx;          ///< lower bound in x
  int  Ubx;          ///< upper bound in x
  int  Lby;          ///< lower bound in y
  int  Uby;          ///< upper bound in y
  int  Lbz;          ///< lower bound in z
  int  Ubz;          ///< upper bound in z
  int  sendlb[3][3]; ///< lower bound for send
  int  sendub[3][3]; ///< upper bound for send
  int  recvlb[3][3]; ///< lower bound for recv
  int  recvub[3][3]; ///< upper bound for recv

  xt::xtensor<float64, 1> xc;        ///< x coordinate
  xt::xtensor<float64, 1> yc;        ///< y coordinate
  xt::xtensor<float64, 1> zc;        ///< z coordinate
  float64                 delx;      ///< grid size in x
  float64                 dely;      ///< grid size in y
  float64                 delz;      ///< grid size in z
  float64                 xlim[3];   ///< physical domain in x
  float64                 ylim[3];   ///< physical domain in y
  float64                 zlim[3];   ///< physical domain in z
  float64                 gxlim[3];  ///< global physical domain in x
  float64                 gylim[3];  ///< global physical domain in y
  float64                 gzlim[3];  ///< global physical domain in z
  MpiBufferVec            mpibufvec; ///< MPI buffer vector

public:
  ///
  /// @brief constructor
  /// @param dims number of grids for each direction
  /// @param id Chunk ID
  ///
  Chunk3D(const int dims[3], int id = 0);

  ///
  /// @brief setup initial condition (pure virtual)
  /// @param config configuration
  ///
  virtual void setup(json& config) = 0;

  ///
  /// @brief begin boundary exchange (pure virtual)
  /// @param mode mode of boundary exchange
  ///
  virtual void set_boundary_begin(int mode) override = 0;

  ///
  /// @brief end boundary exchange (pure virtual)
  /// @param mode mode of boundary exchange
  ///
  virtual void set_boundary_end(int mode) override = 0;

  ///
  /// @brief pack the content of Chunk into given `buffer`
  /// @param buffer pointer to buffer to pack
  /// @param address first address of buffer to which the data will be packed
  /// @return `address` + (number of bytes packed as a result)
  ///
  virtual int pack(void* buffer, int address) override;

  ///
  /// @brief unpack the content of Chunk from given `buffer`
  /// @param buffer point to buffer from unpack
  /// @param address first address of buffer from which the data will be unpacked
  /// @return `address` + (number of bytes unpacked as a result)
  ///
  virtual int unpack(void* buffer, int address) override;

  ///
  /// @brief set the global context of Chunk
  /// @param offset offset for each direction in global dimensions
  /// @param gdims global number of grids for each direction
  ///
  virtual void set_global_context(const int* offset, const int* gdims);

  ///
  /// @brief set MPI communicator to MpiBuffer of given `mode`
  /// @param mode mode to specify MpiBuffer
  /// @param comm MPI communicator to be set to MpiBuffer
  ///
  virtual void set_mpi_communicator(int mode, MPI_Comm& comm);

  ///
  /// @brief count particles in cells to prepare for sorting
  /// @param particle particle species
  /// @param Lbp first index of particle array to be counted
  /// @param Ubp last index of particle array to be counted (inclusive)
  /// @param reset reset the count array before counting
  ///
  virtual void count_particle(PtrParticle particle, int Lbp, int Ubp, bool reset = true);

  ///
  /// @brief perform particle sorting
  /// @param particle list of particle to be sorted
  ///
  virtual void sort_particle(ParticleVec& particle);

  ///
  /// @brief inject particle into the system
  /// @param particle list of particle to which new particles will be injected
  ///
  virtual void inject_particle(ParticleVec& particle);

  ///
  /// @brief query status of boundary exchange
  /// @param mode mode of boundary exchange
  /// @return true if boundary exchange is finished and false otherwise
  ///
  virtual bool set_boundary_query(int mode = 0) override;

  ///
  /// @brief set physical boundary condition
  /// @param mode mode of boundary exchange
  ///
  virtual void set_boundary_physical(int mode = 0) override;

  ///
  /// @brief set boundary condition to particle array
  /// @param particle particle species
  /// @param Lbp first index of particle array
  /// @param Ubp last index of particle array (inclusive)
  ///
  virtual void set_boundary_particle(PtrParticle particle, int Lbp, int Ubp);

  ///
  /// @brief setup MPI Buffer
  /// @param mpibuf MPI buffer to be setup
  /// @param mode +1 for send, -1 for recv, 0 for both
  /// @param headbyte number of bytes used for header
  /// @param elembyte number of bytes for each element
  ///
  void set_mpi_buffer(PtrMpiBuffer mpibuf, int mode, int headbyte, int elembyte);

  ///
  /// @brief setup MPI Buffer
  /// @param mpibuf MPI buffer to be setup
  /// @param mode +1 for send, -1 for recv, 0 for both
  /// @param headbyte number of bytes used for header
  /// @param sizebyte number of bytes
  ///
  void set_mpi_buffer(PtrMpiBuffer mpibuf, int mode, int headbyte, const int sizebyte[3][3][3]);

  ///
  /// @brief return MpiBuffer of given mode of boundary exchange
  /// @param mode mode of MpiBuffer
  /// @return PtrMpiBuffer or std::shared_ptr<MpiBuffer>
  ///
  PtrMpiBuffer get_mpi_buffer(int mode)
  {
    return mpibufvec[mode];
  }

  ///
  /// @brief get lower bound in x
  ///
  float64 get_xmin()
  {
    return xlim[0];
  }

  ///
  /// @brief get upper bound in x
  ///
  float64 get_xmax()
  {
    return xlim[1];
  }

  ///
  /// @brief get lower bound in y
  ///
  float64 get_ymin()
  {
    return ylim[0];
  }

  ///
  /// @brief get upper bound in y
  ///
  float64 get_ymax()
  {
    return ylim[1];
  }

  ///
  /// @brief get lower bound in z
  ///
  float64 get_zmin()
  {
    return zlim[0];
  }

  ///
  /// @brief get upper bound in z
  ///
  float64 get_zmax()
  {
    return zlim[1];
  }

protected:
  ///
  /// @brief pack and start boundary exchange
  /// @tparam Halo boundary halo class
  /// @param mpibuf MIP buffer
  /// @param halo boundary halo object
  ///
  template <class Halo>
  void begin_bc_exchange(PtrMpiBuffer mpibuf, Halo& halo);

  ///
  /// @brief wait boundary exchange and unpack
  /// @tparam Halo boundary halo class
  /// @param mpibuf MIP buffer
  /// @param halo boundary halo object
  ///
  template <class Halo>
  void end_bc_exchange(PtrMpiBuffer mpibuf, Halo& halo);

  ///
  /// @brief pack diagnostic for load array
  /// @param buffer pointer to buffer to pack
  /// @param address first address of buffer to which the data will be packed
  /// @return `address` + (number of bytes packed as a result)
  ///
  int pack_diagnostic_load(void* buffer, int address);

  ///
  /// @brief pack diagnostic for coordinate array
  /// @param buffer pointer to buffer to pack
  /// @param address first address of buffer to which the data will be packed
  /// @param dir direction of coordinate
  /// @return `address` + (number of bytes packed as a result)
  ///
  int pack_diagnostic_coord(void* buffer, int address, int dir);

  ///
  /// @brief pack diagnostic for field quantity
  /// @tparam T typename for field array
  /// @param buffer pointer to buffer to pack
  /// @param address first address of buffer to which the data will be packed
  /// @param u field quantity to be packed
  /// @return `address` + (number of bytes packed as a result)
  ///
  template <typename T>
  int pack_diagnostic_field(void* buffer, int address, T& u);

  ///
  /// @brief pack diagnostic for particle (single species)
  /// @param buffer pointer to buffer to pack
  /// @param address first address of buffer to which the data will be packed
  /// @param p particle species
  /// @return `address` + (number of bytes packed as a result)
  ///
  int pack_diagnostic_particle(void* buffer, int address, PtrParticle p);
};

//
// implementation follows
//
#define DEFINE_MEMBER(type, name)                                                                  \
  template <int Nb>                                                                                \
  type Chunk3D<Nb>::name

DEFINE_MEMBER(, Chunk3D)
(const int dims[3], int id)
    : Chunk<3>(dims, id), delx(1.0), dely(1.0), delz(1.0), require_sort(true)
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

DEFINE_MEMBER(int, pack)(void* buffer, int address)
{
  int count = address;

  count += Chunk<3>::pack(buffer, count);
  count += memcpy_count(buffer, xc.data(), xc.size() * sizeof(float64), count, 0);
  count += memcpy_count(buffer, yc.data(), yc.size() * sizeof(float64), count, 0);
  count += memcpy_count(buffer, zc.data(), zc.size() * sizeof(float64), count, 0);
  count += memcpy_count(buffer, &delx, sizeof(float64), count, 0);
  count += memcpy_count(buffer, &dely, sizeof(float64), count, 0);
  count += memcpy_count(buffer, &delz, sizeof(float64), count, 0);
  count += memcpy_count(buffer, xlim, 3 * sizeof(float64), count, 0);
  count += memcpy_count(buffer, ylim, 3 * sizeof(float64), count, 0);
  count += memcpy_count(buffer, zlim, 3 * sizeof(float64), count, 0);
  count += memcpy_count(buffer, gxlim, 3 * sizeof(float64), count, 0);
  count += memcpy_count(buffer, gylim, 3 * sizeof(float64), count, 0);
  count += memcpy_count(buffer, gzlim, 3 * sizeof(float64), count, 0);
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

DEFINE_MEMBER(int, unpack)(void* buffer, int address)
{
  int count = address;

  count += Chunk<3>::unpack(buffer, count);
  count += memcpy_count(xc.data(), buffer, xc.size() * sizeof(float64), 0, count);
  count += memcpy_count(yc.data(), buffer, yc.size() * sizeof(float64), 0, count);
  count += memcpy_count(zc.data(), buffer, zc.size() * sizeof(float64), 0, count);
  count += memcpy_count(&delx, buffer, sizeof(float64), 0, count);
  count += memcpy_count(&dely, buffer, sizeof(float64), 0, count);
  count += memcpy_count(&delz, buffer, sizeof(float64), 0, count);
  count += memcpy_count(xlim, buffer, 3 * sizeof(float64), 0, count);
  count += memcpy_count(ylim, buffer, 3 * sizeof(float64), 0, count);
  count += memcpy_count(zlim, buffer, 3 * sizeof(float64), 0, count);
  count += memcpy_count(gxlim, buffer, 3 * sizeof(float64), 0, count);
  count += memcpy_count(gylim, buffer, 3 * sizeof(float64), 0, count);
  count += memcpy_count(gzlim, buffer, 3 * sizeof(float64), 0, count);
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

DEFINE_MEMBER(void, set_global_context)(const int* offset, const int* gdims)
{
  this->gdims[0]  = gdims[0];
  this->gdims[1]  = gdims[1];
  this->gdims[2]  = gdims[2];
  this->offset[0] = offset[0];
  this->offset[1] = offset[1];
  this->offset[2] = offset[2];

  // local domain
  zlim[0] = offset[0] * delz;
  zlim[1] = offset[0] * delz + dims[0] * delz;
  zlim[2] = zlim[1] - zlim[0];
  ylim[0] = offset[1] * dely;
  ylim[1] = offset[1] * dely + dims[1] * dely;
  ylim[2] = ylim[1] - ylim[0];
  xlim[0] = offset[2] * delx;
  xlim[1] = offset[2] * delx + dims[2] * delx;
  xlim[2] = xlim[1] - xlim[0];

  // local coordinate
  zc = zlim[0] + delz * (xt::arange<float64>(Lbz - Nb, Ubz + Nb + 1) - Lbz + 0.5);
  yc = ylim[0] + dely * (xt::arange<float64>(Lby - Nb, Uby + Nb + 1) - Lby + 0.5);
  xc = xlim[0] + delx * (xt::arange<float64>(Lbx - Nb, Ubx + Nb + 1) - Lbx + 0.5);

  // global domain
  gzlim[0] = 0.0;
  gzlim[1] = gdims[0] * delz;
  gzlim[2] = gzlim[1] - gzlim[0];
  gylim[0] = 0.0;
  gylim[1] = gdims[1] * dely;
  gylim[2] = gylim[1] - gylim[0];
  gxlim[0] = 0.0;
  gxlim[1] = gdims[2] * delx;
  gxlim[2] = gxlim[1] - gxlim[0];
}

DEFINE_MEMBER(void, set_mpi_communicator)(int mode, MPI_Comm& comm)
{
  if (mode >= 0 && mode < mpibufvec.size()) {
    mpibufvec[mode]->comm = comm;
  } else {
    ERRORPRINT("invalid index %d for mpibufvec\n", mode);
  }
}

DEFINE_MEMBER(void, count_particle)(PtrParticle particle, int Lbp, int Ubp, bool reset)
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
    rdh[0]    = 1 / delz;
    rdh[1]    = 1 / dely;
    rdh[2]    = 1 / delx;
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
  float64*  xu            = particle->xu.data();

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

DEFINE_MEMBER(void, sort_particle)(ParticleVec& particle)
{
  for (int is = 0; is < particle.size(); is++) {
    count_particle(particle[is], 0, particle[is]->Np - 1, true);
    particle[is]->sort();
  }
}

DEFINE_MEMBER(void, inject_particle)(ParticleVec& particle)
{
}

DEFINE_MEMBER(bool, set_boundary_query)(int mode)
{
  int  flag   = 0;
  int  bcmode = mode;
  bool send   = (mode & SendMode) == SendMode; // send flag
  bool recv   = (mode & RecvMode) == RecvMode; // recv flag

  // remove send/recv bits
  bcmode &= ~SendMode;
  bcmode &= ~RecvMode;

  // MPI buffer
  PtrMpiBuffer mpibuf = mpibufvec[bcmode];

#pragma omp critical
  if (send == true && recv == true) {
    // both send/recv
    MPI_Testall(27, mpibuf->sendreq.data(), &flag, MPI_STATUSES_IGNORE);
    MPI_Testall(27, mpibuf->recvreq.data(), &flag, MPI_STATUSES_IGNORE);
  } else if (send == true) {
    // send
    MPI_Testall(27, mpibuf->sendreq.data(), &flag, MPI_STATUSES_IGNORE);
  } else if (recv == true) {
    // recv
    MPI_Testall(27, mpibuf->recvreq.data(), &flag, MPI_STATUSES_IGNORE);
  }

  return !(flag == 0);
}

DEFINE_MEMBER(void, set_boundary_physical)(int mode)
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
  // NOTE: trick to take care of round-off error
  float64 xlength = gxlim[2] - std::numeric_limits<float64>::epsilon();
  float64 ylength = gylim[2] - std::numeric_limits<float64>::epsilon();
  float64 zlength = gzlim[2] - std::numeric_limits<float64>::epsilon();

  // push particle position
  for (int ip = Lbp; ip <= Ubp; ip++) {
    float64* xu = &particle->xu(ip, 0);

    // apply periodic boundary condition
    xu[0] += (xu[0] < gxlim[0]) * xlength - (xu[0] >= gxlim[1]) * xlength;
    xu[1] += (xu[1] < gylim[0]) * ylength - (xu[1] >= gylim[1]) * ylength;
    xu[2] += (xu[2] < gzlim[0]) * zlength - (xu[2] >= gzlim[1]) * zlength;
  }
}

DEFINE_MEMBER(void, set_mpi_buffer)
(PtrMpiBuffer mpibuf, int mode, int headbyte, int elembyte)
{
  int size = 0;

  for (int iz = 0; iz < 3; iz++) {
    for (int iy = 0; iy < 3; iy++) {
      for (int ix = 0; ix < 3; ix++) {
        if (iz == 1 && iy == 1 && ix == 1) {
          mpibuf->bufsize(iz, iy, ix) = 0;
          mpibuf->bufaddr(iz, iy, ix) = size;
        } else {
          int nz = recvub[0][iz] - recvlb[0][iz] + 1;
          int ny = recvub[1][iy] - recvlb[1][iy] + 1;
          int nx = recvub[2][ix] - recvlb[2][ix] + 1;

          mpibuf->bufsize(iz, iy, ix) = headbyte + elembyte * nz * ny * nx;
          mpibuf->bufaddr(iz, iy, ix) = size;
          size += mpibuf->bufsize(iz, iy, ix);
        }
      }
    }
  }

  // buffer allocation
  if (mode == +1 || mode == 0) {
    mpibuf->sendbuf.resize(size);
  }
  if (mode == -1 || mode == 0) {
    mpibuf->recvbuf.resize(size);
  }
}

DEFINE_MEMBER(void, set_mpi_buffer)
(PtrMpiBuffer mpibuf, int mode, int headbyte, const int sizebyte[3][3][3])
{
  // buffer size
  int size = 0;

  for (int iz = 0; iz < 3; iz++) {
    for (int iy = 0; iy < 3; iy++) {
      for (int ix = 0; ix < 3; ix++) {
        if (iz == 1 && iy == 1 && ix == 1) {
          mpibuf->bufsize(iz, iy, ix) = 0;
          mpibuf->bufaddr(iz, iy, ix) = size;
        } else {
          mpibuf->bufsize(iz, iy, ix) = headbyte + sizebyte[iz][iy][ix];
          mpibuf->bufaddr(iz, iy, ix) = size;
          size += mpibuf->bufsize(iz, iy, ix);
        }
      }
    }
  }

  // buffer allocation
  if (mode == +1 || mode == 0) {
    mpibuf->sendbuf.resize(size);
  }
  if (mode == -1 || mode == 0) {
    mpibuf->recvbuf.resize(size);
  }
}

DEFINE_MEMBER(template <class Halo> void, begin_bc_exchange)
(PtrMpiBuffer mpibuf, Halo& halo)
{
  // pre-process
  halo.pre_pack(mpibuf);

  for (int dirz = -1, iz = 0; dirz <= +1; dirz++, iz++) {
    for (int diry = -1, iy = 0; diry <= +1; diry++, iy++) {
      for (int dirx = -1, ix = 0; dirx <= +1; dirx++, ix++) {
        // clang-format off
        int send_bound[3][2] = {
          sendlb[0][iz], sendub[0][iz],
          sendlb[1][iy], sendub[1][iy],
          sendlb[2][ix], sendub[2][ix]
        };
        int recv_bound[3][2] = {
          recvlb[0][iz], recvub[0][iz],
          recvlb[1][iy], recvub[1][iy],
          recvlb[2][ix], recvub[2][ix]
        };
        // clang-format on

        // pack
        bool status = halo.pack(mpibuf, iz, iy, ix, send_bound, recv_bound);

#pragma omp critical
        if (status) {
          int   nbrank  = get_nb_rank(dirz, diry, dirx);
          int   sendtag = get_sndtag(dirz, diry, dirx);
          int   recvtag = get_rcvtag(dirz, diry, dirx);
          void* sendptr = halo.send_buffer;
          void* recvptr = halo.recv_buffer;
          int   sendcnt = halo.send_count;
          int   recvcnt = halo.recv_count;

          // send/recv calls
          MPI_Isend(sendptr, sendcnt, mpibuf->sendtype(iz, iy, ix), nbrank, sendtag, mpibuf->comm,
                    &mpibuf->sendreq(iz, iy, ix));
          MPI_Irecv(recvptr, recvcnt, mpibuf->recvtype(iz, iy, ix), nbrank, recvtag, mpibuf->comm,
                    &mpibuf->recvreq(iz, iy, ix));
        } else {
          // no send/recv required
          mpibuf->sendreq(iz, iy, ix) = MPI_REQUEST_NULL;
          mpibuf->recvreq(iz, iy, ix) = MPI_REQUEST_NULL;
        }
      }
    }
  }

  // post-process
  halo.post_pack(mpibuf);
}

DEFINE_MEMBER(template <class Halo> void, end_bc_exchange)
(PtrMpiBuffer mpibuf, Halo& halo)
{
#pragma omp critical
  {
    // wait for MPI recv calls to complete
    MPI_Waitall(27, mpibuf->recvreq.data(), MPI_STATUSES_IGNORE);
  }

  // pre-process
  halo.pre_unpack(mpibuf);

  //
  // unpack recv buffer
  //
  for (int dirz = -1, iz = 0; dirz <= +1; dirz++, iz++) {
    for (int diry = -1, iy = 0; diry <= +1; diry++, iy++) {
      for (int dirx = -1, ix = 0; dirx <= +1; dirx++, ix++) {
        // skip physical boundary
        if (get_nb_rank(dirz, diry, dirx) == MPI_PROC_NULL)
          continue;

        // clang-format off
        int send_bound[3][2] = {
          sendlb[0][iz], sendub[0][iz],
          sendlb[1][iy], sendub[1][iy],
          sendlb[2][ix], sendub[2][ix]
        };
        int recv_bound[3][2] = {
          recvlb[0][iz], recvub[0][iz],
          recvlb[1][iy], recvub[1][iy],
          recvlb[2][ix], recvub[2][ix]
        };
        // clang-format on

        // unpack
        bool status = halo.unpack(mpibuf, iz, iy, ix, send_bound, recv_bound);
      }
    }
  }

  // post-proces
  halo.post_unpack(mpibuf);

#pragma omp critical
  {
    // wait for MPI send calls to complete
    MPI_Waitall(27, mpibuf->sendreq.data(), MPI_STATUSES_IGNORE);
  }
}

DEFINE_MEMBER(int, pack_diagnostic_load)(void* buffer, int address)
{
  int count = sizeof(float64) * load.size() + address;

  if (buffer == nullptr) {
    return count;
  }

  float64* ptr = reinterpret_cast<float64*>(static_cast<uint8_t*>(buffer) + address);
  std::copy(load.begin(), load.end(), ptr);

  return count;
}

DEFINE_MEMBER(int, pack_diagnostic_coord)(void* buffer, int address, int dir)
{
  size_t size  = dims[dir];
  int    count = sizeof(float64) * size + address;

  if (buffer == nullptr) {
    return count;
  }

  float64* ptr = reinterpret_cast<float64*>(static_cast<uint8_t*>(buffer) + address);

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

DEFINE_MEMBER(template <typename T> int, pack_diagnostic_field)
(void* buffer, int address, T& u)
{
  // calculate number of elements
  size_t size = dims[0] * dims[1] * dims[2];
  for (int i = 3; i < u.dimension(); i++) {
    size *= u.shape(i);
  }

  int count = sizeof(float64) * size + address;

  if (buffer == nullptr) {
    return count;
  }

  auto Iz = xt::range(Lbz, Ubz + 1);
  auto Iy = xt::range(Lby, Uby + 1);
  auto Ix = xt::range(Lbx, Ubx + 1);
  auto vv = xt::strided_view(u, {Iz, Iy, Ix, xt::ellipsis()});

  // packing
  float64* ptr = reinterpret_cast<float64*>(static_cast<uint8_t*>(buffer) + address);
  std::copy(vv.begin(), vv.end(), ptr);

  return count;
}

DEFINE_MEMBER(int, pack_diagnostic_particle)
(void* buffer, int address, PtrParticle p)
{
  int count = address;

  count += memcpy_count(buffer, p->xu.data(), p->Np * Particle::Nc * sizeof(float64), count, 0);

  return count;
}

#undef DEFINE_MEMBER

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
