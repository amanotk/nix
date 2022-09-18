// -*- C++ -*-
#ifndef _CHUNK3D_HPP_
#define _CHUNK3D_HPP_

#include "buffer.hpp"
#include "chunk.hpp"
#include "common.hpp"
#include "debug.hpp"
#include "jsonio.hpp"
#include "particle.hpp"
#include "xtensorall.hpp"

template <int Nb>
class Chunk3D : public Chunk<3>
{
public:
  using json      = common::json;
  using T_array3d = xt::xtensor_fixed<int, xt::xshape<3, 3, 3>>;
  using T_request = xt::xtensor_fixed<MPI_Request, xt::xshape<3, 3, 3>>;

  ///
  /// MPI buffer struct
  ///
  struct MpiBuffer {
    MPI_Comm  comm;
    Buffer    sendbuf;
    Buffer    recvbuf;
    T_array3d bufsize;
    T_array3d bufaddr;
    T_request sendreq;
    T_request recvreq;

    // constructor
    MpiBuffer() : comm(MPI_COMM_WORLD)
    {
    }

    // packing
    int pack(void *buffer, const int address)
    {
      using common::memcpy_count;

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

    // unpacking
    int unpack(void *buffer, const int address)
    {
      using common::memcpy_count;

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
  int  ndims[3];     ///< number of global grids
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

  xt::xtensor<float64, 1> xc;        ///< x coordiante
  xt::xtensor<float64, 1> yc;        ///< y coordiante
  xt::xtensor<float64, 1> zc;        ///< z coordiante
  float64                 delh;      ///< grid size
  float64                 xlim[3];   ///< physical domain in x
  float64                 ylim[3];   ///< physical domain in y
  float64                 zlim[3];   ///< physical domain in z
  MpiBufferVec            mpibufvec; ///< MPI buffer vector

  int pack_diagnostic_load(void *buffer, const int address);

  int pack_diagnostic_coord(void *buffer, const int address, const int dir);

  int pack_diagnostic_field(void *buffer, const int address, xt::xtensor<float64, 4> &u);

  void begin_bc_exchange(PtrMpiBuffer mpibuf, ParticleVec &particle);

  void end_bc_exchange(PtrMpiBuffer mpibuf, ParticleVec &particle);

  void begin_bc_exchange(PtrMpiBuffer mpibuf, xt::xtensor<float64, 4> &array, bool moment = false);

  void end_bc_exchange(PtrMpiBuffer mpibuf, xt::xtensor<float64, 4> &array, bool momet = false);

  template <typename T>
  void set_mpi_buffer(PtrMpiBuffer mpibuffer, const int headbyte, const T &elembyte)
  {
    const std::vector<size_t> shape = {3, 3};

    auto I   = xt::all();
    auto J   = xt::newaxis();
    auto xlb = xt::adapt(&recvlb[0][0], 9, xt::no_ownership(), shape);
    auto xub = xt::adapt(&recvub[0][0], 9, xt::no_ownership(), shape);
    auto xss = xub - xlb + 1;
    auto pos = xt::eval(xt::view(xss, 0, I, J, J) * xt::view(xss, 1, J, I, J) *
                        xt::view(xss, 2, J, J, I) * elembyte);

    // no send/recv with itself
    pos(1, 1, 1) = 0;

    // buffer allocation
    {
      int size = headbyte + xt::sum(pos)();
      mpibuffer->sendbuf.resize(size);
      mpibuffer->recvbuf.resize(size);
    }

    // buffer size
    mpibuffer->bufsize = pos;

    // buffer address
    pos    = xt::cumsum(pos);
    pos    = xt::roll(pos, 1);
    pos(0) = 0;
    pos.reshape({3, 3, 3});
    mpibuffer->bufaddr = headbyte + pos;
  }

public:
  Chunk3D(const int dims[3], const int id = 0);

  virtual ~Chunk3D() override;

  virtual void reset_load() override;

  virtual std::vector<float64> get_load() override;

  virtual float64 get_total_load() override;

  virtual int pack(void *buffer, const int address) override;

  virtual int unpack(void *buffer, const int address) override;

  virtual void set_global_context(const int *offset, const int *ndims);

  virtual void set_mpi_communicator(const int mode, MPI_Comm &comm);

  virtual void count_particle(PtrParticle particle, const int Lbp, const int Ubp,
                              bool reset = true);

  virtual void sort_particle(ParticleVec &particle);

  virtual bool set_boundary_query(const int mode = 0);

  virtual void set_boundary_physical(const int mode = 0);

  virtual void set_boundary_particle(PtrParticle particle, int Lbp, int Ubp);

  virtual void setup(json &config) = 0;

  virtual void set_boundary_begin(const int mode) = 0;

  virtual void set_boundary_end(const int mode) = 0;
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
