// -*- C++ -*-
#ifndef _CHUNK3D_HPP_
#define _CHUNK3D_HPP_

#include "buffer.hpp"
#include "chunk.hpp"
#include "debug.hpp"
#include "jsonio.hpp"
#include "particle.hpp"
#include "xtensorall.hpp"

template <int Nb>
class Chunk3D : public Chunk<3>
{
public:
  using T_bufaddr = xt::xtensor_fixed<int, xt::xshape<3, 3, 3>>;
  using T_request = xt::xtensor_fixed<MPI_Request, xt::xshape<3, 3, 3>>;

  enum SendRecvMode {
    SendMode = 0b01000000000000, // 4096
    RecvMode = 0b10000000000000, // 8192
  };

  enum PackMode {
    PackAllQuery = 0,
    PackAll      = 1,
  };

  /// MPI buffer struct
  struct MpiBuffer {
    MPI_Comm  comm;
    size_t    bufsize;
    Buffer    sendbuf;
    Buffer    recvbuf;
    T_bufaddr bufaddr;
    T_request sendreq;
    T_request recvreq;
  };
  typedef std::vector<std::unique_ptr<MpiBuffer>> MpiBufferVec;

  /// boundary margin
  static const int boundary_margin = Nb;

protected:
  bool require_sort; ///< sort flag
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

  void count_particle(ParticleList &particle, int *Lbp, int *Ubp, bool reset = true);

  void begin_bc_exchange(MpiBuffer *mpibuf, xt::xtensor<float64, 4> &array);

  void begin_bc_exchange(MpiBuffer *mpibuf, ParticleList &particle);

  void end_bc_exchange(MpiBuffer *mpibuf, xt::xtensor<float64, 4> &array, bool append = false);

  void end_bc_exchange(MpiBuffer *mpibuf, ParticleList &particle);

  template <typename T>
  void set_mpi_buffer(const int headbyte, const T &elembyte, MpiBuffer *mpibuffer)
  {
    const std::vector<size_t> shape = {3, 3};

    auto I   = xt::all();
    auto J   = xt::newaxis();
    auto xlb = xt::adapt(&recvlb[0][0], 9, xt::no_ownership(), shape);
    auto xub = xt::adapt(&recvub[0][0], 9, xt::no_ownership(), shape);
    auto xss = xub - xlb + 1;
    auto pos =
        xt::eval(xt::view(xss, 0, I, J, J) * xt::view(xss, 1, J, I, J) * xt::view(xss, 2, J, J, I));

    // no send/recv with itself
    pos          = pos * elembyte;
    pos(1, 1, 1) = 0;

    mpibuffer->bufsize = headbyte + xt::sum(pos)(); // buffer size

    // calculate buffer address and size
    pos    = xt::cumsum(pos);
    pos    = xt::roll(pos, 1);
    pos(0) = 0;
    pos.reshape({3, 3, 3});

    mpibuffer->bufaddr = headbyte + pos; // buffer address

    // buffer allocation
    mpibuffer->sendbuf.resize(mpibuffer->bufsize);
    mpibuffer->recvbuf.resize(mpibuffer->bufsize);

    // default communicator
    mpibuffer->comm = MPI_COMM_WORLD;
  }

public:
  Chunk3D(const int dims[3], const int id = 0);

  virtual ~Chunk3D() override;

  virtual void initialize_load() override;

  virtual float64 get_load() override;

  virtual int pack(const int mode, void *buffer) override;

  virtual int unpack(const int mode, void *buffer) override;

  virtual void set_coordinate(const float64 delh, const int offset[3]);

  virtual bool set_boundary_query(const int mode = 0);

  virtual void set_boundary_physical(const int mode = 0);

  virtual void push(const float64 delt) = 0;

  virtual void set_boundary_begin(const int mode) = 0;

  virtual void set_boundary_end(const int mode) = 0;
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
