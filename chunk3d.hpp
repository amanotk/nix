// -*- C++ -*-
#ifndef _Chunk3D_HPP_
#define _Chunk3D_HPP_

#include "buffer.hpp"
#include "chunk.hpp"
#include "debug.hpp"
#include "jsonio.hpp"
#include "xtensorall.hpp"

template <int Nb>
class Chunk3D : public Chunk<3>
{
public:
  using T_bufaddr = xt::xtensor_fixed<int, xt::xshape<3, 3, 3>>;

  enum PackMode {
    PackAll = 1,
    PackAllQuery,
  };

  /// boundary margin
  static const int boundary_margin = Nb;

protected:
  int Lbx;          ///< lower bound in x
  int Ubx;          ///< upper bound in x
  int Lby;          ///< lower bound in y
  int Uby;          ///< upper bound in y
  int Lbz;          ///< lower bound in z
  int Ubz;          ///< upper bound in z
  int sendlb[3][3]; ///< lower bound for send
  int sendub[3][3]; ///< upper bound for send
  int recvlb[3][3]; ///< lower bound for recv
  int recvub[3][3]; ///< upper bound for recv

  xt::xtensor<float64, 1> xc;      ///< x coordiante
  xt::xtensor<float64, 1> yc;      ///< y coordiante
  xt::xtensor<float64, 1> zc;      ///< z coordiante
  float64                 delh;    ///< grid size
  float64                 xlim[3]; ///< physical domain in x
  float64                 ylim[3]; ///< physical domain in y
  float64                 zlim[3]; ///< physical domain in z

  MPI_Request sendreq[3][3][3]; ///< MPI request
  MPI_Request recvreq[3][3][3]; ///< MPI request
  size_t      bufsize;          // MPI buffer size
  T_bufaddr   bufaddr;          // MPI buffer address array
  Buffer      sendbuf;          ///< MPI send buffer
  Buffer      recvbuf;          ///< MPI recv buffer

public:
  Chunk3D(const int dims[3], const int id = 0);

  virtual ~Chunk3D() override;

  virtual void initialize_load() override;

  virtual float64 get_load() override;

  virtual int pack(const int mode, void *buffer) override;

  virtual int unpack(const int mode, void *buffer) override;

  virtual void set_buffer_address();

  virtual void set_coordinate(const float64 delh, const int offset[3]);

  virtual void push(const float64 delt) = 0;

  virtual void set_boundary_begin(const int mode) = 0;

  virtual void set_boundary_end(const int mode) = 0;

  virtual bool set_boundary_query(const int mode) = 0;
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
