// -*- C++ -*-
#ifndef _FDTD_HPP_
#define _FDTD_HPP_

#include "pk3/buffer.hpp"
#include "pk3/chunk.hpp"
#include "pk3/debug.hpp"
#include "pk3/jsonio.hpp"
#include "pk3/xtensorall.hpp"

#include <functional>

class FDTD : public BaseChunk<3>
{
public:
  using T_function = std::function<void(float64, float64, float64, float64 *)>;
  enum PackMode {
    PackAll = 1,
    PackAllQuery,
    PackEmf,
    PackEmfQuery,
  };

protected:
  static const int Nb = 1; ///< boundary margin
  int              Lbx;    ///< lower bound in x
  int              Ubx;    ///< upper bound in x
  int              Lby;    ///< lower bound in y
  int              Uby;    ///< upper bound in y
  int              Lbz;    ///< lower bound in z
  int              Ubz;    ///< upper bound in z

  xt::xtensor<float64, 1> xc;      ///< x coordiante
  xt::xtensor<float64, 1> yc;      ///< y coordiante
  xt::xtensor<float64, 1> zc;      ///< z coordiante
  xt::xtensor<float64, 4> uf;      ///< electromagnetic field
  float64                 cc;      ///< speed of light
  float64                 delh;    ///< grid size
  float64                 xlim[3]; ///< physical domain in x
  float64                 ylim[3]; ///< physical domain in y
  float64                 zlim[3]; ///< physical domain in z

  MPI_Request sendreq[3][2]; ///< MPI request
  MPI_Request recvreq[3][2]; ///< MPI request
  size_t      bufsize[4];    ///< MPI buffer size
  Buffer      sendbuf;       ///< MPI send buffer
  Buffer      recvbuf;       ///< MPI recv buffer

public:
  FDTD(const int dims[3], const int id = 0);

  virtual ~FDTD() override;

  virtual void initialize_load() override;

  virtual float64 get_load() override;

  virtual int pack(const int mode, void *buffer) override;

  virtual int unpack(const int mode, void *buffer) override;

  virtual void setup(const float64 cc, const float64 delh, const int offset[3],
                     T_function initializer);

  virtual void push(const float64 delt);

  virtual int pack_diagnostic(void *buffer, const bool query);

  virtual void set_boundary_begin();

  virtual void set_boundary_end();

  virtual bool set_boundary_query(const int mode);

  virtual void set_boundary_physical(const int dir);

  virtual void set_boundary();
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
