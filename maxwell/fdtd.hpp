// -*- C++ -*-
#ifndef _FDTD_HPP_
#define _FDTD_HPP_

#include "pk3/buffer.hpp"
#include "pk3/chunk3d.hpp"
#include "pk3/debug.hpp"
#include "pk3/jsonio.hpp"
#include "pk3/xtensorall.hpp"

class FDTD : public Chunk3D<1>
{
public:
  using Chunk      = Chunk3D<1>;
  using T_function = std::function<void(float64, float64, float64, float64 *)>;

  enum PackMode {
    PackAll = 1,
    PackAllQuery,
    PackEmf,
    PackEmfQuery,
  };

  /// boundary margin
  static const int Nb = Chunk::boundary_margin;

protected:
  xt::xtensor<float64, 4> uf; ///< electromagnetic field
  float64                 cc; ///< speed of light

public:
  FDTD(const int dims[3], const int id = 0);

  virtual ~FDTD() override;

  virtual int pack(const int mode, void *buffer) override;

  virtual int unpack(const int mode, void *buffer) override;

  virtual void setup(const float64 cc, const float64 delh, const int offset[3],
                     T_function initializer);

  virtual void push(const float64 delt) override;

  virtual int pack_diagnostic(void *buffer, const bool query);

  virtual void set_boundary_begin() override;

  virtual void set_boundary_end() override;

  virtual bool set_boundary_query(const int mode) override;

  virtual void set_boundary_physical(const int dir) override;
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
