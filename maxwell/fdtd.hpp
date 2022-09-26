// -*- C++ -*-
#ifndef _FDTD_HPP_
#define _FDTD_HPP_

#include "../buffer.hpp"
#include "../chunk3d.hpp"
#include "../debug.hpp"
#include "../jsonio.hpp"
#include "../xtensorall.hpp"

using namespace nix;

class FDTD : public Chunk3D<1>
{
public:
  using Chunk      = Chunk3D<1>;
  using T_function = std::function<void(float64, float64, float64, float64 *)>;

  /// boundary margin
  static constexpr int Nb = Chunk::boundary_margin;

protected:
  xt::xtensor<float64, 4> uf; ///< electromagnetic field
  float64                 cc; ///< speed of light

public:
  FDTD(const int dims[3], const int id = 0);

  virtual ~FDTD() override;

  virtual int pack(void *buffer, const int address) override;

  virtual int unpack(void *buffer, const int address) override;

  virtual void setup(json &config) override;

  virtual void push(const float64 delt);

  virtual int pack_diagnostic(int mode, void *buffer, const int address);

  virtual void set_boundary_begin(const int mode = 0) override;

  virtual void set_boundary_end(const int mode = 0) override;
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
