// -*- C++ -*-
#ifndef _MAXWELL_HPP_
#define _MAXWELL_HPP_

#include "fdtd.hpp"
#include "pk3/application.hpp"
#include "pk3/chunkmap.hpp"
#include "pk3/jsonio.hpp"

using ChunkMap = BaseChunkMap<3>;
using Base     = BaseApplication<FDTD, ChunkMap>;

//
// Maxwell
//
class Maxwell : public Base
{
protected:
  std::string prefix;   ///< output filename prefix
  int         interval; ///< data output interval
  int         kdir;     ///< wave propagation direction (for initial condition)
  float64     cc;       ///< speed of light

public:
  Maxwell(int argc, char **argv) : Base(argc, argv)
  {
  }

  virtual void initialize(int argc, char **argv) override;

  virtual void push() override;

  virtual void diagnostic() override;

  virtual void initializer(float64 z, float64 y, float64 x, float64 *eb);
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
