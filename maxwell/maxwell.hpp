// -*- C++ -*-
#ifndef _MAXWELL_HPP_
#define _MAXWELL_HPP_

#include "../application.hpp"
#include "../chunkmap.hpp"
#include "../jsonio.hpp"
#include "fdtd.hpp"

using BaseApp = Application<FDTD, ChunkMap<3>>;

//
// Maxwell
//
class Maxwell : public BaseApp
{
protected:
  using Chunk = FDTD;

  std::string prefix;   ///< output filename prefix
  int         interval; ///< data output interval
  int         kdir;     ///< wave propagation direction (for initial condition)
  float64     cc;       ///< speed of light

public:
  Maxwell(int argc, char **argv) : BaseApp(argc, argv)
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
