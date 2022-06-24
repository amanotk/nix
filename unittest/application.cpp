// -*- C++ -*-

///
/// Pseudo Code
///
/// $Id$
///
#include "application.hpp"
#include "chunk.hpp"
#include "chunkmap.hpp"

typedef BaseApplication<BaseChunk<3>,BaseChunkMap> T_app;


int main(int argc, char **argv)
{
  T_app app(argc, argv);
  app.main(std::cout);
  return 0;
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
