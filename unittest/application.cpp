// -*- C++ -*-

///
/// Pseudo Code
///
/// $Id$
///
#include "application.hpp"
#include "chunk.hpp"
#include "chunkmap.hpp"

using Chunk    = BaseChunk<3>;
using ChunkMap = BaseChunkMap<3>;
using T_app    = BaseApplication<Chunk, ChunkMap>;

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
