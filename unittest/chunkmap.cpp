// -*- C++ -*-

///
/// ChunkMap test code
///
/// $Id$
///
#include "chunkmap.hpp"
#include "cmdline.hpp"

int main(int argc, char **argv)
{
  // parse command line arguments
  cmdline::parser parser;

  parser.add<int>("nx", 'x', "#Chunk in x direction", false, 0);
  parser.add<int>("ny", 'y', "#Chunk in y direction", false, 0);
  parser.add<int>("nz", 'z', "#Chunk in z direction", false, 0);
  parser.add<std::string>("output", 'o', "output filename", true);

  parser.parse_check(argc, argv);

  // test ChunkMap
  int nz = parser.get<int>("nz");
  int ny = parser.get<int>("ny");
  int nx = parser.get<int>("nx");
  int dims[3] = {nz, ny, nx};

  BaseChunkMap chunkmap(dims);

  std::ofstream f(parser.get<std::string>("output").c_str());

  chunkmap.debug(f);

  f.close();

  return 0;
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
