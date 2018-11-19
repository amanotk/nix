// -*- C++ -*-

///
/// PatchMap test code
///
/// $Id$
///
#include "patchmap.hpp"
#include "cmdline.hpp"

int main(int argc, char **argv)
{
  // parse command line arguments
  cmdline::parser parser;

  parser.add<int>("nx", 'x', "#Patch in x direction", false, 0);
  parser.add<int>("ny", 'y', "#Patch in y direction", false, 0);
  parser.add<int>("nz", 'z', "#Patch in z direction", false, 0);
  parser.add<std::string>("output", 'o', "output filename", true);

  parser.parse_check(argc, argv);

  // test PatchMap
  int nz = parser.get<int>("nz");
  int ny = parser.get<int>("ny");
  int nx = parser.get<int>("nx");
  int dims[3] = {nz, ny, nx};

  BasePatchMap patchmap(dims);

  std::ofstream f(parser.get<std::string>("output").c_str());

  patchmap.debug(f);

  f.close();

  return 0;
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
