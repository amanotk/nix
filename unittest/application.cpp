// -*- C++ -*-

///
/// Pseudo Code
///
/// $Id$
///
#include "application.hpp"
#include "patch.hpp"
#include "patchmap.hpp"

typedef BaseApplication<BasePatch<3>,BasePatchMap> T_app;


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
