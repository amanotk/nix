// -*- C++ -*-

#include "maxwell.hpp"

using Application = Maxwell;

//
// main
//
int main(int argc, char **argv)
{
  Application app(argc, argv);
  return app.main(std::cout);
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
