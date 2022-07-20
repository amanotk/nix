// -*- C++ -*-

#include "maxwell.hpp"

class Application : public Maxwell
{
public:
  Application(int argc, char **argv) : Maxwell(argc, argv)
  {
  }

  virtual void initializer(float64 z, float64 y, float64 x, float64 *eb) override
  {
    float64 kk = common::pi2 / xlim[2];
    float64 ff = cos(kk * x);
    float64 gg = sin(kk * x);

    eb[0] = 0;
    eb[1] = ff;
    eb[2] = gg;
    eb[3] = 0;
    eb[4] = gg;
    eb[5] = ff;
  }
};

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
