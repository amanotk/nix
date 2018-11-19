// -*- C++ -*-

///
/// Hilbert Curve Test Code
///
/// $Id$
///
#include <iostream>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <bitset>
#include <fstream>
#include "tinyformat.hpp"
#include "hilbert.hpp"

int main()
{
  typedef HilbertCurve<2> HC2;
  typedef HilbertCurve<3> HC3;

  const int order = 4;
  const char *fn2 = "hilbert2d.dat";
  const char *fn3 = "hilbert3d.dat";

  // debug 2D and 3D hilbert curve
  HC2::debug();
  HC3::debug();


  // generate 2D hilbert curve datafile
  {
    const int entry     = 0;
    const int direction = 0;

    std::ofstream ofs(fn2);

    for(int i=0; i < 1 << (2*order); i++) {
      int coord[2];
      HC2::get_coordinate(i, entry, direction, order, coord);

      ofs << tfm::format("%3d, %3d\n", coord[0], coord[1]);
    }

    ofs.close();
  }


  // generate 3D hilbert curve datafile
  {
    const int entry     = 0;
    const int direction = 0;

    std::ofstream ofs(fn3);

    for(int i=0; i < 1 << (3*order); i++) {
      int coord[3];
      HC3::get_coordinate(i, entry, direction, order, coord);

      ofs << tfm::format("%3d, %3d, %3d\n", coord[0], coord[1], coord[2]);
    }

    ofs.close();
  }

  return 0;
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
