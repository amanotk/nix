// -*- C++ -*-

///
/// Code Description
///
/// $Id$
///
#include <iostream>
#include "patch.hpp"
#include "patchmap.hpp"

int main()
{
  const int N = 3;
  const int patch_shape[N] = {4, 4, 4};
  const int patch_dims[N] = {2, 4, 6};
  BasePatch<N> p(0, patch_shape);
  BasePatchMap map(patch_dims);

  for(int dirz=-1; dirz <= +1 ;dirz++) {
    for(int diry=-1; diry <= +1 ;diry++) {
      for(int dirx=-1; dirx <= +1 ;dirx++) {
        std::cout << std::bitset<31>(p.get_sndtag(dirz, diry, dirx, p.get_id()))
                  << std::endl;
      }
    }
  }

  return 0;
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
