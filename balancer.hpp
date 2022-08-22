// -*- C++ -*-
#ifndef _BALANCER_HPP_
#define _BALANCER_HPP_

///
/// Balancer
///
/// $Id$
///
#include "common.hpp"

///
/// Definition of Balancer
///
class Balancer
{
public:
  /// calculate decomposition from load array
  virtual void partition(const int nc, const int nr, float64 load[], int rank[])
  {
    float64 boundary[nr + 1];
    float64 avgload, prv, cur, difc, difp;

    // calculate average load
    avgload = 0.0;
    for (int i = 0; i < nc; i++) {
      avgload += load[i] / nr;
    }

    // find boundaries for decomposition
    cur = 0.0;
    for (int i = 0, r = 0; i < nc; i++) {
      prv = cur;
      cur += load[i];
      difc = cur - avgload;
      difp = avgload - prv;

      if (difc < 0.0)
        continue;

      if (std::abs(difc) < std::abs(difp)) {
        boundary[r + 1] = i + 1;
        cur             = 0.0;
      } else {
        boundary[r + 1] = i;
        cur             = load[i];
      }
      r++;
    }
    boundary[0]  = 0;
    boundary[nr] = nc;

    // assign rankmap
    for (int r = 0; r < nr; r++) {
      for (int i = boundary[r]; i < boundary[r + 1]; i++) {
        rank[i] = r;
      }
    }
  }
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
