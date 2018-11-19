// -*- C++ -*-
#ifndef _HILBERT_HPP_
#define _HILBERT_HPP_

///
/// Hilbert Curve
///
/// $Id$
///
#include "common.hpp"

///
/// Hilbert Curve Generator
///
/// This implementation is due to the following references:
///
/// * Pierre de Buyl
///   http://pdebuyl.be/blog/2015/hilbert-curve.html
///
/// * Chris Hamilton
///   https://www.cs.dal.ca/research/techreports/cs-2006-07
///
template <int DIM>
class HilbertCurve
{
private:
  static int pow2(int n)
  {
    return 1 << n;
  }

  static int bit_at(int x, int i)
  {
    return (x & pow2(i)) >> i;
  }

  static int rotate_r(int x, int d=1)
  {
    int bit, out;

    d = d % DIM;

    out = x >> d;
    for(int i=0; i < d; i++) {
      bit = (x & pow2(i)) >> i;
      out = out | (bit << (DIM+i-d));
    }

    return out;
  }

  static int rotate_l(int x, int d=1)
  {
    int bit, out, exc;

    d = d % DIM;

    out = x << d;
    exc = out;
    out = out & pow2(DIM)-1;
    for(int i=0; i < d; i++) {
      bit = (x & pow2(DIM-1-d+i+1)) >> (DIM-1-d+i+1);
      out = out | (bit << i);
    }

    return out;
  }

  static int graycode(int x)
  {
    return x ^ (x >> 1);
  }

  static int inv_graycode(int x)
  {
    int i = x;

    for(int j=1; j < DIM; j++) {
      i = i ^ (x >> j);
    }
    return i;
  }

  static int inter_direction(int x)
  {
    return static_cast<int>(std::log2(graycode(x) ^ graycode(x+1)));
  }

  static int intra_direction(int x)
  {
    if( x == 0 ) {
      return 0;
    } else if ( x%2 == 0 ) {
      return inter_direction(x-1) % DIM;
    } else {
      return inter_direction(x) % DIM;
    }
  }

  static int transform(int e, int d, int b)
  {
    return rotate_r(b ^ e, d+1);
  }

  static int inv_transform(int e, int d, int b)
  {
    return transform(rotate_r(e, d+1), DIM-d-2, b);
  }

  static int entry_point(int x)
  {
    if( x == 0 ) {
      return 0;
    } else {
      return graycode(2*static_cast<int>(std::floor((x-1)/2)));
    }
  }

  static int exit_point(int x)
  {
    return entry_point(pow2(DIM)-1-x) ^ pow2(DIM-1);
  }

public:
  ///
  /// return Hilbert curve index associated with given cartesian coordiante
  ///
  static int get_index(int coord[DIM],
                       int entry, int direction, int order)
  {
    int ve = entry;
    int vd = direction;

    int index = 0;
    for(int i=order-1; i > -1 ;i--) {
      int l, w, ll[DIM];

      for(int j=0; j < DIM ;j++) {
        ll[j] = bit_at(coord[j], i);
      }

      l = 0;
      for(int j=0; j < DIM ;j++) {
        l += ll[j] * pow2(j);
      }

      l = transform(ve, vd, l);
      w = inv_graycode(l);

      ve = ve ^ rotate_l(entry_point(w), vd+1);
      vd = (vd + intra_direction(w) + 1) % DIM;

      index = (index << DIM) | w;
    }

    return index;
  }


  ///
  /// return cartesian coordiante for given Hilbert curve index
  ///
  static void get_coordinate(int x,
                             int entry, int direction, int order,
                             int coord[DIM])
  {
    int ve = entry;
    int vd = direction;

    for(int i=0; i < DIM ; i++) {
      coord[i] = 0;
    }

    for(int i=order-1; i > -1 ;i--) {
      int l, w, ww[DIM];

      for(int j=0; j < DIM; j++) {
        ww[j] = bit_at(x, i*DIM + j);
      }

      w = 0;
      for(int j=0; j < DIM; j++) {
        w += ww[j] * pow2(j);
      }

      l = graycode(w);
      l = inv_transform(ve, vd, l);

      for(int j=0; j < DIM; j++) {
        coord[j] += bit_at(l, j) << i;
      }

      ve = ve ^ rotate_l(entry_point(w), vd+1);
      vd = (vd + intra_direction(w) + 1) % DIM;
    }
  }


  ///
  /// debugging function
  ///
  static void debug()
  {
    std::cout << tfm::format("\n*** "
                             "Checking HilbertCurve<%1d> routines...\n\n",
                             DIM);

    // check bit_at
    {
      std::random_device rd;
      std::mt19937 mt(rd());

      int x = mt();

      std::cout << "--- bit_at ---" << std::endl;

      std::cout << tfm::format("bits : %6s <===> ", std::bitset<DIM>(x));
      for(int i=0; i < DIM ;i++) {
        std::cout << tfm::format("%1d ", bit_at(x, DIM-i-1));
      }
      std::cout << std::endl;
    }

    // rotate
    {
      int x;

      std::cout << "--- rotate ---" << std::endl;

      // right
      x = 2;
      std::cout << "right : ";
      for(int i=0; i < DIM ;i++) {
        std::cout << tfm::format("%6s =>", std::bitset<DIM>(x));
        x = rotate_r(x);
      }
      std::cout << tfm::format("%6s\n", std::bitset<DIM>(x));

      // left
      x = 2;
      std::cout << "left  : ";
      for(int i=0; i < DIM ;i++) {
        std::cout << tfm::format("%6s =>", std::bitset<DIM>(x));
        x = rotate_l(x);
      }
      std::cout << tfm::format("%6s\n", std::bitset<DIM>(x));
    }

    // graycode, entry, exit points
    {
      std::cout << "--- graycode --- " << std::endl;

      for(int i=0; i < pow2(DIM); i++) {
        std::cout << tfm::format("%5d => "
                                 "graycode = %6s, entry = %6s, exit = %6s\n",
                                 i,
                                 std::bitset<DIM>(graycode(i)),
                                 std::bitset<DIM>(entry_point(i)),
                                 std::bitset<DIM>(exit_point(i)));
      }
    }

    // index <=> coordinate mapping consistency
    {
      const int max_order = 6;

      std::random_device rd;
      std::mt19937 mt(rd());
      std::uniform_int_distribution<int> urand(0, DIM-1);

      for(int order=1; order < max_order ;order++) {
        // random entry point and direction
        int entry     = urand(mt);
        int direction = urand(mt);
        int index, coord[DIM];

        std::cout << tfm::format("*** checking %1dD curve: "
                                 "order = %2d, "
                                 "entry = %2d, "
                                 "direction = %2d\n",
                                 DIM, order, entry, direction);
        for(int i=0; i < pow2(2*order); i++) {
          // get coordiante
          get_coordinate(i, entry, direction, order, coord);

          // get index
          index = get_index(coord, entry, direction, order);

          // check
          assert(index == i);
        }
      }
    }
  }
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
