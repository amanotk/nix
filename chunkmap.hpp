// -*- C++ -*-
#ifndef _CHUNKMAP_HPP_
#define _CHUNKMAP_HPP_

///
/// ChunkMap
///
/// $Id$
///
#include "common.hpp"
#include "hilbert.hpp"
#include "mdarray.hpp"


///
/// BaseChunkMap
///
/// * rank   : MPI rank for each chunk
///            Defines mappping from chunk ID to MPI rank.
///
/// * coord  : cartesian coordiante for each index
///            Defines mapping from hilbert index to cartesian coordinate
///
/// The chunk ID is defined with row-major ordering of chunkes in cartesian
/// coordinate. Mapping between chunk ID and cartesian coordinate may be
/// calculated via get_chunk() and get_coordinate() methods.
///
class BaseChunkMap
{
protected:
  typedef xt::xtensor<int,1>  IntArray1D;
  typedef xt::xtensor<int,2>  IntArray2D;

  int          Np;        ///< number of total chunkes
  int          pdims[3];  ///< chunk dimension
  int          stride[3]; ///< chunk stride
  IntArray1D   rank;      ///< chunk id to MPI rank map
  IntArray2D   coord;     ///< index to cartesian coordinate map


  int ilog2(int x)
  {
    int l = 0;
    int m = x;

    while( m >>= 1 ) {
      l++;
    }

    // check
    if( (1 << l) == x ) {
      return l;
    }

    // error
    return -1;
  }

  void build_mapping_1d(int dims[3], int dirs[3])
  {
    // simple 1D ordering
    int iz = 0;
    int iy = 0;
    for(int ix=0; ix < dims[2]; ix++) {
      int idx = ix;

      coord(idx, dirs[2]) = iz;
      coord(idx, dirs[1]) = iy;
      coord(idx, dirs[0]) = ix;
    }
  }

  void build_mapping_2d(int dims[3], int dirs[3])
  {
    typedef HilbertCurve<2> HC2;

    // repeat 2D square hilbert curve in one direction if needed
    int nsquare   = dims[1];
    int nrepeat   = dims[2] / nsquare;
    int nsquare2  = nsquare*nsquare;
    int order     = ilog2(nsquare);
    int entry     = 0;
    int direction = 0;

    for(int ir=0; ir < nrepeat; ir++) {
      int iz = 0;
      for(int iy=0; iy < nsquare ;iy++) {
        for(int ix=0; ix < nsquare ;ix++) {
          int c[2] = {iy, ix};
          int jx   = ix + ir*nsquare;
          int idx0 = nsquare2*ir;
          int idx  = HC2::get_index(c, entry, direction, order) + idx0;

          coord(idx, dirs[0]) = iz;
          coord(idx, dirs[1]) = iy;
          coord(idx, dirs[2]) = jx;
        }
      }
    }
  }

  void build_mapping_3d(int dims[3], int dirs[3])
  {
    typedef HilbertCurve<3> HC3;

    // repeat 3D cube hilbert curve in two directions if needed
    int ncube     = dims[0];
    int nrepeat1  = dims[1] / ncube;
    int nrepeat2  = dims[2] / ncube;
    int ncube3    = ncube*ncube*ncube;
    int order     = ilog2(ncube);
    int entry     = 0;
    int direction = 0;

    for(int ir2=0; ir2 < nrepeat2; ir2++) {
      for(int ir1=0; ir1 < nrepeat1; ir1++) {
        for(int iz=0; iz < ncube; iz++) {
          for(int iy=0; iy < ncube; iy++) {
            for(int ix=0; ix < ncube; ix++) {
              int c[3] = {iz, iy, ix};
              int jy   = iy + ir1*ncube;
              int jx   = ix + ir2*ncube;
              int idx0 = ncube3*ir1 + ncube3*nrepeat1*ir2;
              int idx  = HC3::get_index(c, entry, direction, order) + idx0;

              coord(idx, dirs[0]) = iz;
              coord(idx, dirs[1]) = jy;
              coord(idx, dirs[2]) = jx;
            }
          }
        }
      }
    }
  }

  void build_mapping(int dims[3], int dirs[3])
  {
    // assume dims are already sorted in ascending order
    if( dims[0] == 1 && dims[1] == 1 ) {
      // 1D
      build_mapping_1d(dims, dirs);
    } else if( dims[0] == 1 ) {
      // 2D
      check_dimension_2d(dims, dirs);
      build_mapping_2d(dims, dirs);
    } else {
      // 3D
      check_dimension_3d(dims, dirs);
      build_mapping_3d(dims, dirs);
    }
  }

  void check_dimension_2d(const int dims[3], const int dirs[3])
  {
    bool status = true;

    // first dimension
    status = status & (dims[0] == 1);

    // second dimension must be power of 2
    status = status & (ilog2(dims[1]) > 0);

    // third dimension must be multiple of second
    status = status & (dims[2] % dims[1] == 0);

    if( !status ) {
      std::cerr << tfm::format("Error: incompatible dimensions: "
                               "%8d, %8d, %8d\n",
                               dims[0], dims[1], dims[2]);
      exit(-1);
    }
  }

  void check_dimension_3d(const int dims[3], const int dirs[3])
  {
    bool status = true;

    // first dimension must be power of 2
    status = status & (ilog2(dims[0]) > 0);

    // second dimension must be multiple of first
    status = status & (dims[1] % dims[0] == 0);

    // third dimension must be multiple of first
    status = status & (dims[2] % dims[0] == 0);

    if( !status ) {
      std::cerr << tfm::format("Error: incompatible dimensions: "
                               "%8d, %8d, %8d\n",
                               dims[0], dims[1], dims[2]);
      exit(-1);
    }
  }

  void sort_dimension(int dims[3], int dirs[3])
  {
    typedef std::pair<int,int> T_pair;

    std::vector<T_pair> p;

    // initialize pair
    for(int i=0; i < 3 ;i++) {
      p.push_back(std::make_pair(2-i, dims[i]));
    }

    // sort by dims
    std::stable_sort(p.begin(), p.end(),
                     [](const T_pair &x, const T_pair &y)
                     {return x.second < y.second;});

    // return result
    for(int i=0; i < 3; i++) {
      dirs[i] = p[i].first;
      dims[i] = p[i].second;
      std::cout << tfm::format("dirs = %3d, dims = %3d\n", dirs[i], dims[i]);
    }
  }


public:
  // constructor
  BaseChunkMap(const int dims[3])
  {
    Np        = dims[0]*dims[1]*dims[2];
    pdims[0]  = dims[0];
    pdims[1]  = dims[1];
    pdims[2]  = dims[2];
    stride[0] = dims[1]*dims[2];
    stride[1] = dims[2];
    stride[2] = 1;

    // memory allocation
    xt::resize(rank, {Np});
    xt::resize(coord, {Np, 3});

    // build mapping with pseudo hilbert curve
    {
      int dims[3] = {pdims[0], pdims[1], pdims[2]};
      int dirs[3] = {0, 1, 2};

      sort_dimension(dims, dirs);
      build_mapping(dims, dirs);
    }
  }


  // debug output
  void debug(std::ostream &ofs)
  {
    for(int ip=0; ip < Np ;ip++) {
      ofs << tfm::format("%3d, %3d, %3d\n",
                         coord(ip,0), coord(ip,1), coord(ip,2));
    }
  }


  // set rank for chunk id
  void set_rank(const int pid, const int r)
  {
    rank(pid) = r;
  }


  // return process rank associated with chunk id
  int get_rank(const int pid)
  {
    return rank(pid);
  }


  // return patach coordinate
  void get_coordiante(const int pid, int &pz, int &py, int &px)
  {
    pz = coord(pid,0);
    py = coord(pid,1);
    px = coord(pid,2);
  }
};


// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
