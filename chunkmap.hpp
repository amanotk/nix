// -*- C++ -*-
#ifndef _CHUNKMAP_HPP_
#define _CHUNKMAP_HPP_

///
/// ChunkMap
///
/// $Id$
///
#include "common.hpp"
#include "utils/json.hpp"
#include "utils/mpistream.hpp"
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
  using json = nlohmann::ordered_json;
  typedef xt::xtensor<int,1>  IntArray1D;
  typedef xt::xtensor<int,2>  IntArray2D;
  typedef xt::xtensor<int,3>  IntArray3D;

  int          size;     ///< number of total chunkes
  int          dims[3];  ///< chunk dimension
  IntArray1D   rank;     ///< chunk id to MPI rank map
  IntArray2D   coord;    ///< chunk id to coordinate map
  IntArray3D   chunkid;  ///< coordiante to chunk id map


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

    // set coordiante to chunk ID mapping
    for(int i=0; i < size ;i++) {
      int jx, jy, jz;
      get_coordinate(i, jz, jy, jx);
      chunkid(jz, jy, jx) = i;
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
  BaseChunkMap(const int cdims[3])
  {
    size     = cdims[0]*cdims[1]*cdims[2];
    dims[0]  = cdims[0];
    dims[1]  = cdims[1];
    dims[2]  = cdims[2];

    // memory allocation
    xt::resize(rank, {size});
    xt::resize(coord, {size, 3});
    xt::resize(chunkid, {dims[0], dims[1], dims[2]});

    // build mapping with pseudo hilbert curve
    {
      int dimensions[3] = {dims[0], dims[1], dims[2]};
      int directions[3] = {0, 1, 2};

      sort_dimension(dimensions, directions);
      build_mapping(dimensions, directions);
    }
  }


  // debug output
  void debug(std::ostream &ofs)
  {
    for(int i=0; i < size ;i++) {
      ofs << tfm::format("%3d, %3d, %3d\n",
                         coord(i,0), coord(i,1), coord(i,2));
    }
  }


  // set rank for chunk id
  void set_rank(const int id, const int r)
  {
    rank(id) = r;
  }


  // return process rank associated with chunk id
  int get_rank(const int id)
  {
    if( id >= 0 && id < size ) {
      return rank(id);
    } else {
      return MPI_PROC_NULL;
    }
  }


  // return chunk coordinate associated with chunk id
  void get_coordinate(const int id, int &cz, int &cy, int &cx)
  {
    if( id >= 0 && id < size ) {
      cz = coord(id, 0);
      cy = coord(id, 1);
      cx = coord(id, 2);
    } else {
      cz = -1;
      cy = -1;
      cx = -1;
    }
  }


  // return chunk id associated with coordinate
  int get_chunkid(const int cz, const int cy, const int cx)
  {
    if( (cz >= 0 && cz < dims[2]) &&
        (cy >= 0 && cy < dims[1]) &&
        (cx >= 0 && cx < dims[0]) ) {
      return chunkid(cz, cy, cx);
    } else {
      return -1;
    }
  }


  // dump to json
  virtual void json_dump(std::ostream &out)
  {
    json obj;

    // meta data
    {
      obj["size"] = size;
      obj["dims"] = {dims[0], dims[1], dims[2]};
    }

    // id
    {
      json chunkid_obj = json::array();

      for(int iz=0; iz < dims[0] ;iz++) {
        json cy = json::array();
        for(int iy=0; iy < dims[1] ;iy++) {
          json cx = json::array();
          for(int ix=0; ix < dims[2] ;ix++) {
            cx.push_back(chunkid(iz, iy, ix));
          }
          cy.push_back(cx);
        }
        chunkid_obj.push_back(cy);
      }
      obj["chunkid"] = chunkid_obj;
    }

    // coordinate
    {
      json coord_obj = json::array();

      for(int id=0; id < size ;id++) {
        int ix, iy, iz;
        get_coordinate(id, iz, iy, ix);
        coord_obj.push_back({iz, iy, ix});
      }
      obj["coord"] = coord_obj;
    }

    // rank
    {
      json rank_obj = json::array();

      for(int id=0; id < size ;id++) {
        int rank = get_rank(id);
        rank_obj.push_back(rank);
      }
      obj["rank"] = rank_obj;
    }

    // output
    {
      json root;
      root= { {"chunkmap", obj} };
      out << std::setw(2) << root << std::endl;
    }
  }
};


// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
