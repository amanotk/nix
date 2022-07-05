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
#include "utils/json.hpp"
#include "utils/mpistream.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"

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
  typedef xt::xtensor<int, 1> IntArray1D;
  typedef xt::xtensor<int, 2> IntArray2D;
  typedef xt::xtensor<int, 3> IntArray3D;

  int size;           ///< number of total chunkes
  int dims[3];        ///< chunk dimension
  IntArray1D rank;    ///< chunk id to MPI rank map
  IntArray2D coord;   ///< chunk id to coordinate map
  IntArray3D chunkid; ///< coordiante to chunk id map

  int ilog2(int x);

  void build_mapping_1d(int dims[3], int dirs[3]);

  void build_mapping_2d(int dims[3], int dirs[3]);

  void build_mapping_3d(int dims[3], int dirs[3]);

  void check_dimension_2d(const int dims[3], const int dirs[3]);

  void check_dimension_3d(const int dims[3], const int dirs[3]);

  void sort_dimension(int dims[3], int dirs[3]);

public:
  // constructor
  BaseChunkMap(const int cdims[3]);

  virtual void build_mapping(int dims[3], int dirs[3]);

  virtual void json_dump(std::ostream &out);

  // set rank for chunk id
  void set_rank(const int id, const int r)
  {
    rank(id) = r;
  }

  // return process rank associated with chunk id
  int get_rank(const int id)
  {
    if (id >= 0 && id < size) {
      return rank(id);
    } else {
      return MPI_PROC_NULL;
    }
  }

  // return chunk coordinate associated with chunk id
  void get_coordinate(const int id, int &cz, int &cy, int &cx)
  {
    if (id >= 0 && id < size) {
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
    if ((cz >= 0 && cz < dims[2]) && (cy >= 0 && cy < dims[1]) &&
        (cx >= 0 && cx < dims[0])) {
      return chunkid(cz, cy, cx);
    } else {
      return -1;
    }
  }
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
