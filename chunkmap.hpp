// -*- C++ -*-
#ifndef _CHUNKMAP_HPP_
#define _CHUNKMAP_HPP_

///
/// ChunkMap
///
/// $Id$
///
#include "common.hpp"
#include "json.hpp"
#include "jsonio.hpp"
#include "sfc.hpp"
#include "xtensorall.hpp"

#include <mpi.h>

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
template <int N>
class BaseChunkMap
{
protected:
  using json = nlohmann::ordered_json;
  typedef xt::xtensor<int, 1> IntArray1D;
  typedef xt::xtensor<int, 2> IntArray2D;
  typedef xt::xtensor<int, N> IntArrayND;

  int        size;    ///< number of total chunkes
  int        dims[3]; ///< chunk dimension
  IntArray1D rank;    ///< chunk id to MPI rank map
  IntArray2D coord;   ///< chunk id to coordinate map
  IntArrayND chunkid; ///< coordiante to chunk id map

public:
  BaseChunkMap(const int Cx);

  BaseChunkMap(const int Cy, const int Cx);

  BaseChunkMap(const int Cz, const int Cy, const int Cx);

  BaseChunkMap(const int dims[N]);

  virtual bool validate();

  virtual void save(json &obj, MPI_File *fh, size_t *disp);

  virtual void json_save(std::ostream &out);

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

  void get_coordinate(const int id, int &cx);

  void get_coordinate(const int id, int &cy, int &cx);

  void get_coordinate(const int id, int &cz, int &cy, int &cx);

  int get_chunkid(const int cx);

  int get_chunkid(const int cy, const int cx);

  int get_chunkid(const int cz, const int cy, const int cx);
};

template <>
inline void BaseChunkMap<1>::get_coordinate(const int id, int &cx)
{
  if (id >= 0 && id < size) {
    cx = coord(id, 0);
  } else {
    cx = -1;
  }
}

template <>
inline int BaseChunkMap<1>::get_chunkid(int cx)
{
  if (cx >= 0 && cx < dims[0]) {
    return chunkid(cx);
  } else {
    return -1;
  }
}

template <>
inline void BaseChunkMap<2>::get_coordinate(const int id, int &cy, int &cx)
{
  if (id >= 0 && id < size) {
    cx = coord(id, 0);
    cy = coord(id, 1);
  } else {
    cy = -1;
    cx = -1;
  }
}

template <>
inline int BaseChunkMap<2>::get_chunkid(int cy, int cx)
{
  if ((cy >= 0 && cy < dims[0]) && (cx >= 0 && cx < dims[1])) {
    return chunkid(cy, cx);
  } else {
    return -1;
  }
}

template <>
inline int BaseChunkMap<3>::get_chunkid(int cz, int cy, int cx)
{
  if ((cz >= 0 && cz < dims[0]) && (cy >= 0 && cy < dims[1]) && (cx >= 0 && cx < dims[2])) {
    return chunkid(cz, cy, cx);
  } else {
    return -1;
  }
}

template <>
inline void BaseChunkMap<3>::get_coordinate(const int id, int &cz, int &cy, int &cx)
{
  if (id >= 0 && id < size) {
    cx = coord(id, 0);
    cy = coord(id, 1);
    cz = coord(id, 2);
  } else {
    cz = -1;
    cy = -1;
    cx = -1;
  }
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
