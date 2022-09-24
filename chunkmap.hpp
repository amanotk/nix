// -*- C++ -*-
#ifndef _CHUNKMAP_HPP_
#define _CHUNKMAP_HPP_

#include "jsonio.hpp"
#include "nix.hpp"
#include "sfc.hpp"
#include "xtensorall.hpp"

///
/// @brief ChunkMap class
/// @tparam Ndim number of dimensions
///
/// The chunk ID is defined with row-major ordering of chunkes in cartesian
/// coordinate. Mapping between chunk ID and cartesian coordinate may be
/// calculated via get_chunk() and get_coordinate() methods.
///
template <int Ndim>
class ChunkMap
{
protected:
  using json       = nix::json;
  using IntArray1D = xt::xtensor<int, 1>;
  using IntArray2D = xt::xtensor<int, 2>;
  using IntArrayND = xt::xtensor<int, Ndim>;

  int        size;    ///< number of total chunks
  int        dims[3]; ///< chunk dimension
  IntArray1D rank;    ///< chunk ID to MPI rank map
  IntArray2D coord;   ///< chunk ID to coordinate map
  IntArrayND chunkid; ///< coordiante to chunk ID map

public:
  ///
  /// @brief constructor for 1D map
  /// @param Cx number of chunk in x direction
  ///
  ChunkMap(const int Cx);

  ///
  /// @brief constructor for 2D map
  /// @param Cy number of chunk in y direction
  /// @param Cx number of chunk in x direction
  ///
  ChunkMap(const int Cy, const int Cx);

  ///
  /// @brief constructor for 3D map
  /// @param Cz number of chunk in z direction
  /// @param Cy number of chunk in y direction
  /// @param Cx number of chunk in x direction
  ///
  ChunkMap(const int Cz, const int Cy, const int Cx);

  ///
  /// @brief constructor
  /// @param dims number of chunk in each direction
  ///
  ChunkMap(const int dims[Ndim]);

  ///
  /// @brief check the validity of map
  /// @return true if it is valid map, false oterwise
  ///
  virtual bool validate();

  ///
  /// @brief save map information as json object
  /// @param obj json object to which map information will be stored
  ///
  virtual void save_json(json &obj);

  ///
  /// @brief load map information from json object
  /// @param obj json object from which map information will be loaded
  ///
  virtual void load_json(json &obj);

  ///
  /// @brief set process rank for given chunk ID
  /// @param id chunk ID
  /// @param r rank
  ///
  void set_rank(const int id, const int r);

  ///
  /// @brief get process rank associated with chunk ID
  /// @param id chunk ID
  /// @return rank
  ///
  int get_rank(const int id);

  ///
  /// @brief get coordinate of chunk for 1D map
  /// @param id chunk ID
  /// @param cx x coordinate of chunk will be stored
  ///
  void get_coordinate(const int id, int &cx);

  ///
  /// @brief get coordinate of chunk for 2D map
  /// @param id chunk ID
  /// @param cy y coordinate of chunk will be stored
  /// @param cx x coordinate of chunk will be stored
  ///
  void get_coordinate(const int id, int &cy, int &cx);

  ///
  /// @brief get coordinate of chunk for 3D map
  /// @param id chunk ID
  /// @param cz z coordinate of chunk will be stored
  /// @param cy y coordinate of chunk will be stored
  /// @param cx x coordinate of chunk will be stored
  ///
  void get_coordinate(const int id, int &cz, int &cy, int &cx);

  ///
  /// @brief get chunk ID for 1D map
  /// @param cx x coordinate of chunk
  /// @return chunk ID
  ///
  int get_chunkid(const int cx);

  ///
  /// @brief get chunk ID for 2D map
  /// @param cy y coordinate of chunk
  /// @param cx x coordinate of chunk
  /// @return chunk ID
  ///
  int get_chunkid(const int cy, const int cx);

  ///
  /// @brief get chunk ID for 3D map
  /// @param cz z coordinate of chunk
  /// @param cy y coordinate of chunk
  /// @param cx x coordinate of chunk
  /// @return chunk ID
  ///
  int get_chunkid(const int cz, const int cy, const int cx);
};

template <int Ndim>
inline void ChunkMap<Ndim>::set_rank(const int id, const int r)
{
  rank(id) = r;
}

template <int Ndim>
inline int ChunkMap<Ndim>::get_rank(const int id)
{
  if (id >= 0 && id < size) {
    return rank(id);
  } else {
    return MPI_PROC_NULL;
  }
}

template <>
inline void ChunkMap<1>::get_coordinate(const int id, int &cx)
{
  if (id >= 0 && id < size) {
    cx = coord(id, 0);
  } else {
    cx = -1;
  }
}

template <>
inline int ChunkMap<1>::get_chunkid(int cx)
{
  if (cx >= 0 && cx < dims[0]) {
    return chunkid(cx);
  } else {
    return -1;
  }
}

template <>
inline void ChunkMap<2>::get_coordinate(const int id, int &cy, int &cx)
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
inline int ChunkMap<2>::get_chunkid(int cy, int cx)
{
  if ((cy >= 0 && cy < dims[0]) && (cx >= 0 && cx < dims[1])) {
    return chunkid(cy, cx);
  } else {
    return -1;
  }
}

template <>
inline int ChunkMap<3>::get_chunkid(int cz, int cy, int cx)
{
  if ((cz >= 0 && cz < dims[0]) && (cy >= 0 && cy < dims[1]) && (cx >= 0 && cx < dims[2])) {
    return chunkid(cz, cy, cx);
  } else {
    return -1;
  }
}

template <>
inline void ChunkMap<3>::get_coordinate(const int id, int &cz, int &cy, int &cx)
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
