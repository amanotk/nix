// -*- C++ -*-
#ifndef _CHUNKMAP_HPP_
#define _CHUNKMAP_HPP_

#include "jsonio.hpp"
#include "nix.hpp"
#include "sfc.hpp"
#include "xtensorall.hpp"

NIX_NAMESPACE_BEGIN

///
/// @brief ChunkMap class
/// @tparam Ndim number of dimensions
///
/// The chunk ID is defined with row-major ordering of chunks in cartesian
/// coordinate. Mapping between chunk ID and cartesian coordinate may be
/// calculated via get_chunk() and get_coordinate() methods.
///
template <int Ndim>
class ChunkMap
{
protected:
  using IntArray1D = xt::xtensor<int, 1>;
  using IntArray2D = xt::xtensor<int, 2>;
  using IntArrayND = xt::xtensor<int, Ndim>;

  int        size;    ///< number of total chunks
  int        dims[3]; ///< chunk dimension
  IntArray1D rank;    ///< chunk ID to MPI rank map
  IntArray2D coord;   ///< chunk ID to coordinate map
  IntArrayND chunkid; ///< coordinate to chunk ID map

public:
  ///
  /// @brief constructor for 1D map
  /// @param Cx number of chunk in x direction
  ///
  ChunkMap(int Cx);

  ///
  /// @brief constructor for 2D map
  /// @param Cy number of chunk in y direction
  /// @param Cx number of chunk in x direction
  ///
  ChunkMap(int Cy, int Cx);

  ///
  /// @brief constructor for 3D map
  /// @param Cz number of chunk in z direction
  /// @param Cy number of chunk in y direction
  /// @param Cx number of chunk in x direction
  ///
  ChunkMap(int Cz, int Cy, int Cx);

  ///
  /// @brief constructor
  /// @param dims number of chunk in each direction
  ///
  ChunkMap(const int dims[Ndim]);

  ///
  /// @brief check the validity of map
  /// @return true if it is valid map, false otherwise
  ///
  virtual bool validate();

  ///
  /// @brief save map information as json object
  /// @param obj json object to which map information will be stored
  ///
  virtual void save_json(json& obj);

  ///
  /// @brief load map information from json object
  /// @param obj json object from which map information will be loaded
  ///
  virtual void load_json(json& obj);

  ///
  /// @brief set process rank for given chunk ID
  /// @param id chunk ID
  /// @param r rank
  ///
  void set_rank(int id, int r)
  {
    rank(id) = r;
  }

  ///
  /// @brief get process rank associated with chunk ID
  /// @param id chunk ID
  /// @return rank
  ///
  int get_rank(int id)
  {
    if (id >= 0 && id < size) {
      return rank(id);
    } else {
      return MPI_PROC_NULL;
    }
  }

  ///
  /// @brief get coordinate of chunk for 1D map
  /// @param id chunk ID
  /// @param cx x coordinate of chunk will be stored
  ///
  void get_coordinate(int id, int& cx);

  ///
  /// @brief get coordinate of chunk for 2D map
  /// @param id chunk ID
  /// @param cy y coordinate of chunk will be stored
  /// @param cx x coordinate of chunk will be stored
  ///
  void get_coordinate(int id, int& cy, int& cx);

  ///
  /// @brief get coordinate of chunk for 3D map
  /// @param id chunk ID
  /// @param cz z coordinate of chunk will be stored
  /// @param cy y coordinate of chunk will be stored
  /// @param cx x coordinate of chunk will be stored
  ///
  void get_coordinate(int id, int& cz, int& cy, int& cx);

  ///
  /// @brief get chunk ID for 1D map
  /// @param cx x coordinate of chunk
  /// @return chunk ID
  ///
  int get_chunkid(int cx);

  ///
  /// @brief get chunk ID for 2D map
  /// @param cy y coordinate of chunk
  /// @param cx x coordinate of chunk
  /// @return chunk ID
  ///
  int get_chunkid(int cy, int cx);

  ///
  /// @brief get chunk ID for 3D map
  /// @param cz z coordinate of chunk
  /// @param cy y coordinate of chunk
  /// @param cx x coordinate of chunk
  /// @return chunk ID
  ///
  int get_chunkid(int cz, int cy, int cx);
};

//
// implementation follows
//

#define DEFINE_MEMBER1(type, name)                                                                 \
  template <>                                                                                      \
  inline type ChunkMap<1>::name
#define DEFINE_MEMBER2(type, name)                                                                 \
  template <>                                                                                      \
  inline type ChunkMap<2>::name
#define DEFINE_MEMBER3(type, name)                                                                 \
  template <>                                                                                      \
  inline type ChunkMap<3>::name

DEFINE_MEMBER1(, ChunkMap)(int Cx)
{
  size    = Cx;
  dims[0] = Cx;
  dims[1] = 1;
  dims[2] = 1;

  // memory allocation
  {
    std::vector<size_t> dims1 = {static_cast<size_t>(size)};
    std::vector<size_t> dims2 = {static_cast<size_t>(size), 1};
    std::vector<size_t> dims3 = {static_cast<size_t>(Cx)};

    rank.resize(dims1);
    coord.resize(dims2);
    chunkid.resize(dims3);

    rank.fill(0);
    coord.fill(0);
    chunkid.fill(0);
  }

  // build mapping
  sfc::get_map1d(Cx, chunkid, coord);
}

DEFINE_MEMBER2(, ChunkMap)(int Cy, int Cx)
{
  size    = Cy * Cx;
  dims[0] = Cy;
  dims[1] = Cx;
  dims[2] = 1;

  // memory allocation
  {
    std::vector<size_t> dims1 = {static_cast<size_t>(size)};
    std::vector<size_t> dims2 = {static_cast<size_t>(size), 2};
    std::vector<size_t> dims3 = {static_cast<size_t>(Cy), static_cast<size_t>(Cx)};

    rank.resize(dims1);
    coord.resize(dims2);
    chunkid.resize(dims3);

    rank.fill(0);
    coord.fill(0);
    chunkid.fill(0);
  }

  // build mapping
  sfc::get_map2d(Cy, Cx, chunkid, coord);
}

DEFINE_MEMBER3(, ChunkMap)(int Cz, int Cy, int Cx)
{
  size    = Cz * Cy * Cx;
  dims[0] = Cz;
  dims[1] = Cy;
  dims[2] = Cx;

  // memory allocation
  {
    std::vector<size_t> dims1 = {static_cast<size_t>(size)};
    std::vector<size_t> dims2 = {static_cast<size_t>(size), 3};
    std::vector<size_t> dims3 = {static_cast<size_t>(Cz), static_cast<size_t>(Cy),
                                 static_cast<size_t>(Cx)};

    rank.resize(dims1);
    coord.resize(dims2);
    chunkid.resize(dims3);

    rank.fill(0);
    coord.fill(0);
    chunkid.fill(0);
  }

  // build mapping
  sfc::get_map3d(Cz, Cy, Cx, chunkid, coord);
}

DEFINE_MEMBER1(, ChunkMap)(const int dims[1]) : ChunkMap<1>(dims[0])
{
}

DEFINE_MEMBER2(, ChunkMap)(const int dims[2]) : ChunkMap<2>(dims[0], dims[1])
{
}

DEFINE_MEMBER3(, ChunkMap)(const int dims[3]) : ChunkMap<3>(dims[0], dims[1], dims[2])
{
}

DEFINE_MEMBER1(bool, validate)()
{
  return sfc::check_index(chunkid);
}

DEFINE_MEMBER2(bool, validate)()
{
  return sfc::check_index(chunkid) & sfc::check_locality2d(coord);
}

DEFINE_MEMBER3(bool, validate)()
{
  return sfc::check_index(chunkid) & sfc::check_locality3d(coord);
}

DEFINE_MEMBER1(void, save_json)(json& obj)
{
  // meta data
  obj["size"]  = size;
  obj["ndim"]  = 1;
  obj["shape"] = {dims[0]};

  // map
  obj["chunkid"] = chunkid;
  obj["coord"]   = coord;
  obj["rank"]    = rank;
}

DEFINE_MEMBER2(void, save_json)(json& obj)
{
  // meta data
  obj["size"]  = size;
  obj["ndim"]  = 2;
  obj["shape"] = {dims[0], dims[1]};

  // map
  obj["chunkid"] = chunkid;
  obj["coord"]   = coord;
  obj["rank"]    = rank;
}

DEFINE_MEMBER3(void, save_json)(json& obj)
{
  // meta data
  obj["size"]  = size;
  obj["ndim"]  = 3;
  obj["shape"] = {dims[0], dims[1], dims[2]};

  // map
  obj["chunkid"] = chunkid;
  obj["coord"]   = coord;
  obj["rank"]    = rank;
}

DEFINE_MEMBER1(void, load_json)(json& obj)
{
  if (obj["ndim"].get<int>() != 1) {
    ERROR << tfm::format("Invalid input to ChunkMap<1>::load_json");
  }

  // meta data
  size    = obj["size"].get<int>();
  dims[0] = obj["shape"][0].get<int>();

  // map
  chunkid = obj["chunkid"];
  coord   = obj["coord"];
  rank    = obj["rank"];
}

DEFINE_MEMBER2(void, load_json)(json& obj)
{
  if (obj["ndim"].get<int>() != 2) {
    ERROR << tfm::format("Invalid input to ChunkMap<2>::load_json");
  }

  // meta data
  size    = obj["size"].get<int>();
  dims[0] = obj["shape"][0].get<int>();
  dims[1] = obj["shape"][1].get<int>();

  // map
  chunkid = obj["chunkid"];
  coord   = obj["coord"];
  rank    = obj["rank"];
}

DEFINE_MEMBER3(void, load_json)(json& obj)
{
  if (obj["ndim"].get<int>() != 3) {
    ERROR << tfm::format("Invalid input to ChunkMap<3>::load_json");
  }

  // meta data
  size    = obj["size"].get<int>();
  dims[0] = obj["shape"][0].get<int>();
  dims[1] = obj["shape"][1].get<int>();
  dims[2] = obj["shape"][2].get<int>();

  // map
  chunkid = obj["chunkid"];
  coord   = obj["coord"];
  rank    = obj["rank"];
}

DEFINE_MEMBER1(void, get_coordinate)(int id, int& cx)
{
  if (id >= 0 && id < size) {
    cx = coord(id, 0);
  } else {
    cx = -1;
  }
}

DEFINE_MEMBER1(int, get_chunkid)(int cx)
{
  if (cx >= 0 && cx < dims[0]) {
    return chunkid(cx);
  } else {
    return -1;
  }
}

DEFINE_MEMBER2(void, get_coordinate)(int id, int& cy, int& cx)
{
  if (id >= 0 && id < size) {
    cx = coord(id, 0);
    cy = coord(id, 1);
  } else {
    cy = -1;
    cx = -1;
  }
}

DEFINE_MEMBER2(int, get_chunkid)(int cy, int cx)
{
  if ((cy >= 0 && cy < dims[0]) && (cx >= 0 && cx < dims[1])) {
    return chunkid(cy, cx);
  } else {
    return -1;
  }
}

DEFINE_MEMBER3(void, get_coordinate)(int id, int& cz, int& cy, int& cx)
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

DEFINE_MEMBER3(int, get_chunkid)(int cz, int cy, int cx)
{
  if ((cz >= 0 && cz < dims[0]) && (cy >= 0 && cy < dims[1]) && (cx >= 0 && cx < dims[2])) {
    return chunkid(cz, cy, cx);
  } else {
    return -1;
  }
}

#undef DEFINE_MEMBER
#undef DEFINE_MEMBER1
#undef DEFINE_MEMBER2
#undef DEFINE_MEMBER3

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
