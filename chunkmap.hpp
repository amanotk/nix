// -*- C++ -*-
#ifndef _CHUNKMAP_HPP_
#define _CHUNKMAP_HPP_

#include "nix.hpp"
#include "sfc.hpp"
#include "xtensorall.hpp"

NIX_NAMESPACE_BEGIN

///
/// @brief ChunkMap class
///
/// The chunk ID is defined with row-major ordering of chunks in cartesian
/// coordinate. Mapping between chunk ID and cartesian coordinate may be
/// calculated via get_chunk() and get_coordinate() methods.
///
template <int Dimension>
class ChunkMap
{
protected:
  using IntArray1D = xt::xtensor<int, 1>;
  using IntArray2D = xt::xtensor<int, 2>;
  using IntArrayND = xt::xtensor<int, Dimension>;

  int              size;           ///< number of total chunks
  int              dims[3];        ///< chunk dimension
  int              periodicity[3]; ///< periodicity in each direction
  std::vector<int> boundary;       ///< rank boundary
  IntArray2D       coord;          ///< chunk ID to coordinate map
  IntArrayND       chunkid;        ///< coordinate to chunk ID map

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
  ChunkMap(const int dims[Dimension]);

  ///
  /// @brief check the validity of map
  /// @return true if it is valid map, false otherwise
  ///
  virtual bool validate();

  ///
  /// @brief get map information as json object
  /// @return obj json object
  ///
  virtual json to_json();

  ///
  /// @brief restore map information from json object
  /// @param obj json object
  ///
  virtual void from_json(json& obj);

  ///
  /// @brief set periodicity in each direction
  /// @param pz periodicity in z direction
  /// @param py periodicity in y direction
  /// @param px periodicity in x direction
  ///
  virtual void set_periodicity(int pz, int py, int px)
  {
    periodicity[0] = pz;
    periodicity[1] = py;
    periodicity[2] = px;
  }

  ///
  /// @brief return neighbor coordinate for a specific direction `dir`
  /// @param coord index of coordinate
  /// @param delta difference of index of coordinate from `coord`
  /// @param dir direction of coordinate
  /// @return `coord + delta` if not at boundary, otherwise boundary condition dependent
  ///
  virtual int get_neighbor_coord(int coord, int delta, int dir)
  {
    int cdir = coord + delta;

    if (periodicity[dir] == 1) {
      cdir = cdir >= 0 ? cdir : dims[dir] - 1;
      cdir = cdir < dims[dir] ? cdir : 0;
    } else {
      cdir = (cdir >= 0 && cdir < dims[dir]) ? cdir : -1;
    }

    return cdir;
  }

  ///
  /// @brief get process rank associated with chunk ID
  /// @param id chunk ID
  /// @return rank
  ///
  int get_rank(int id)
  {
    if (id >= 0 && id < size) {
      auto it = std::upper_bound(boundary.begin(), boundary.end(), id);
      return std::distance(boundary.begin(), it) - 1;
    } else {
      return MPI_PROC_NULL;
    }
  }

  ///
  /// @brief set process rank boundary
  /// @param boundary array of rank boundary to set
  ///
  virtual void set_rank_boundary(std::vector<int>& boundary)
  {
    this->boundary = boundary;
  }

  ///
  /// @brief get process rank boundary
  /// @return array of rank boundary
  ///
  std::vector<int> get_rank_boundary()
  {
    return boundary;
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

DEFINE_MEMBER1(, ChunkMap)(int Cx) : periodicity{1, 1, 1}
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

    coord.resize(dims2);
    chunkid.resize(dims3);

    coord.fill(0);
    chunkid.fill(0);
  }

  // build mapping
  sfc::get_map1d(Cx, chunkid, coord);
}

DEFINE_MEMBER2(, ChunkMap)(int Cy, int Cx) : periodicity{1, 1, 1}
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

    coord.resize(dims2);
    chunkid.resize(dims3);

    coord.fill(0);
    chunkid.fill(0);
  }

  // build mapping
  sfc::get_map2d(Cy, Cx, chunkid, coord);
}

DEFINE_MEMBER3(, ChunkMap)(int Cz, int Cy, int Cx) : periodicity{1, 1, 1}
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

    coord.resize(dims2);
    chunkid.resize(dims3);

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

DEFINE_MEMBER1(json, to_json)()
{
  json obj;

  // meta data
  obj["size"]        = size;
  obj["ndim"]        = 1;
  obj["shape"]       = {dims[0]};
  obj["periodicity"] = {periodicity[0]};

  // map
  obj["chunkid"]  = chunkid;
  obj["coord"]    = coord;
  obj["boundary"] = boundary;

  return obj;
}

DEFINE_MEMBER2(json, to_json)()
{
  json obj;

  // meta data
  obj["size"]        = size;
  obj["ndim"]        = 2;
  obj["shape"]       = {dims[0], dims[1]};
  obj["periodicity"] = {periodicity[0], periodicity[1]};

  // map
  obj["chunkid"]  = chunkid;
  obj["coord"]    = coord;
  obj["boundary"] = boundary;

  return obj;
}

DEFINE_MEMBER3(json, to_json)()
{
  json obj;

  // meta data
  obj["size"]        = size;
  obj["ndim"]        = 3;
  obj["shape"]       = {dims[0], dims[1], dims[2]};
  obj["periodicity"] = {periodicity[0], periodicity[1], periodicity[2]};

  // map
  obj["chunkid"]  = chunkid;
  obj["coord"]    = coord;
  obj["boundary"] = boundary;

  return obj;
}

DEFINE_MEMBER1(void, from_json)(json& obj)
{
  if (obj["ndim"].get<int>() != 1) {
    ERROR << tfm::format("Invalid input to ChunkMap<1>::load_json");
  }

  // meta data
  size           = obj["size"].get<int>();
  dims[0]        = obj["shape"][0].get<int>();
  periodicity[0] = obj["periodicity"][0].get<int>();

  // map
  chunkid  = obj["chunkid"];
  coord    = obj["coord"];
  boundary = obj["boundary"].get<std::vector<int>>();
}

DEFINE_MEMBER2(void, from_json)(json& obj)
{
  if (obj["ndim"].get<int>() != 2) {
    ERROR << tfm::format("Invalid input to ChunkMap<2>::load_json");
  }

  // meta data
  size           = obj["size"].get<int>();
  dims[0]        = obj["shape"][0].get<int>();
  dims[1]        = obj["shape"][1].get<int>();
  periodicity[0] = obj["periodicity"][0].get<int>();
  periodicity[1] = obj["periodicity"][1].get<int>();

  // map
  chunkid  = obj["chunkid"];
  coord    = obj["coord"];
  boundary = obj["boundary"].get<std::vector<int>>();
}

DEFINE_MEMBER3(void, from_json)(json& obj)
{
  if (obj["ndim"].get<int>() != 3) {
    ERROR << tfm::format("Invalid input to ChunkMap<3>::load_json");
  }

  // meta data
  size           = obj["size"].get<int>();
  dims[0]        = obj["shape"][0].get<int>();
  dims[1]        = obj["shape"][1].get<int>();
  dims[2]        = obj["shape"][2].get<int>();
  periodicity[0] = obj["periodicity"][0].get<int>();
  periodicity[1] = obj["periodicity"][1].get<int>();
  periodicity[2] = obj["periodicity"][2].get<int>();

  // map
  chunkid  = obj["chunkid"];
  coord    = obj["coord"];
  boundary = obj["boundary"].get<std::vector<int>>();
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
