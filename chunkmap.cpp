// -*- C++ -*-
#include "chunkmap.hpp"

#define DEFINE_MEMBER(type, name)                                                                  \
  template <int Ndim>                                                                              \
  type ChunkMap<Ndim>::name
#define DEFINE_MEMBER1(type, name)                                                                 \
  template <>                                                                                      \
  type ChunkMap<1>::name
#define DEFINE_MEMBER2(type, name)                                                                 \
  template <>                                                                                      \
  type ChunkMap<2>::name
#define DEFINE_MEMBER3(type, name)                                                                 \
  template <>                                                                                      \
  type ChunkMap<3>::name

DEFINE_MEMBER1(, ChunkMap)(const int Cx)
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

DEFINE_MEMBER2(, ChunkMap)(const int Cy, const int Cx)
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

DEFINE_MEMBER3(, ChunkMap)(const int Cz, const int Cy, const int Cx)
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

DEFINE_MEMBER1(void, save_json)(json &obj)
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

DEFINE_MEMBER2(void, save_json)(json &obj)
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

DEFINE_MEMBER3(void, save_json)(json &obj)
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

DEFINE_MEMBER1(void, load_json)(json &obj)
{
  if (obj["ndim"].get<int>() != 1) {
    ERRORPRINT("Invalid input to ChunkMap<1>::load_json\n");
  }

  // meta data
  size    = obj["size"].get<int>();
  dims[0] = obj["shape"][0].get<int>();

  // map
  chunkid = obj["chunkid"];
  coord   = obj["coord"];
  rank    = obj["rank"];
}

DEFINE_MEMBER2(void, load_json)(json &obj)
{
  if (obj["ndim"].get<int>() != 2) {
    ERRORPRINT("Invalid input to ChunkMap<2>::load_json\n");
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

DEFINE_MEMBER3(void, load_json)(json &obj)
{
  if (obj["ndim"].get<int>() != 3) {
    ERRORPRINT("Invalid input to ChunkMap<3>::load_json\n");
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

// explicit instantiation
template class ChunkMap<1>;
template class ChunkMap<2>;
template class ChunkMap<3>;

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
