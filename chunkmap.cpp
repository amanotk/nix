// -*- C++ -*-
#include "chunkmap.hpp"

#define DEFINE_MEMBER(type, name)                                                                  \
  template <int N>                                                                                 \
  type ChunkMap<N>::name
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

DEFINE_MEMBER(void, save)(json &obj, MPI_File *fh, size_t *disp)
{
  int32_t nd;
  int32_t sh[3];
  size_t  datasize;

  datasize = sizeof(int);
  jsonio::put_attribute(obj, "size", *disp, size);
  jsonio::write_single(fh, disp, &size, datasize);

  datasize = sizeof(int) * 3;
  jsonio::put_attribute(obj, "dims", *disp, 3, dims);
  jsonio::write_single(fh, disp, &dims[0], datasize);

  nd       = 1;
  sh[0]    = size;
  datasize = sizeof(int) * rank.size();
  jsonio::put_metadata(obj, "rank", "i4", "", *disp, datasize, nd, sh);
  jsonio::write_single(fh, disp, rank.data(), datasize);

  nd       = 2;
  sh[0]    = size;
  sh[1]    = N;
  datasize = sizeof(int) * coord.size();
  jsonio::put_metadata(obj, "coord", "i4", "", *disp, datasize, nd, sh);
  jsonio::write_single(fh, disp, coord.data(), datasize);

  nd       = N;
  sh[0]    = dims[0];
  sh[1]    = dims[1];
  sh[2]    = dims[2];
  datasize = sizeof(int) * chunkid.size();
  jsonio::put_metadata(obj, "chunkid", "i4", "", *disp, datasize, nd, sh);
  jsonio::write_single(fh, disp, chunkid.data(), datasize);
}

DEFINE_MEMBER1(void, json_save)(std::ostream &out)
{
  json obj;

  // meta data
  {
    obj["size"]  = size;
    obj["ndim"]  = 1;
    obj["shape"] = {dims[0]};
  }

  // id
  {
    json chunkid_obj = json::array();

    for (int ix = 0; ix < dims[1]; ix++) {
      chunkid_obj.push_back(chunkid(ix));
    }
    obj["chunkid"] = chunkid_obj;
  }

  // coordinate
  {
    json coord_obj = json::array();

    for (int id = 0; id < size; id++) {
      int ix;
      get_coordinate(id, ix);
      coord_obj.push_back({ix});
    }
    obj["coord"] = coord_obj;
  }

  // rank
  {
    json rank_obj = json::array();

    for (int id = 0; id < size; id++) {
      int rank = get_rank(id);
      rank_obj.push_back(rank);
    }
    obj["rank"] = rank_obj;
  }

  // output
  {
    json root;
    root = {{"chunkmap", obj}};
    out << std::setw(2) << root << std::endl;
  }
}

DEFINE_MEMBER2(void, json_save)(std::ostream &out)
{
  json obj;

  // meta data
  {
    obj["size"]  = size;
    obj["ndim"]  = 2;
    obj["shape"] = {dims[0], dims[1]};
  }

  // id
  {
    json chunkid_obj = json::array();

    for (int iy = 0; iy < dims[0]; iy++) {
      json cx = json::array();
      for (int ix = 0; ix < dims[1]; ix++) {
        cx.push_back(chunkid(iy, ix));
      }
      chunkid_obj.push_back(cx);
    }
    obj["chunkid"] = chunkid_obj;
  }

  // coordinate
  {
    json coord_obj = json::array();

    for (int id = 0; id < size; id++) {
      int ix, iy;
      get_coordinate(id, iy, ix);
      coord_obj.push_back({iy, ix});
    }
    obj["coord"] = coord_obj;
  }

  // rank
  {
    json rank_obj = json::array();

    for (int id = 0; id < size; id++) {
      int rank = get_rank(id);
      rank_obj.push_back(rank);
    }
    obj["rank"] = rank_obj;
  }

  // output
  {
    json root;
    root = {{"chunkmap", obj}};
    out << std::setw(2) << root << std::endl;
  }
}

DEFINE_MEMBER3(void, json_save)(std::ostream &out)
{
  json obj;

  // meta data
  {
    obj["size"]  = size;
    obj["ndim"]  = 3;
    obj["shape"] = {dims[0], dims[1], dims[2]};
  }

  // id
  {
    json chunkid_obj = json::array();

    for (int iz = 0; iz < dims[0]; iz++) {
      json cy = json::array();
      for (int iy = 0; iy < dims[1]; iy++) {
        json cx = json::array();
        for (int ix = 0; ix < dims[2]; ix++) {
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

    for (int id = 0; id < size; id++) {
      int ix, iy, iz;
      get_coordinate(id, iz, iy, ix);
      coord_obj.push_back({iz, iy, ix});
    }
    obj["coord"] = coord_obj;
  }

  // rank
  {
    json rank_obj = json::array();

    for (int id = 0; id < size; id++) {
      int rank = get_rank(id);
      rank_obj.push_back(rank);
    }
    obj["rank"] = rank_obj;
  }

  // output
  {
    json root;
    root = {{"chunkmap", obj}};
    out << std::setw(2) << root << std::endl;
  }
}

// explicit instantiation
template class ChunkMap<1>;
template class ChunkMap<2>;
template class ChunkMap<3>;

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
