// -*- C++ -*-

#include "chunkmap.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include "catch.hpp"

using namespace nix;

template <int Ndim>
class ChunkMapTest : public ChunkMap<Ndim>
{
public:
  using ChunkMap<Ndim>::ChunkMap;
  using ChunkMap<Ndim>::dims;
  using ChunkMap<Ndim>::periodicity;

  void test_dimension(int nz, int ny, int nx)
  {
    REQUIRE(dims[0] == nz);
    REQUIRE(dims[1] == ny);
    REQUIRE(dims[2] == nx);
  }

  void test_periodicity(int pz, int py, int px)
  {
    REQUIRE(periodicity[0] == pz);
    REQUIRE(periodicity[1] == py);
    REQUIRE(periodicity[2] == px);
  }

  void test_get_neighbor_coord(int coord, int delta, int dir)
  {
    int first_coord = 0;
    int last_coord  = dims[dir] - 1;
    int nb_coord    = this->get_neighbor_coord(coord, delta, dir);

    if (periodicity[dir] == 1) {
      if (coord == first_coord && delta == -1) {
        REQUIRE(nb_coord == last_coord);
        return;
      }

      if (coord == last_coord && delta == +1) {
        REQUIRE(nb_coord == first_coord);
        return;
      }
    } else {
      if (coord == first_coord && delta == -1) {
        REQUIRE(nb_coord == MPI_PROC_NULL);
        return;
      }

      if (coord == last_coord && delta == +1) {
        REQUIRE(nb_coord == MPI_PROC_NULL);
        return;
      }
    }

    REQUIRE(nb_coord == coord + delta);
  }
};

TEST_CASE("Initialization")
{
  int Cx = GENERATE(4, 10);
  int Cy = GENERATE(1, 4, 10);
  int Cz = GENERATE(1, 4, 10);

  SECTION("1D")
  {
    ChunkMapTest<1> chunkmap(Cx);
    REQUIRE(chunkmap.validate());
  }

  SECTION("2D")
  {
    ChunkMapTest<2> chunkmap(Cy, Cx);
    REQUIRE(chunkmap.validate());
  }

  SECTION("3D")
  {
    ChunkMapTest<3> chunkmap(Cz, Cy, Cx);
    REQUIRE(chunkmap.validate());
  }
}

TEST_CASE("Dimension")
{
  int Cx = GENERATE(4, 10);
  int Cy = GENERATE(1, 4, 10);
  int Cz = GENERATE(1, 4, 10);

  SECTION("1D")
  {
    ChunkMapTest<1> chunkmap(Cx);
    chunkmap.test_dimension(Cx, 1, 1);
  }

  SECTION("2D")
  {
    ChunkMapTest<2> chunkmap(Cy, Cx);
    chunkmap.test_dimension(Cy, Cx, 1);
  }

  SECTION("3D")
  {
    ChunkMapTest<3> chunkmap(Cz, Cy, Cx);
    chunkmap.test_dimension(Cz, Cy, Cx);
  }
}

TEST_CASE("Periodicity")
{
  int Cx = GENERATE(4, 10);
  int Cy = GENERATE(1, 4, 10);
  int Cz = GENERATE(1, 4, 10);

  SECTION("1D")
  {
    ChunkMapTest<1> chunkmap(Cx);
    chunkmap.test_periodicity(1, 1, 1);

    chunkmap.set_periodicity(0, 1, 1);
    chunkmap.test_periodicity(0, 1, 1);
  }

  SECTION("2D")
  {
    ChunkMapTest<2> chunkmap(Cy, Cx);
    chunkmap.test_periodicity(1, 1, 1);

    chunkmap.set_periodicity(1, 0, 1);
    chunkmap.test_periodicity(1, 0, 1);
  }

  SECTION("3D")
  {
    ChunkMapTest<3> chunkmap(Cz, Cy, Cx);
    chunkmap.test_periodicity(1, 1, 1);

    chunkmap.set_periodicity(1, 1, 0);
    chunkmap.test_periodicity(1, 1, 0);
  }
}

TEST_CASE("get_neighbor_coord")
{
  int Cx = GENERATE(4, 10);
  int Cy = GENERATE(1, 4, 10);
  int Cz = GENERATE(1, 4, 10);

  ChunkMapTest<3> chunkmap(Cz, Cy, Cx);

  SECTION("X")
  {
    chunkmap.test_get_neighbor_coord(0, +1, 2);
    chunkmap.test_get_neighbor_coord(0, -1, 2);
    chunkmap.test_get_neighbor_coord(Cx / 2, +1, 2);
    chunkmap.test_get_neighbor_coord(Cx / 2, -1, 2);
    chunkmap.test_get_neighbor_coord(Cx - 1, +1, 2);
    chunkmap.test_get_neighbor_coord(Cx - 1, -1, 2);

    // non-periodic
    chunkmap.set_periodicity(0, 1, 1);
    chunkmap.test_get_neighbor_coord(0, -1, 2);
    chunkmap.test_get_neighbor_coord(Cx - 1, +1, 2);
  }

  SECTION("Y")
  {
    chunkmap.test_get_neighbor_coord(0, +1, 1);
    chunkmap.test_get_neighbor_coord(0, -1, 1);
    chunkmap.test_get_neighbor_coord(Cy / 2, +1, 1);
    chunkmap.test_get_neighbor_coord(Cy / 2, -1, 1);
    chunkmap.test_get_neighbor_coord(Cy - 1, +1, 1);
    chunkmap.test_get_neighbor_coord(Cy - 1, -1, 1);

    // non-periodic
    chunkmap.set_periodicity(1, 0, 1);
    chunkmap.test_get_neighbor_coord(0, -1, 1);
    chunkmap.test_get_neighbor_coord(Cy - 1, +1, 1);
  }

  SECTION("Z")
  {
    chunkmap.test_get_neighbor_coord(0, +1, 0);
    chunkmap.test_get_neighbor_coord(0, -1, 0);
    chunkmap.test_get_neighbor_coord(Cz / 2, +1, 0);
    chunkmap.test_get_neighbor_coord(Cz / 2, -1, 0);
    chunkmap.test_get_neighbor_coord(Cz - 1, +1, 0);
    chunkmap.test_get_neighbor_coord(Cz - 1, -1, 0);

    // non-periodic
    chunkmap.set_periodicity(1, 1, 0);
    chunkmap.test_get_neighbor_coord(0, -1, 0);
    chunkmap.test_get_neighbor_coord(Cz - 1, +1, 0);
  }
}

TEST_CASE("Save to and load from file")
{
  const std::string filename = "test_chunkmap.json";

  int Cx = GENERATE(4, 10);
  int Cy = GENERATE(1, 4, 10);
  int Cz = GENERATE(1, 4, 10);

  json obj1;
  json obj2;

  SECTION("1D")
  {
    ChunkMapTest<1> chunkmap(Cx);

    // save
    {
      chunkmap.save_json(obj1);

      std::ofstream ofs(filename);
      ofs << std::setw(2) << obj1;
    }

    // load
    {
      std::ifstream ifs(filename);
      ifs >> obj2;

      chunkmap.load_json(obj2);
    }

    // check for load
    REQUIRE(chunkmap.validate());

    // cleanup
    std::remove(filename.c_str());
  }

  SECTION("2D")
  {
    ChunkMapTest<2> chunkmap(Cy, Cx);

    // save
    {
      chunkmap.save_json(obj1);

      std::ofstream ofs(filename);
      ofs << std::setw(2) << obj1;
    }

    // load
    {
      std::ifstream ifs(filename);
      ifs >> obj2;

      chunkmap.load_json(obj2);
    }

    // check for load
    REQUIRE(chunkmap.validate());

    // cleanup
    std::remove(filename.c_str());
  }

  SECTION("3D")
  {
    ChunkMapTest<3> chunkmap(Cz, Cy, Cx);

    // save
    {
      chunkmap.save_json(obj1);

      std::ofstream ofs(filename);
      ofs << std::setw(2) << obj1;
    }

    // load
    {
      std::ifstream ifs(filename);
      ifs >> obj2;

      chunkmap.load_json(obj2);
    }

    // check for load
    REQUIRE(chunkmap.validate());

    // cleanup
    std::remove(filename.c_str());
  }
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
