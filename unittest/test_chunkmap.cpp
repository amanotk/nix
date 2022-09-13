// -*- C++ -*-

#include <fstream>
#include <iostream>

#include "../chunkmap.hpp"
#include "../thirdparty/json.hpp"

#include "../thirdparty/catch.hpp"

using json = nlohmann::ordered_json;

//
// ChunkMap in 1D
//
TEST_CASE("ChunkMap1D")
{
  const std::string filename = "test_chunkmap1.json";
  const int         Cx       = GENERATE(4, 8, 16, 32);

  ChunkMap<1> chunkmap(Cx);
  json        obj1;
  json        obj2;

  // check for initialization
  REQUIRE(chunkmap.validate());

  // save
  {
    chunkmap.save_json(obj1);

    std::ofstream ofs(filename);
    ofs << obj1;
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

//
// ChunkMap in 2D
//
TEST_CASE("ChunkMap2D")
{
  const std::string filename = "test_chunkmap2.json";
  const int         Cx       = GENERATE(4, 6, 10, 20, 30);
  const int         Cy       = GENERATE(4, 6, 10, 20, 30);

  ChunkMap<2> chunkmap(Cy, Cx);
  json        obj1;
  json        obj2;

  // check for initialization
  REQUIRE(chunkmap.validate());

  // save
  {
    chunkmap.save_json(obj1);

    std::ofstream ofs(filename);
    ofs << obj1;
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

//
// ChunkMap in 3D
//
TEST_CASE("ChunkMap3D")
{
  const std::string filename = "test_chunkmap3.json";
  int               Cx       = GENERATE(4, 10, 30);
  int               Cy       = GENERATE(4, 10, 30);
  int               Cz       = GENERATE(4, 10, 30);

  ChunkMap<3> chunkmap(Cz, Cy, Cx);
  json        obj1;
  json        obj2;

  // check for initialization
  REQUIRE(chunkmap.validate());

  // save
  {
    chunkmap.save_json(obj1);

    std::ofstream ofs(filename);
    ofs << obj1;
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

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
