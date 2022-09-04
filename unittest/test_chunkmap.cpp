// -*- C++ -*-

#include <fstream>
#include <iostream>

#include "../chunkmap.hpp"
#include "../thirdparty/json.hpp"

#include "../thirdparty/catch.hpp"

//
// ChunkMap in 1D
//
TEST_CASE("ChunkMap1D")
{
  int         Cx = GENERATE(4, 8, 16, 32);
  ChunkMap<1> chunkmap(Cx);

  {
    const std::string filename = "test_chunkmap1.json";
    std::ofstream ofs(filename.c_str());
    chunkmap.json_save(ofs);
    ofs.close();
    std::remove(filename.c_str());
  }

  REQUIRE(chunkmap.validate());
}

//
// ChunkMap in 2D
//
TEST_CASE("ChunkMap2D")
{
  int         Cx = GENERATE(4, 6, 10, 20, 30);
  int         Cy = GENERATE(4, 6, 10, 20, 30);
  ChunkMap<2> chunkmap(Cy, Cx);

  {
    const std::string filename = "test_chunkmap2.json";
    std::ofstream ofs(filename.c_str());
    chunkmap.json_save(ofs);
    ofs.close();
    std::remove(filename.c_str());
  }

  REQUIRE(chunkmap.validate());
}

//
// ChunkMap in 3D
//
TEST_CASE("ChunkMap3D")
{
  int         Cx = GENERATE(4, 10, 30);
  int         Cy = GENERATE(4, 10, 30);
  int         Cz = GENERATE(4, 10, 30);
  ChunkMap<3> chunkmap(Cz, Cy, Cx);

  {
    const std::string filename = "test_chunkmap3.json";
    std::ofstream ofs(filename.c_str());
    chunkmap.json_save(ofs);
    ofs.close();
    std::remove(filename.c_str());
  }

  REQUIRE(chunkmap.validate());
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
