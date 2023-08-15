// -*- C++ -*-

#include "chunkvec.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include "catch.hpp"

using namespace nix;

class MockChunkMap
{
private:
  int dims[3];

public:
  MockChunkMap(int cz, int cy, int cx) : dims{cz, cy, cx}
  {
  }

  int get_chunkid(int cz, int cy, int cx)
  {
    return 0;
  }

  int get_rank(int id)
  {
    return 0;
  }

  int get_neighbor_coord(int coord, int delta, int dir)
  {
    return 0;
  }

  void get_coordinate(int id, int& cz, int& cy, int& cx)
  {
    cx = 0;
    cy = 0;
    cz = 0;
  }
};

class MockChunk
{
private:
  int myid;

public:
  MockChunk(int id = 0) : myid(id)
  {
  }

  int get_id() const
  {
    return myid;
  }

  void set_nb_id(int dirz, int diry, int dirx, int id)
  {
  }

  void set_nb_rank(int dirz, int diry, int dirx, int rank)
  {
  }
};

class ChunkVecTest : public ChunkVec<std::unique_ptr<MockChunk>>
{
public:
  bool is_sorted()
  {
    return std::is_sorted(this->begin(), this->end(),
                          [](const auto& x, const auto& y) { return x->get_id() < y->get_id(); });
  }
};

TEST_CASE("sort_and_shrink")
{
  using MockChunkVec = std::vector<std::unique_ptr<MockChunk>>;
  const int size     = 10;
  const int max_id   = 4;

  ChunkVecTest chunktest;
  MockChunkVec chunkmock;

  for (int i = 0; i < size; i++) {
    chunktest.push_back(std::make_unique<MockChunk>(i));
    chunkmock.push_back(std::make_unique<MockChunk>(i));
  }

  SECTION("preconditioning")
  {
    REQUIRE(size == chunktest.size());
    REQUIRE(size == chunkmock.size());

    for (int i = 0; i < size; i++) {
      REQUIRE(chunktest[i]->get_id() == chunkmock[i]->get_id());
    }
  }

  SECTION("with pre-sorted data")
  {
    chunktest.sort_and_shrink(max_id);
    REQUIRE(chunktest.is_sorted());
    REQUIRE(max_id + 1 == chunktest.size());
    REQUIRE(max_id + 1 == chunktest.capacity());
  }

  SECTION("with shuffled data")
  {
    // random shuffle
    std::random_device seed_gen;
    std::mt19937       engine(seed_gen());
    std::shuffle(chunktest.begin(), chunktest.end(), engine);

    chunktest.sort_and_shrink(max_id);
    REQUIRE(chunktest.is_sorted());
    REQUIRE(max_id + 1 == chunktest.size());
    REQUIRE(max_id + 1 == chunktest.capacity());
  }
}

TEST_CASE("set_neighbors")
{
  int Cx = GENERATE(1, 4);
  int Cy = GENERATE(1, 4);
  int Cz = GENERATE(1, 4);

  std::unique_ptr<MockChunkMap> chunkmap = std::make_unique<MockChunkMap>(Cz, Cy, Cx);
  ChunkVecTest                  chunktest;

  for (int i = 0; i < Cz * Cy * Cx; i++) {
    chunktest.push_back(std::make_unique<MockChunk>(i));
  }

  // just check if call does not crash
  chunktest.set_neighbors(chunkmap);
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
