// -*- C++ -*-

#include "statehandler.hpp"

#include <iostream>

#include "catch.hpp"

using namespace nix;

class MockChunk;
using ChunkVec = std::vector<std::unique_ptr<MockChunk>>;

static const int ndata    = 10;
static int       ndims[4] = {8, 8, 8, 8 * 8 * 8};
static int       cdims[4] = {2, 2, 2, 2 * 2 * 2};

static float64 get_x_value(int id, int index)
{
  return 100 * id + index;
}

class MockChunk
{
private:
  int     myid;
  float64 myload;

public:
  std::vector<float64> x;

  MockChunk(int dims[4], int id) : myid(id)
  {
    x.resize(ndata);
    std::fill(x.begin(), x.end(), 0.0);
  }

  int get_id()
  {
    return myid;
  }

  int pack(void* buffer, int address)
  {
    int count = address;

    count += memcpy_count(buffer, &myid, sizeof(int), count, 0);
    count += memcpy_count(buffer, x.data(), sizeof(float64) * ndata, count, 0);

    return count;
  }

  int unpack(void* buffer, int address)
  {
    int count = address;

    count += memcpy_count(&myid, buffer, sizeof(int), 0, count);
    count += memcpy_count(x.data(), buffer, sizeof(float64) * ndata, 0, count);

    return count;
  }
};

class MockApplication
{
private:
  int      thisrank;
  int      nprocess;
  double   x;
  double   y;
  double   z;
  ChunkVec chunkvec;

  struct InternalData {
    int*      ndims;
    int*      cdims;
    int&      nprocess;
    int&      thisrank;
    ChunkVec& chunkvec;
  };

  InternalData get_internal_data()
  {
    return {ndims, cdims, nprocess, thisrank, chunkvec};
  }

public:
  MockApplication()
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &thisrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocess);

    x = 1.0;
    y = sqrt(2.0);
    z = exp(3.0);
  }

  std::unique_ptr<MockChunk> create_chunk(int dims[4], int id)
  {
    return std::make_unique<MockChunk>(dims, id);
  }

  json to_json()
  {
    json state = {{"timestamp", nix::wall_clock()},
                  {"ndims", ndims},
                  {"cdims", cdims},
                  {"x", x},
                  {"y", y},
                  {"z", z},
                  {"nprocess", nprocess},
                  {"thisrank", thisrank}};

    return state;
  }

  bool from_json(json& state)
  {
    json current_state = to_json();

    bool consistency = true;

    consistency &= current_state["ndims"] == state["ndims"];
    consistency &= current_state["cdims"] == state["cdims"];
    consistency &= current_state["x"] == state["x"];
    consistency &= current_state["y"] == state["y"];
    consistency &= current_state["z"] == state["z"];
    consistency &= current_state["nprocess"] == state["nprocess"];

    return consistency;
  }

  void prepare_chunkvec(int numchunk)
  {
    chunkvec.resize(0);
    chunkvec.shrink_to_fit();

    for (int i = 0; i < numchunk; i++) {
      int id = i + thisrank * numchunk;
      chunkvec.push_back(create_chunk(ndims, id));

      // fill data
      for (int j = 0; j < ndata; j++) {
        chunkvec[i]->x[j] = get_x_value(id, j);
      }
    }
  }

  bool validate_chunkvec(int numchunk)
  {
    bool status = true;

    status &= numchunk == chunkvec.size();
    for (int i = 0; i < numchunk; i++) {
      int id = i + thisrank * numchunk;
      status &= chunkvec[i]->get_id() == id;

      for (int j = 0; j < ndata; j++) {
        status &= chunkvec[i]->x[j] == get_x_value(id, j);
      }
    }

    return status;
  }

  void test_save_load_application()
  {
    StateHandler statehandler;

    // save
    bool save = statehandler.save_application(*this, get_internal_data(), "foo");
    MPI_Barrier(MPI_COMM_WORLD);

    // load
    bool load = statehandler.load_application(*this, get_internal_data(), "foo");
    MPI_Barrier(MPI_COMM_WORLD);

    // cleanup
    if (thisrank == 0) {
      std::filesystem::remove("foo.msgpack");
    }

    REQUIRE(save == true);
    REQUIRE(load == true);
  }

  void test_save_load_chunkvec()
  {
    const int numchunk = 10;

    StateHandler statehandler;

    prepare_chunkvec(numchunk);

    // save
    bool save = statehandler.save_chunkvec(*this, get_internal_data(), "foo");
    MPI_Barrier(MPI_COMM_WORLD);

    // load
    bool load = statehandler.load_chunkvec(*this, get_internal_data(), "foo");
    MPI_Barrier(MPI_COMM_WORLD);

    REQUIRE(validate_chunkvec(numchunk) == true);

    // cleanup
    if (thisrank == 0) {
      std::filesystem::remove_all("foo");
    }

    REQUIRE(save == true);
    REQUIRE(load == true);
  }
};

TEST_CASE("test_save_load_application")
{
  MockApplication app;
  app.test_save_load_application();
}

TEST_CASE("test_save_load_chunkvec")
{
  MockApplication app;
  app.test_save_load_chunkvec();
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
