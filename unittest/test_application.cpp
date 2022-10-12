// -*- C++ -*-

#include "../application.hpp"
#include "../chunk.hpp"
#include "../chunkmap.hpp"

#include "../thirdparty/catch.hpp"

class TestChunk;

using namespace nix;
using BaseApp = Application<TestChunk, ChunkMap<3>>;

class TestChunk : public Chunk<3>
{
public:
  using Chunk<3>::Chunk;

  virtual float64 get_total_load() override
  {
    static std::random_device                      rd;
    static std::mt19937                            mt(rd());
    static std::uniform_real_distribution<float64> rand(0.75, +1.25);

    for(int i=0; i < load.size() ;i++) {
      load[i] = rand(mt);
    }

    return Chunk<3>::get_total_load();
  }
};

// class for testing Application
class TestApp : public BaseApp
{
public:
  TestApp()
  {
    mpi_init_with_nullptr = true;
  }

  virtual void diagnostic(std::ostream &out) override
  {
    // do nothing
  }

  virtual bool rebuild_chunkmap() override
  {
    bool status = BaseApp::rebuild_chunkmap();

    // check validity
    REQUIRE(validate_chunkmap() == true);

    return status;
  }

  int run_main()
  {
    // command-line arguments
    std::vector<std::string> args = {"TestApp", "-e", "1", "-c", "default.json"};

    cl_argc = args.size();
    cl_argv = new char *[args.size()];
    for (int i = 0; i < args.size(); i++) {
      cl_argv[i] = const_cast<char *>(args[i].c_str());
    }

    // main application loop
    REQUIRE(main(std::cout) == 0);

    delete[] cl_argv;

    return 0;
  }
};

//
// test validity of rebuild_chunkmap()
//
TEST_CASE_METHOD(TestApp, "TestApp")
{
  run_main();
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
