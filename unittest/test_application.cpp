// -*- C++ -*-

#include "application.hpp"
#include "chunk.hpp"
#include "chunkmap.hpp"

#include "catch.hpp"

using Chunk    = BaseChunk<3>;
using ChunkMap = BaseChunkMap<3>;

// class for testing BaseApplication
class TestApplication : public BaseApplication<Chunk, ChunkMap>
{
public:
  TestApplication()
  {
    mpi_init_with_nullptr = false;
  }

  virtual void rebuild_chunkmap() override
  {
    BaseApplication<Chunk, ChunkMap>::rebuild_chunkmap();

    // check validity
    REQUIRE(validate_chunkmap() == true);
  }

  int run_main()
  {
    // command-line arguments
    std::vector<std::string> args = {"TestApplication", "-e", "1", "-c", "default.json"};

    cl_argc = args.size();
    cl_argv = new char *[args.size()];
    for (int i = 0; i < args.size(); i++) {
      cl_argv[i] = const_cast<char *>(args[i].c_str());
    }

    REQUIRE(main(std::cout) == 0);

    delete[] cl_argv;
    return 0;
  }
};

//
// test validity of rebuild_chunkmap()
//
TEST_CASE_METHOD(TestApplication, "TestApplication")
{
  run_main();
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
