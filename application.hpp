// -*- C++ -*-
#ifndef _APPLICATION_HPP_
#define _APPLICATION_HPP_

#include "argparser.hpp"
#include "balancer.hpp"
#include "buffer.hpp"
#include "cfgparser.hpp"
#include "chunkvector.hpp"
#include "logger.hpp"
#include "mpistream.hpp"
#include "nix.hpp"
#include "tinyformat.hpp"
#include <nlohmann/json.hpp>

NIX_NAMESPACE_BEGIN

///
/// @brief Base Application class
/// @tparam Chunk Chunk type
/// @tparam ChunkMap ChunkMap type
///
template <typename Chunk, typename ChunkMap>
class Application
{
protected:
  using ThisType     = Application<Chunk, ChunkMap>;
  using PtrArgParser = std::unique_ptr<ArgParser>;
  using PtrCfgParser = std::unique_ptr<CfgParser>;
  using PtrBalancer  = std::unique_ptr<Balancer>;
  using PtrLogger    = std::unique_ptr<Logger>;
  using PtrChunkMap  = std::unique_ptr<ChunkMap>;
  using PtrChunk     = std::unique_ptr<Chunk>;
  using ChunkVec     = ChunkVector<PtrChunk>;

  PtrArgParser argparser; ///< argument parser
  PtrCfgParser cfgparser; ///< configuration parser
  PtrBalancer  balancer;  ///< load balancer
  PtrLogger    logger;    ///< logger
  PtrChunkMap  chunkmap;  ///< chunkmap
  ChunkVec     chunkvec;  ///< local chunks

  int     cl_argc;  ///< command-line argc
  char**  cl_argv;  ///< command-line argv
  float64 wclock;   ///< wall clock time at initialization
  int     ndims[4]; ///< global grid dimensions
  int     cdims[4]; ///< chunk dimensions
  int     curstep;  ///< current iteration step
  float64 curtime;  ///< current time
  float64 delt;     ///< time step
  float64 delx;     ///< grid size in x
  float64 dely;     ///< grid size in y
  float64 delz;     ///< grid size in z
  float64 cc;       ///< speed of light
  float64 xlim[3];  ///< physical domain in x
  float64 ylim[3];  ///< physical domain in y
  float64 zlim[3];  ///< physical domain in z

  // MPI related
  int  nprocess;              ///< number of mpi processes
  int  thisrank;              ///< my rank
  bool mpi_init_with_nullptr; ///< for testing purpose

  ///
  /// @brief internal data struct
  ///
  struct InternalData {
    int*         ndims;
    int*         cdims;
    int&         nprocess;
    int&         thisrank;
    int&         curstep;
    float64&     curtime;
    PtrChunkMap& chunkmap;
    ChunkVec&    chunkvec;
  };

  ///
  /// @brief return internal data struct
  ///
  InternalData get_internal_data()
  {
    return {ndims, cdims, nprocess, thisrank, curstep, curtime, chunkmap, chunkvec};
  }

public:
  /// @brief default constructor
  Application() : mpi_init_with_nullptr(false)
  {
  }

  ///
  /// @brief constructor
  /// @param argc number of arguments
  /// @param argv array of arguments
  ///
  Application(int argc, char** argv) : mpi_init_with_nullptr(false)
  {
    cl_argc = argc;
    cl_argv = argv;
  }

  ///
  /// @brief main loop of simulation
  /// @return return code of application
  ///
  virtual int main();

  ///
  /// @brief factory to create argument parser
  /// @return parser object
  ///
  virtual std::unique_ptr<ArgParser> create_argparser()
  {
    return std::make_unique<ArgParser>();
  }

  ///
  /// @brief factory to create config parser
  /// @return parser object
  ///
  virtual std::unique_ptr<CfgParser> create_cfgparser()
  {
    return std::make_unique<CfgParser>();
  }

  ///
  /// @brief factory to create balancer
  /// @return balancer object
  ///
  virtual std::unique_ptr<Balancer> create_balancer()
  {
    auto parameter = cfgparser->get_parameter();

    int Cx = parameter.value("Cx", 1);
    int Cy = parameter.value("Cy", 1);
    int Cz = parameter.value("Cz", 1);
    return std::make_unique<Balancer>(Cz * Cy * Cx);
  }

  ///
  /// @brief factory to create logger
  /// @return logger object
  ///
  virtual std::unique_ptr<Logger> create_logger()
  {
    auto application = cfgparser->get_application();

    return std::make_unique<Logger>(thisrank, application["log"]);
  }

  ///
  /// @brief factory to create chunkmap
  /// @return chunkmap object
  ///
  virtual std::unique_ptr<ChunkMap> create_chunkmap()
  {
    auto parameter = cfgparser->get_parameter();

    int Cx = parameter.value("Cx", 1);
    int Cy = parameter.value("Cy", 1);
    int Cz = parameter.value("Cz", 1);
    return std::make_unique<ChunkMap>(Cz, Cy, Cx);
  }

  ///
  /// @brief factory to create chunk object
  /// @param dims local number of grids in each direction
  /// @param id chunk ID
  /// @return chunk object
  ///
  virtual std::unique_ptr<Chunk> create_chunk(const int dims[], int id)
  {
    return std::make_unique<Chunk>(dims, id);
  }

protected:
  ///
  /// @brief initialize application
  /// @param argc number of arguments
  /// @param argv array of arguments
  ///
  virtual void initialize(int argc, char** argv);

  ///
  /// @brief finalize application
  ///
  virtual void finalize();

  ///
  /// @brief initialize MPI
  /// @param argc number of arguments
  /// @param argv array of arguments
  ///
  void initialize_mpi(int* argc, char*** argv);

  ///
  /// @brief finalize MPI
  ///
  void finalize_mpi();

  ///
  /// @brief initialize debug printing
  ///
  void initialize_debugprinting();

  ///
  /// @brief initialize dimensions
  ///
  virtual void initialize_dimensions();

  ///
  /// @brief initialize domain
  ///
  virtual void initialize_domain();

  ///
  /// @brief initialize work load array
  ///
  virtual void initialize_workload();

  ///
  /// @brief initialize chunks
  ///
  virtual void initialize_chunks();

  ///
  /// @brief check the validity of chunks
  /// @return true if the chunks are appropriate
  ///
  virtual bool validate_chunks();

  ///
  /// @brief save profile of run
  ///
  virtual void save_profile();

  ///
  /// @brief performing load balancing
  /// @return return true if rebalancing is performed and false otherwise
  ///
  virtual bool rebalance();

  ///
  /// @brief load a snapshot file for restart
  ///
  virtual void load_snapshot()
  {
  }

  ///
  /// @brief save current state to a snapshot file (for restart)
  ///
  virtual void save_snapshot()
  {
  }

  ///
  /// @brief setup initial conditions
  ///
  virtual void setup()
  {
  }

  ///
  /// @brief perform various diagnostics output
  ///
  virtual void diagnostic()
  {
  }

  ///
  /// @brief advance physical quantities by one step
  ///
  virtual void push()
  {
  }

  ///
  /// @brief check if further push is needed or not
  /// @return true if the maximum physical time is not yet reached and false otherwise
  ///
  virtual bool is_push_needed()
  {
    if (curtime < argparser->get_physical_time_max() + delt) {
      return true;
    }
    return false;
  }

  ///
  /// @brief get available elapsed time
  /// @return available elapsed time in second
  ///
  virtual float64 get_available_etime()
  {
    float64 etime;

    if (thisrank == 0) {
      etime = wall_clock() - wclock;
    }
    MPI_Bcast(&etime, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return argparser->get_elapsed_time_max() - etime;
  }

  ///
  /// @brief take log
  ///
  virtual void take_log()
  {
    // timestamp
    json log = {{"unixtime", nix::wall_clock()}};
    logger->append(curstep, "timestamp", log);

    logger->log(curstep);
  }

  ///
  /// @brief increment step and physical time
  ///
  virtual void increment_time()
  {
    curtime += delt;
    curstep++;
  }
};

//
// implementation follows
//

#define DEFINE_MEMBER(type, name)                                                                  \
  template <typename Chunk, typename ChunkMap>                                                     \
  type Application<Chunk, ChunkMap>::name

DEFINE_MEMBER(int, main)()
{
  //
  // initialize the application
  //
  initialize(cl_argc, cl_argv);
  DEBUG1 << tfm::format("initialize");

  //
  // set initial condition
  //
  setup();
  DEBUG1 << tfm::format("setup");

  //
  // save profile
  //
  save_profile();

  //
  // main loop
  //
  while (is_push_needed()) {
    //
    // output diagnostics
    //
    diagnostic();
    DEBUG1 << tfm::format("step[%s] diagnostic", format_step(curstep));

    //
    // advance physical quantities by one step
    //
    push();
    DEBUG1 << tfm::format("step[%s] push", format_step(curstep));

    //
    // perform rebalance
    //
    rebalance();
    DEBUG1 << tfm::format("step[%s] rebalance", format_step(curstep));

    //
    // take log
    //
    take_log();
    DEBUG1 << tfm::format("step[%s] logging", format_step(curstep));

    //
    // increment step and time
    //
    increment_time();

    //
    // exit if elapsed time exceeds the limit
    //
    if (get_available_etime() < 0) {
      DEBUG1 << tfm::format("step[%s] run out of time", format_step(curstep));
      break;
    }
  }

  //
  // finalize the application
  //
  DEBUG1 << tfm::format("finalize");
  finalize();

  return 0;
}

DEFINE_MEMBER(void, initialize)(int argc, char** argv)
{
  // parse command line arguments
  argparser = create_argparser();
  argparser->parse(argc, argv);

  // parse configuration file
  cfgparser = create_cfgparser();
  cfgparser->parse_file(argparser->get_config());

  // initialize MPI first
  initialize_mpi(&argc, &argv);

  // object initialization
  chunkmap = create_chunkmap();
  logger   = create_logger();
  balancer = create_balancer();

  // misc
  curstep = 0;
  curtime = 0.0;
  initialize_debugprinting();
  initialize_dimensions();
  initialize_domain();
  initialize_workload();

  // chunks
  initialize_chunks();
}

DEFINE_MEMBER(void, finalize)()
{
  this->save_snapshot();

  logger->flush();

  finalize_mpi();
}

DEFINE_MEMBER(void, initialize_mpi)(int* argc, char*** argv)
{
  // initialize MPI with thread support
  {
    int thread_required = MPI_THREAD_SERIALIZED;
    int thread_provided = -1;

    if (mpi_init_with_nullptr == true) {
      MPI_Init_thread(nullptr, nullptr, thread_required, &thread_provided);
    } else {
      MPI_Init_thread(argc, argv, thread_required, &thread_provided);
    }

    if (thread_provided < thread_required) {
      ERROR << tfm::format("Your MPI does not support thread!");
      MPI_Finalize();
      exit(-1);
    }
  }

  MPI_Comm_size(MPI_COMM_WORLD, &nprocess);
  MPI_Comm_rank(MPI_COMM_WORLD, &thisrank);

  // store initial clock
  if (thisrank == 0) {
    wclock = wall_clock();
  }
  MPI_Bcast(&wclock, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // redirect stdout/stderr
  {
    json        config            = cfgparser->get_application()["mpistream"];
    std::string path              = "";
    int         max_files_per_dir = 1024;

    if (config.is_null() == false) {
      path              = config.value("path", path);
      max_files_per_dir = config.value("max_files_per_dir", max_files_per_dir);
    }

    MpiStream::initialize(path, max_files_per_dir);
  }
}

DEFINE_MEMBER(void, finalize_mpi)()
{
  // release stdout/stderr
  MpiStream::finalize();

  MPI_Finalize();
}

DEFINE_MEMBER(void, initialize_debugprinting)()
{
  DebugPrinter::init();
  DebugPrinter::set_level(argparser->get_verbosity());
}

DEFINE_MEMBER(void, initialize_dimensions)()
{
  json parameter = cfgparser->get_parameter();

  int nx = parameter.value("Nx", 1);
  int ny = parameter.value("Ny", 1);
  int nz = parameter.value("Nz", 1);
  int cx = parameter.value("Cx", 1);
  int cy = parameter.value("Cy", 1);
  int cz = parameter.value("Cz", 1);

  ndims[0] = nz;
  ndims[1] = ny;
  ndims[2] = nx;
  ndims[3] = ndims[0] * ndims[1] * ndims[2];
  cdims[0] = cz;
  cdims[1] = cy;
  cdims[2] = cx;
  cdims[3] = cdims[0] * cdims[1] * cdims[2];
}

DEFINE_MEMBER(void, initialize_domain)()
{
  json parameter = cfgparser->get_parameter();

  delt = parameter.value("delt", 1.0);
  delx = parameter.value("delh", 1.0);
  dely = parameter.value("delh", 1.0);
  delz = parameter.value("delh", 1.0);

  int nx = parameter.value("Nx", 1);
  int ny = parameter.value("Ny", 1);
  int nz = parameter.value("Nz", 1);

  xlim[0] = 0;
  xlim[1] = delx * nx;
  xlim[2] = xlim[1] - xlim[0];
  ylim[0] = 0;
  ylim[1] = dely * ny;
  ylim[2] = ylim[1] - ylim[0];
  zlim[0] = 0;
  zlim[1] = delz * nz;
  zlim[2] = zlim[1] - zlim[0];
}

DEFINE_MEMBER(void, initialize_workload)()
{
  balancer->fill_load(1.0);
}

DEFINE_MEMBER(void, initialize_chunks)()
{
  const int nchunk_global = cdims[3];

  // local dimensions
  int dims[3] = {ndims[0] / cdims[0], ndims[1] / cdims[1], ndims[2] / cdims[2]};

  // error check
  if (nchunk_global < nprocess) {
    ERROR << tfm::format("Number of processes should not exceed number of chunks");
    ERROR << tfm::format("* number of processes = %8d", nprocess);
    ERROR << tfm::format("* number of chunks    = %8d", nchunk_global);
    finalize();
    exit(-1);
  }

  // initial assignment
  auto boundary = balancer->assign_initial(nprocess);
  chunkmap->set_rank_boundary(boundary);

  // local chunks
  int nchunk = boundary[thisrank + 1] - boundary[thisrank];
  chunkvec.resize(nchunk);
  for (int i = 0, id = boundary[thisrank]; id < boundary[thisrank + 1]; i++, id++) {
    chunkvec[i] = create_chunk(dims, id);
  }
  chunkvec.set_neighbors(chunkmap);

  assert(validate_chunks() == true);
}

DEFINE_MEMBER(bool, validate_chunks)()
{
  bool status = chunkvec.validate(chunkmap);

  MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_CXX_BOOL, MPI_LAND, MPI_COMM_WORLD);

  return status;
}

DEFINE_MEMBER(void, save_profile)()
{
  if (thisrank == 0) {
    std::string filename = "profile.msgpack";

    // content
    json content = {{"timestamp", wclock},
                    {"nprocess", nprocess},
                    {"configuration", cfgparser->get_root()},
                    {"chunkmap", chunkmap->to_json()}};

    // serialize and output
    std::vector<std::uint8_t> buffer = json::to_msgpack(content);

    std::ofstream ofs(filename, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
    ofs.close();
  }
}

DEFINE_MEMBER(bool, rebalance)()
{
  const int nchunk_global = cdims[3];

  bool status   = false;
  json log      = {};
  json config   = cfgparser->get_application()["rebalance"];
  int  interval = 10;
  int  loglevel = 1;

  if (config.is_null() == false) {
    interval = config.value("interval", interval);
    loglevel = config.value("loglevel", loglevel);
  }

  DEBUG2 << "rebalance() start";
  float64 wclock1 = nix::wall_clock();

  if (curstep == 0 && loglevel >= 1) {
    // log initial boundary
    log["boundary"] = chunkmap->get_rank_boundary();
  } else if (curstep % interval == 0) {
    // update global load of chunks
    balancer->update_global_load(get_internal_data());

    // find new assignment
    auto boundary = chunkmap->get_rank_boundary();
    boundary      = balancer->assign(boundary);

    // sned/recv chunks
    balancer->sendrecv_chunk(*this, get_internal_data(), boundary);
    chunkmap->set_rank_boundary(boundary);
    chunkvec.set_neighbors(chunkmap);

    assert(validate_chunks() == true);

    if (loglevel >= 1) {
      log["boundary"] = boundary;
    }

    status = true;
  }

  DEBUG2 << "rebalance() end";
  float64 wclock2 = nix::wall_clock();

  log["elapsed"] = wclock2 - wclock1;
  logger->append(curstep, "rebalance", log);

  return status;
}

#undef DEFINE_MEMBER

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
