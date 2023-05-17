// -*- C++ -*-
#ifndef _APPLICATION_HPP_
#define _APPLICATION_HPP_

#include "balancer.hpp"
#include "buffer.hpp"
#include "cmdline.hpp"
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
  using ThisType    = Application<Chunk, ChunkMap>;
  using PtrBalancer = std::unique_ptr<Balancer>;
  using PtrChunk    = std::unique_ptr<Chunk>;
  using PtrChunkMap = std::unique_ptr<ChunkMap>;
  using PtrLogger   = std::unique_ptr<Logger>;
  using FloatVec    = std::vector<float64>;
  using ChunkVec    = std::vector<PtrChunk>;
  using cmdparser   = cmdline::parser;

  int         debug;    ///< debug level
  int         cl_argc;  ///< command-line argc
  char**      cl_argv;  ///< command-line argv
  std::string cfg_file; ///< configuration file name
  json        cfg_json; ///< configuration json object
  cmdparser   parser;   ///< command line parser
  float64     wclock;   ///< wall clock time at initialization
  PtrBalancer balancer; ///< load balancer
  int         numchunk; ///< number of chunkes in current process
  ChunkVec    chunkvec; ///< chunk array
  PtrChunkMap chunkmap; ///< global chunkmap
  PtrLogger   logger;   ///< logger
  FloatVec    workload; ///< global load array
  float64     tmax;     ///< maximum physical time
  float64     emax;     ///< maximum elapsed time
  std::string loadfile; ///< snapshot to be loaded
  std::string savefile; ///< snapshot to be saved
  int         ndims[4]; ///< global grid dimensions
  int         cdims[4]; ///< chunk dimensions
  int         curstep;  ///< current iteration step
  float64     curtime;  ///< current time
  float64     delt;     ///< time step
  float64     delx;     ///< grid size in x
  float64     dely;     ///< grid size in y
  float64     delz;     ///< grid size in z
  float64     cc;       ///< speed of light
  float64     xlim[3];  ///< physical domain in x
  float64     ylim[3];  ///< physical domain in y
  float64     zlim[3];  ///< physical domain in z

  // MPI related
  int  periodic[3];           ///< flag for periodic boundary
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
    int&         numchunk;
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
    return {ndims, cdims, nprocess, thisrank, numchunk, curstep, curtime, chunkmap, chunkvec};
  }

public:
  /// @brief default constructor
  Application() : mpi_init_with_nullptr(false), periodic{1, 1, 1}
  {
  }

  ///
  /// @brief constructor
  /// @param argc number of arguments
  /// @param argv array of arguments
  ///
  Application(int argc, char** argv) : mpi_init_with_nullptr(false), periodic{1, 1, 1}
  {
    cl_argc = argc;
    cl_argv = argv;
  }

  ///
  /// @brief main loop of simulation
  /// @param out output stream
  /// @return return code of application
  ///
  virtual int main(std::ostream& out);

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
  /// @brief setup command-line options
  ///
  virtual void setup_cmd();

  ///
  /// @brief parse command line options
  /// @param argc number of arguments
  /// @param argv array of arguments
  ///
  virtual void parse_cmd(int argc, char** argv);

  ///
  /// @brief parse configuration file
  ///
  virtual void parse_cfg();

  ///
  /// @brief save profile of run
  ///
  virtual void save_profile();

  ///
  /// @brief initialize MPI
  /// @param argc number of arguments
  /// @param argv array of arguments
  ///
  void initialize_mpi(int* argc, char*** argv);

  ///
  /// @brief initialize debug printing
  /// @param level debug printing level
  ///
  void initialize_debugprinting(int level);

  ///
  /// @brief finalize MPI
  /// @param cleanup return code
  ///
  void finalize_mpi(int cleanup);

  ///
  /// @brief return neighbor coordinate for a specific direction `dir`
  /// @param coord index of coordinate
  /// @param delta difference of index of coordinate from `coord`
  /// @param dir direction of coordinate
  /// @return `coord + delta` if not at boundary, otherwise boundary condition dependent
  ///
  int get_nb_coord(int coord, int delta, int dir);

  ///
  /// @brief set neighbors of chunks
  ///
  virtual void set_chunk_neighbors();

  ///
  /// @brief initialize application
  /// @param argc number of arguments
  /// @param argv array of arguments
  ///
  virtual void initialize(int argc, char** argv);

  ///
  /// @brief accumulate work load of local chunks
  ///
  virtual void accumulate_workload();

  ///
  /// @brief get work load of all chunks
  ///
  virtual void get_global_workload();

  ///
  /// @brief initialize work load array
  ///
  virtual void initialize_workload();

  ///
  /// @brief initialize chunkmap object
  ///
  virtual void initialize_chunkmap();

  ///
  /// @brief performing load balancing
  /// @return return true if rebalancing is performed and false otherwise
  ///
  virtual bool rebalance();

  ///
  /// @brief check the validity of chunkmap object
  /// @return true if the chunkmap is valid and false otherwise
  ///
  virtual bool validate_chunkmap();

  ///
  /// @brief check if the number of chunks per rank does not exceed MAX_CHUNK_PER_RANK
  /// @return true if it is okay and false otherwise
  ///
  virtual bool validate_numchunk();

  ///
  /// @brief finalize application
  /// @param cleanup return code
  ///
  virtual void finalize(int cleanup = 0);

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
    // load snapshot
    this->load_snapshot();
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
    if (curtime < tmax + delt) {
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

    return emax - etime;
  }

  ///
  /// @brief logging
  ///
  virtual void logging()
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

DEFINE_MEMBER(int, main)(std::ostream& out)
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
    // logging
    //
    logging();
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
  {
    int cleanup = debug == 0 ? 0 : -1;
    finalize(cleanup);
  }

  return 0;
}

DEFINE_MEMBER(void, setup_cmd)()
{
  const float64     etmax = 60 * 60;
  const float64     ptmax = std::numeric_limits<float64>::max();
  const std::string fn    = "default.json";

  parser.add<std::string>("config", 'c', "configuration file", false, fn);
  parser.add<std::string>("load", 'l', "load file for restart", false, "");
  parser.add<std::string>("save", 's', "save file for restart", false, "");
  parser.add<float64>("tmax", 't', "maximum physical time", false, ptmax);
  parser.add<float64>("emax", 'e', "maximum elased time [sec]", false, etmax);
  parser.add<int>("debug", 'd', "debug level", false, 0);
}

DEFINE_MEMBER(void, parse_cmd)(int argc, char** argv)
{
  // setup command-line parser
  setup_cmd();

  // parse
  parser.parse_check(argc, argv);

  cfg_file = parser.get<std::string>("config");
  loadfile = parser.get<std::string>("load");
  savefile = parser.get<std::string>("save");
  tmax     = parser.get<float64>("tmax");
  emax     = parser.get<float64>("emax");
  debug    = parser.get<int>("debug");
}

DEFINE_MEMBER(void, parse_cfg)()
{
  // read configuration file
  {
    std::ifstream f(cfg_file.c_str());
    cfg_json = json::parse(f, nullptr, true, true);
  }

  // check sections: application, diagnostic, parameter
  {
    bool status = true;

    if (cfg_json["application"].is_null()) {
      ERROR << tfm::format("Configuration file misses `application` section");
      status = false;
    }

    if (cfg_json["diagnostic"].is_null()) {
      ERROR << tfm::format("Configuration file misses `diagnostic` section");
      status = false;
    }

    if (cfg_json["parameter"].is_null()) {
      ERROR << tfm::format("Configuration file misses `parameter` section");
      status = false;
    }

    if (status == false) {
      finalize(-1);
      exit(-1);
    }
  }

  // time step and grid size
  json    parameter = cfg_json["parameter"];
  float64 delh      = parameter.value("delh", 1.0);

  delt = parameter.value("delt", 1.0);
  delx = delh;
  dely = delh;
  delz = delh;

  // get dimensions
  int nx = parameter.value("Nx", 1);
  int ny = parameter.value("Ny", 1);
  int nz = parameter.value("Nz", 1);
  int cx = parameter.value("Cx", 1);
  int cy = parameter.value("Cy", 1);
  int cz = parameter.value("Cz", 1);

  // check dimensions
  if (!(nz % cz == 0 && ny % cy == 0 && nx % cx == 0)) {
    ERROR << tfm::format("Number of grid must be divisible by number of chunk");
    ERROR << tfm::format("Nx, Ny, Nz = [%4d, %4d, %4d]", nx, ny, nz);
    ERROR << tfm::format("Cx, Cy, Cz = [%4d, %4d, %4d]", cx, cy, cz);
    finalize(-1);
    exit(-1);
  }

  ndims[0] = nz;
  ndims[1] = ny;
  ndims[2] = nx;
  ndims[3] = ndims[0] * ndims[1] * ndims[2];
  cdims[0] = cz;
  cdims[1] = cy;
  cdims[2] = cx;
  cdims[3] = cdims[0] * cdims[1] * cdims[2];

  // set global domain size
  xlim[0] = 0;
  xlim[1] = delx * ndims[2];
  xlim[2] = xlim[1] - xlim[0];
  ylim[0] = 0;
  ylim[1] = dely * ndims[1];
  ylim[2] = ylim[1] - ylim[0];
  zlim[0] = 0;
  zlim[1] = delz * ndims[0];
  zlim[2] = zlim[1] - zlim[0];
}

DEFINE_MEMBER(void, save_profile)()
{
  if (thisrank == 0) {
    std::string filename = "profile.msgpack";

    // timestamp
    json timestamp_json = wclock;

    // chunkmap
    json cmap_json;
    chunkmap->save_json(cmap_json);

    // content
    json content = {{"timestamp", timestamp_json},
                    {"nprocess", nprocess},
                    {"configuration", cfg_json},
                    {"chunkmap", cmap_json}};

    // serialize and output
    std::vector<std::uint8_t> buffer = json::to_msgpack(content);

    std::ofstream ofs(filename, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
    ofs.close();
  }
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
  mpistream::initialize(argv[0][0]);
}

DEFINE_MEMBER(void, initialize_debugprinting)(int level)
{
  DebugPrinter::init();
  DebugPrinter::set_level(level);
}

DEFINE_MEMBER(void, finalize_mpi)(int cleanup)
{
  // release stdout/stderr
  mpistream::finalize(cleanup);

  MPI_Finalize();
}

DEFINE_MEMBER(int, get_nb_coord)(int coord, int delta, int dir)
{
  int cdir = coord + delta;

  if (periodic[dir] == 1) {
    cdir = cdir >= 0 ? cdir : cdims[dir] - 1;
    cdir = cdir < cdims[dir] ? cdir : 0;
  } else {
    cdir = (cdir >= 0 && cdir < cdims[dir]) ? cdir : MPI_PROC_NULL;
  }

  return cdir;
}

DEFINE_MEMBER(void, set_chunk_neighbors)()
{
  for (int i = 0; i < numchunk; i++) {
    int ix, iy, iz;
    int id = chunkvec[i]->get_id();
    chunkmap->get_coordinate(id, iz, iy, ix);

    for (int dirz = -1; dirz <= +1; dirz++) {
      for (int diry = -1; diry <= +1; diry++) {
        for (int dirx = -1; dirx <= +1; dirx++) {
          // neighbor coordiante
          int cz = get_nb_coord(iz, dirz, 0);
          int cy = get_nb_coord(iy, diry, 1);
          int cx = get_nb_coord(ix, dirx, 2);

          // set neighbor id
          int nbid = chunkmap->get_chunkid(cz, cy, cx);
          chunkvec[i]->set_nb_id(dirz, diry, dirx, nbid);

          // set neighbor rank
          int nbrank = chunkmap->get_rank(nbid);
          chunkvec[i]->set_nb_rank(dirz, diry, dirx, nbrank);
        }
      }
    }
  }
}

DEFINE_MEMBER(void, initialize)(int argc, char** argv)
{
  initialize_mpi(&argc, &argv);

  // parse command line arguments
  parse_cmd(argc, argv);

  // parse configuration file
  parse_cfg();

  // initialize current physical time and time step
  curstep  = 0;
  curtime  = 0.0;
  balancer = std::make_unique<Balancer>();
  logger   = std::make_unique<Logger>(cfg_json["application"]["log"]);

  initialize_debugprinting(debug);
  initialize_chunkmap();
}

DEFINE_MEMBER(void, accumulate_workload)()
{
  const int Nc = cdims[3];

  // local workload
  for (int i = 0; i < numchunk; i++) {
    int id = chunkvec[i]->get_id();

    workload[id] += chunkvec[i]->get_total_load();
  }
}

DEFINE_MEMBER(void, get_global_workload)()
{
  std::vector<int> rcnt(nprocess);
  std::vector<int> disp(nprocess);

  // recv count
  std::fill(rcnt.begin(), rcnt.end(), 0);
  for (int i = 0; i < workload.size(); i++) {
    int rank = chunkmap->get_rank(i);
    rcnt[rank]++;
  }

  // displacement
  disp[0] = 0;
  for (int r = 0; r < nprocess - 1; r++) {
    disp[r + 1] = disp[r] + rcnt[r];
  }

  MPI_Allgatherv(MPI_IN_PLACE, rcnt[thisrank], MPI_FLOAT64_T, workload.data(), rcnt.data(),
                 disp.data(), MPI_FLOAT64_T, MPI_COMM_WORLD);
}

DEFINE_MEMBER(void, initialize_workload)()
{
  std::fill(workload.begin(), workload.end(), 1.0);
}

DEFINE_MEMBER(void, initialize_chunkmap)()
{
  const int Nc      = cdims[3];
  int       dims[3] = {ndims[0] / cdims[0], ndims[1] / cdims[1], ndims[2] / cdims[2]};

  std::vector<int> boundary(nprocess + 1);

  // error check
  if (Nc < nprocess) {
    ERROR << tfm::format("Number of processes should not exceed number of chunks");
    ERROR << tfm::format("* number of processes = %8d", nprocess);
    ERROR << tfm::format("* number of chunks    = %8d", Nc);
    finalize(-1);
    exit(-1);
  }

  // create chunkmap
  chunkmap = std::make_unique<ChunkMap>(cdims);

  // allocate workload and initialize
  workload.resize(Nc);
  initialize_workload();

  // initial assignment
  balancer->assign(workload, boundary, true);

  // set rank in chunkmap using boundary
  for (int i = 0, rank = 0; i < Nc; i++) {
    if (i < boundary[rank + 1]) {
      chunkmap->set_rank(i, rank);
    } else if (i == boundary[rank + 1]) {
      rank++;
      chunkmap->set_rank(i, rank);
    } else {
      json error = boundary;
      ERROR << tfm::format("Inconsistent boundary array detected");
      ERROR << error.dump(2);
      assert(false);
    }
  }

  // create local chunkvec
  numchunk = boundary[thisrank + 1] - boundary[thisrank];
  chunkvec.resize(numchunk);
  for (int i = 0, id = boundary[thisrank]; id < boundary[thisrank + 1]; i++, id++) {
    chunkvec[i] = create_chunk(dims, id);
  }

  // set neighbor
  set_chunk_neighbors();

  // reset workload
  std::fill(workload.begin(), workload.end(), 0.0);

  // check chunkmap
  assert(validate_numchunk() == true);
  assert(validate_chunkmap() == true);
}

DEFINE_MEMBER(bool, rebalance)()
{
  const int        Nc = cdims[3];
  std::vector<int> boundary(nprocess + 1);
  std::vector<int> newrank(Nc);

  bool status   = false;
  json config   = cfg_json["application"]["rebalance"];
  json log      = {};
  int  interval = config.value("interval", 10);
  int  loglevel = config.value("loglevel", 1);

  DEBUG2 << "rebalance() start";
  float64 wclock1 = nix::wall_clock();

  // accumulate workload
  accumulate_workload();

  if (curstep == 0) {
    // calculate boundary from rank in chunkamp
    for (int i = 0, rank = 0; i < Nc; i++) {
      if (rank == chunkmap->get_rank(i))
        continue;
      // found a boundary
      boundary[rank + 1] = i;
      rank++;
    }
    boundary[0]        = 0;
    boundary[nprocess] = Nc;

    // log
    if (loglevel >= 1) {
      log["boundary"] = boundary;
    }

    if (loglevel >= 2) {
      log["workload"] = workload;
    }
  } else if (curstep % interval == 0) {
    // calculate workload
    get_global_workload();

    // calculate boundary from rank in chunkamp
    for (int i = 0, rank = 0; i < Nc; i++) {
      if (rank == chunkmap->get_rank(i))
        continue;
      // found a boundary
      boundary[rank + 1] = i;
      rank++;
    }
    boundary[0]        = 0;
    boundary[nprocess] = Nc;

    //
    // rebalance
    //
    balancer->assign(workload, boundary);
    balancer->get_rank(boundary, newrank);
    balancer->sendrecv_chunk(*this, get_internal_data(), newrank);

    // reset rank in chunkmap using boundary
    for (int i = 0, rank = 0; i < Nc; i++) {
      if (i < boundary[rank + 1]) {
        chunkmap->set_rank(i, rank);
      } else if (i == boundary[rank + 1]) {
        rank++;
        chunkmap->set_rank(i, rank);
      } else {
        ERROR << tfm::format("Inconsistent boundary array detected");
        ERROR << tfm::format("* boundary[%08d] = %08d", rank + 1, boundary[rank + 1]);
        assert(false);
      }
    }

    // reset neighbor
    set_chunk_neighbors();

    // reset workload
    std::fill(workload.begin(), workload.end(), 0.0);

    // check number of chunks
    assert(validate_numchunk() == true);

    // log
    if (loglevel >= 1) {
      log["boundary"] = boundary;
    }

    if (loglevel >= 2) {
      log["workload"] = workload;
    }

    status = true;
  }

  DEBUG2 << "rebalance() end";
  float64 wclock2 = nix::wall_clock();

  log["elapsed"] = wclock2 - wclock1;
  logger->append(curstep, "rebalance", log);

  return status;
}

DEFINE_MEMBER(bool, validate_chunkmap)()
{
  bool status = true;

  // for each chunk
  for (int i = 0; i < numchunk; i++) {
    int ix, iy, iz;
    chunkmap->get_coordinate(chunkvec[i]->get_id(), iz, iy, ix);

    // check neighbor ID and rank
    for (int dirz = -1; dirz <= +1; dirz++) {
      for (int diry = -1; diry <= +1; diry++) {
        for (int dirx = -1; dirx <= +1; dirx++) {
          int cz     = get_nb_coord(iz, dirz, 0);
          int cy     = get_nb_coord(iy, diry, 1);
          int cx     = get_nb_coord(ix, dirx, 2);
          int nbid   = chunkvec[i]->get_nb_id(dirz, diry, dirx);
          int nbrank = chunkvec[i]->get_nb_rank(dirz, diry, dirx);
          int id     = chunkmap->get_chunkid(cz, cy, cx);
          int rank   = chunkmap->get_rank(id);

          status = status & (id == nbid);
          status = status & (rank == nbrank);
        }
      }
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_CXX_BOOL, MPI_LAND, MPI_COMM_WORLD);
  return status;
}

DEFINE_MEMBER(bool, validate_numchunk)()
{
  if (numchunk > MAX_CHUNK_PER_RANK) {
    ERROR << tfm::format("Number of chunk per rank should not exceed %8d", MAX_CHUNK_PER_RANK);
    return false;
  }

  return true;
}

DEFINE_MEMBER(void, finalize)(int cleanup)
{
  // save snapshot
  this->save_snapshot();

  // save log
  logger->save(curstep, true);

  // MPI
  finalize_mpi(cleanup);
}

#undef DEFINE_MEMBER

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
