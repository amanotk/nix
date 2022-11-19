// -*- C++ -*-
#ifndef _APPLICATION_HPP_
#define _APPLICATION_HPP_

#include "balancer.hpp"
#include "buffer.hpp"
#include "cmdline.hpp"
#include "jsonio.hpp"
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
template <class Chunk, class ChunkMap>
class Application
{
protected:
  using cmdparser   = cmdline::parser;
  using PtrBalancer = std::unique_ptr<Balancer>;
  using PtrChunk    = std::unique_ptr<Chunk>;
  using PtrChunkMap = std::unique_ptr<ChunkMap>;
  using FloatVec    = std::vector<float64>;
  using ChunkVec    = std::vector<PtrChunk>;

  int         cleanup;  ///< cleanup flag
  int         cl_argc;  ///< command-line argc
  char**      cl_argv;  ///< command-line argv
  std::string cfg_file; ///< configuration file name
  json        cfg_json; ///< configuration json object
  json        log_json; ///< log json object
  cmdparser   parser;   ///< command line parser
  float64     wclock;   ///< wall clock time at initialization
  PtrBalancer balancer; ///< load balancer
  int         numchunk; ///< number of chunkes in current process
  ChunkVec    chunkvec; ///< chunk array
  PtrChunkMap chunkmap; ///< global chunkmap
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
  int    periodic[3];           ///< flag for periodic boundary
  int    nprocess;              ///< number of mpi processes
  int    thisrank;              ///< my rank
  Buffer sendbuf;               ///< send buffer
  Buffer recvbuf;               ///< recv buffer
  bool   mpi_init_with_nullptr; ///< for testing purpose

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
  /// @brief initialize MPI
  /// @param argc number of arguments
  /// @param argv array of arguments
  ///
  void initialize_mpi(int* argc, char*** argv);

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
  int get_nb_coord(const int coord, const int delta, const int dir);

  ///
  /// @brief initialize application
  /// @param argc number of arguments
  /// @param argv array of arguments
  ///
  virtual void initialize(int argc, char** argv);

  ///
  /// @brief load a snapshot file for restart
  ///
  virtual void load();

  ///
  /// @brief save current state to a snapshot file (for restart)
  ///
  virtual void save();

  ///
  /// @brief setup initial conditions
  ///
  virtual void setup();

  ///
  /// @brief output log
  ///
  virtual void log();

  ///
  /// @brief perform various diagnostics output
  /// @param out output stream to which console message will be printed (if any)
  ///
  virtual void diagnostic(std::ostream& out);

  ///
  /// @brief advance physical quantities by one step
  ///
  virtual void push();

  ///
  /// @brief check if further push is needed or not
  /// @return true if the maximum physical time is not yet reached and false otherwise
  ///
  virtual bool is_push_needed();

  ///
  /// @brief get available elapsed time
  /// @return available elapsed time in second
  ///
  virtual float64 get_available_etime();

  ///
  /// @brief increment step and physical time
  ///
  virtual void increment_time();

  ///
  /// @brief append given json to log json object at current step
  ///
  virtual void append_log_step(json& obj);

  ///
  /// @brief save log json object to file
  /// @param flush force flush if true
  ///
  virtual void save_log(bool flush = false);

  ///
  /// @brief accumulate work load of local chunks
  ///
  virtual void accumulate_workload();

  ///
  /// @brief get work load of all chunks
  ///
  virtual void get_global_workload();

  ///
  /// @brief factory to create chunk object
  /// @param dims local number of grids in each direction
  /// @param id chunk ID
  /// @return chunk object
  ///
  virtual std::unique_ptr<Chunk> create_chunk(const int dims[], const int id);

  ///
  /// @brief initialize chunkmap object
  ///
  virtual void initialize_chunkmap();

  ///
  /// @brief rebuild chunkmap object by performing load balancing
  /// @return return true if rebuild performed and false otherwise
  ///
  virtual bool rebuild_chunkmap();

  ///
  /// @brief check the validity of chunkmap object
  /// @return true if the chunkmap is valid and false otherwise
  ///
  virtual bool validate_chunkmap();

  ///
  /// @brief perform MPI send/recv of chunks for load balancing
  /// @param newrank array of ranks to which chunks are assigned
  ///
  virtual void sendrecv_chunk(std::vector<int>& newrank);

  ///
  /// @brief set neighbors of chunks
  ///
  virtual void set_chunk_neighbors();

  ///
  /// @brief write all chunk data retrieved with given `mode`
  /// @param fh MPI file handle to which the data will be written
  /// @param disp displacement of the file from its beginning
  /// @param mode mode of data to be retrieved from chunks
  ///
  virtual void write_chunk_all(MPI_File& fh, size_t& disp, const int mode);

  ///
  /// @brief wait boundary exchange operation in `queue` and perform unpacking
  /// @param queue list of chunk IDs performing boundary exchange
  /// @param mode mode of boundary exchange
  ///
  virtual void wait_bc_exchange(std::set<int>& queue, const int mode);

  ///
  /// @brief finalize application
  /// @param cleanup return code
  ///
  virtual void finalize(int cleanup = 0);

public:
  /// @brief default constructor
  Application();

  ///
  /// @brief constructor
  /// @param argc number of arguments
  /// @param argv array of arguments
  ///
  Application(int argc, char** argv);

  ///
  /// @brief main loop of simulation
  /// @param out output stream
  /// @return return code of application
  ///
  virtual int main(std::ostream& out);
};

//
// implementation follows
//

#define DEFINE_MEMBER(type, name)                                                                  \
  template <class Chunk, class ChunkMap>                                                           \
  type Application<Chunk, ChunkMap>::name

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
}

DEFINE_MEMBER(void, parse_cmd)(int argc, char** argv)
{
  // setup command-line parser
  setup_cmd();

  // parse
  parser.parse_check(argc, argv);

  tmax     = parser.get<float64>("tmax");
  emax     = parser.get<float64>("emax");
  loadfile = parser.get<std::string>("load");
  savefile = parser.get<std::string>("save");
  cfg_file = parser.get<std::string>("config");
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
      tfm::format(std::cerr, "Error: configuration file misses `application` section\n");
      status = false;
    }

    if (cfg_json["diagnostic"].is_null()) {
      tfm::format(std::cerr, "Error: configuration file misses `diagnostic` section\n");
      status = false;
    }

    if (cfg_json["parameter"].is_null()) {
      tfm::format(std::cerr, "Error: configuration file misses `parameter` section\n");
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
    ERRORPRINT("Number of grid must be divisible by number of chunk\n"
               "Nx, Ny, Nz = [%4d, %4d, %4d]\n"
               "Cx, Cy, Cz = [%4d, %4d, %4d]\n",
               nx, ny, nz, cx, cy, cz);
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

DEFINE_MEMBER(void, initialize_mpi)(int* argc, char*** argv)
{
  // initialize MPI
  if (mpi_init_with_nullptr == true) {
    MPI_Init(nullptr, nullptr);
  } else {
    MPI_Init(argc, argv);
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

DEFINE_MEMBER(void, finalize_mpi)(int cleanup)
{
  // release stdout/stderr
  mpistream::finalize(cleanup);

  MPI_Finalize();
}

DEFINE_MEMBER(int, get_nb_coord)(const int coord, const int delta, const int dir)
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

DEFINE_MEMBER(void, initialize)(int argc, char** argv)
{
  // parse command line arguments
  parse_cmd(argc, argv);

  // parse configuration file
  parse_cfg();

  // initialize current physical time and time step
  curstep = 0;
  curtime = 0.0;

  // MPI
  initialize_mpi(&argc, &argv);

  // chunkmap
  initialize_chunkmap();

  // load balancer
  balancer = std::make_unique<Balancer>();

  // buffer; 16 kB by default
  int bufsize = 1024 * 16;
  sendbuf.resize(bufsize);
  recvbuf.resize(bufsize);
}

DEFINE_MEMBER(void, load)()
{
  if (!loadfile.empty()) {
    LOGPRINT1(std::cout, "Load snapshot from %s\n", loadfile.c_str());
  }
  LOGPRINT1(std::cout, "No load file specified\n");
}

DEFINE_MEMBER(void, save)()
{
  if (!savefile.empty()) {
    LOGPRINT1(std::cout, "Save snapshot to %s\n", savefile.c_str());
  }
  LOGPRINT1(std::cout, "No save file specified\n");
}

DEFINE_MEMBER(void, setup)()
{
  // load snapshot
  this->load();
}

DEFINE_MEMBER(void, log)()
{
  json obj = cfg_json["application"]["log"];

  // get parameters from json
  int         loglevel = obj.value("level", 0);
  float64     interval = obj.value("interval", 1.0);
  std::string prefix   = obj.value("prefix", "log");
  std::string path     = obj.value("path", ".") + "/";
  std::string numstep  = tfm::format("%06d", curstep);

  // filename
  std::string fn_data = prefix + ".data";

  MPI_File fh;
  size_t   disp;
  json     dataset;
  json     log_step = {{"timestamp", wall_clock()}};

  //
  // initial setup
  //
  if (curstep == 0) {
    // create data file
    jsonio::open_file((path + fn_data).c_str(), &fh, &disp, "w");
    jsonio::close_file(&fh);

    // create json file
    log_json["meta"]    = {{"endian", nix::get_endian_flag()}, {"rawfile", fn_data}, {"order", 1}};
    log_json["dataset"] = {};
    log_json["log"]     = {};
    log_json["flushed"] = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
  }

  //
  // chunk load
  //
  if (loglevel >= 2) {
    const std::string name    = "load_" + numstep;
    const char        desc[]  = "chunk load";
    const int         ndim    = 2;
    const int         dims[2] = {cdims[3], Chunk::NumLoadMode};
    const int         size    = dims[0] * dims[1] * sizeof(float64);

    jsonio::open_file((path + fn_data).c_str(), &fh, &disp, "a");

    jsonio::put_metadata(dataset, name.c_str(), "f8", desc, disp, size, ndim, dims);
    this->write_chunk_all(fh, disp, Chunk::DiagnosticLoad);

    jsonio::close_file(&fh);
  }

  // update
  log_json["dataset"].push_back(dataset);
  append_log_step(log_step);

  // save log
  save_log();
}

DEFINE_MEMBER(void, diagnostic)(std::ostream& out)
{
  out << tfm::format("*** step = %8d (time = %10.5f)\n", curstep, curtime);
}

DEFINE_MEMBER(void, push)()
{
  curtime += delt;
  curstep++;
}

DEFINE_MEMBER(bool, is_push_needed)()
{
  if (curtime < tmax + delt) {
    return true;
  }
  return false;
}

DEFINE_MEMBER(float64, get_available_etime)()
{
  float64 etime;

  if (thisrank == 0) {
    etime = wall_clock() - wclock;
  }
  MPI_Bcast(&etime, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return emax - etime;
}

DEFINE_MEMBER(void, increment_time)()
{
  curtime += delt;
  curstep++;
}

DEFINE_MEMBER(void, append_log_step)(json& obj)
{
  std::string numstep = tfm::format("%06d", curstep);

  if (log_json["log"].contains(numstep) == false) {
    log_json["log"][numstep] = {};
  }

  for (auto it = obj.begin(); it != obj.end(); ++it) {
    log_json["log"][numstep][it.key()] = it.value();
  }
}

DEFINE_MEMBER(void, save_log)(bool force)
{
  json obj = cfg_json["application"]["log"];

  // get parameters from json
  float64     interval = obj.value("interval", 1.0);
  std::string prefix   = obj.value("prefix", "log");
  std::string path     = obj.value("path", ".") + "/";
  std::string filename = prefix + ".json";

  float64 wclock              = wall_clock();
  float64 etime_since_flushed = wclock - log_json.value("flushed", 0.0);

  if (thisrank == 0 && (force == true || etime_since_flushed > interval)) {
    std::ofstream ofs(filename);
    log_json["flushed"] = wclock;

    ofs << std::setw(2) << log_json << std::flush;
    ofs.close();
  }
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

DEFINE_MEMBER(std::unique_ptr<Chunk>, create_chunk)(const int dims[], const int id)
{
  return std::make_unique<Chunk>(dims, id);
}

DEFINE_MEMBER(void, initialize_chunkmap)()
{
  const int Nc = cdims[3];

  // error check
  if (Nc < nprocess) {
    ERRORPRINT("Number of processes exceeds number of chunks\n"
               "* number of processes = %8d\n"
               "* number of chunks    = %8d\n",
               nprocess, Nc);
    finalize(-1);
    exit(-1);
  }

  //
  // initialize chunkmap and chunkvec
  //
  {
    int dims[3] = {ndims[0] / cdims[0], ndims[1] / cdims[1], ndims[2] / cdims[2]};
    int idzero  = 0;

    chunkmap = std::make_unique<ChunkMap>(cdims);

    for (int rank = 0; rank < nprocess; rank++) {
      int mc = (Nc + rank) / nprocess;

      // initialize global chunkmap
      for (int id = idzero; id < idzero + mc; id++) {
        chunkmap->set_rank(id, rank);
      }

      if (rank == thisrank) {
        // initialize local chunkvec
        numchunk = mc;
        chunkvec.resize(numchunk);
        for (int i = 0; i < numchunk; i++) {
          chunkvec[i] = create_chunk(dims, idzero + i);
        }
      }

      idzero += mc;
    }

    set_chunk_neighbors();
  }

  //
  // allocate workload and initialize
  //
  workload.resize(Nc);
  std::fill(workload.begin(), workload.end(), 0.0);
}

DEFINE_MEMBER(bool, rebuild_chunkmap)()
{
  const int        Nc = cdims[3];
  std::vector<int> boundary(nprocess + 1);
  std::vector<int> oldrank(Nc);
  std::vector<int> newrank(Nc);

  json    rebuild;
  int     interval = cfg_json["application"]["rebuild"].value("interval", 10);
  int     loglevel = cfg_json["application"]["log"].value("level", 0);
  float64 wclock1  = 0;
  float64 wclock2  = 0;

  // accumulate workload
  accumulate_workload();

  if (curstep != 0 && curstep % interval != 0)
    return false;

  //
  // now rebuild chunkmap
  //

  // *** rebuild start ***
  wclock1 = wall_clock();

  get_global_workload();

  // calculate new chunk distribution
  for (int i = 0; i < Nc; i++) {
    oldrank[i] = chunkmap->get_rank(i);
  }
  balancer->get_boundary(oldrank, boundary);
  balancer->partition(nprocess, workload, boundary);
  balancer->get_rank(boundary, newrank);

  // send/recv chunk
  sendrecv_chunk(newrank);

  // reset rank
  for (int id = 0; id < Nc; id++) {
    chunkmap->set_rank(id, newrank[id]);
  }
  set_chunk_neighbors();

  // *** rebuild end ***
  wclock2 = wall_clock();

  //
  // log
  //
  {
    if (loglevel >= 0) {
      rebuild["start"]   = wclock1;
      rebuild["end"]     = wclock2;
      rebuild["elapsed"] = wclock2 - wclock1;
    }

    if (loglevel >= 1) {
      rebuild["boundary"] = boundary;
      rebuild["rankload"] = balancer->get_rankload(boundary, workload);
    }

    if (loglevel >= 2) {
      int  nexchange = 0;
      json chunkid;
      json old_rank;
      json new_rank;

      for (int i = 0; i < Nc; i++) {
        if (newrank[i] != oldrank[i]) {
          nexchange++;
          chunkid.push_back(i);
          old_rank.push_back(oldrank[i]);
          new_rank.push_back(newrank[i]);
        }
      }

      rebuild["exchange"]             = {};
      rebuild["exchange"]["number"]   = nexchange;
      rebuild["exchange"]["chunkid"]  = chunkid;
      rebuild["exchange"]["old_rank"] = old_rank;
      rebuild["exchange"]["new_rank"] = new_rank;
    }

    {
      json log_step = {{"rebuild", rebuild}};
      append_log_step(log_step);
    }
  }

  // reset
  std::fill(workload.begin(), workload.end(), 0.0);

  return true;
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

DEFINE_MEMBER(void, sendrecv_chunk)(std::vector<int>& newrank)
{
  const int dims[3] = {ndims[0] / cdims[0], ndims[1] / cdims[1], ndims[2] / cdims[2]};
  const int Ncmax   = Chunk::get_max_id();
  const int Nc      = cdims[0] * cdims[1] * cdims[2];
  const int rankmin = 0;
  const int rankmax = nprocess - 1;

  int rank_l = thisrank > rankmin ? thisrank - 1 : MPI_PROC_NULL;
  int rank_r = thisrank < rankmax ? thisrank + 1 : MPI_PROC_NULL;

  //
  // check buffer size and reallocate if necessary
  //
  {
    int sendsize   = 0;
    int sendsize_l = 0;
    int sendsize_r = 0;
    int recvsize   = 0;
    int recvsize_l = 0;
    int recvsize_r = 0;

    for (int i = 0; i < numchunk; i++) {
      int id = chunkvec[i]->get_id();

      if (newrank[id] == thisrank - 1) {
        sendsize_l = std::max(sendsize_l, chunkvec[i]->pack(nullptr, 0));
      }
      if (newrank[id] == thisrank + 1) {
        sendsize_r = std::max(sendsize_r, chunkvec[i]->pack(nullptr, 0));
      }
    }

    // get maximum possible chunk size
    {
      MPI_Request request[4];

      MPI_Isend(&sendsize_l, sizeof(int), MPI_BYTE, rank_l, 1, MPI_COMM_WORLD, &request[0]);
      MPI_Isend(&sendsize_r, sizeof(int), MPI_BYTE, rank_r, 2, MPI_COMM_WORLD, &request[1]);
      MPI_Irecv(&recvsize_l, sizeof(int), MPI_BYTE, rank_l, 2, MPI_COMM_WORLD, &request[2]);
      MPI_Irecv(&recvsize_r, sizeof(int), MPI_BYTE, rank_r, 1, MPI_COMM_WORLD, &request[3]);

      MPI_Waitall(4, request, MPI_STATUSES_IGNORE);

      sendsize = std::max(sendsize_l, sendsize_r);
      recvsize = std::max(recvsize_l, recvsize_r);
    }

    // resize send buffer
    if (sendsize > sendbuf.size) {
      sendbuf.resize(sendsize);
    }

    // resize recv buffer
    if (recvsize > recvbuf.size) {
      recvbuf.resize(recvsize);
    }
  }

  //
  // function for sending chunk
  //
  auto send_chunk = [&](int chunkid, int rank, int tag, int dir, int pos) {
    MPI_Request request;
    int         size;
    uint8_t*    buf = sendbuf.get(pos);

    if (chunkid < 0 || chunkid >= newrank.size()) {
      return;
    }

    auto it    = std::find_if(chunkvec.begin(), chunkvec.end(),
                           [&](PtrChunk& p) { return p->get_id() == chunkid; });
    int  index = std::distance(chunkvec.begin(), it);

    while (newrank[chunkid] == rank) {
      // pack
      size = chunkvec[index]->pack(buf, 0);
      chunkvec[index]->set_id(Ncmax); // to be removed

      MPI_Isend(buf, size, MPI_BYTE, rank, tag, MPI_COMM_WORLD, &request);
      MPI_Wait(&request, MPI_STATUS_IGNORE);

      chunkid += dir;
      index += dir;
    }
  };

  //
  // function for receiving chunk
  //
  auto recv_chunk = [&](int chunkid, int rank, int tag, int dir, int pos) {
    MPI_Request request;
    int         size = recvbuf.size - pos;
    uint8_t*    buf  = recvbuf.get(pos);

    if (chunkid < 0 || chunkid >= newrank.size()) {
      return;
    }

    while (newrank[chunkid] == thisrank) {
      MPI_Irecv(buf, size, MPI_BYTE, rank, tag, MPI_COMM_WORLD, &request);
      MPI_Wait(&request, MPI_STATUS_IGNORE);

      // unpack
      PtrChunk p = create_chunk(dims, 0);
      p->unpack(buf, 0);
      chunkvec.push_back(std::move(p));

      chunkid += dir;
    }
  };

  //
  // chunk exchange at odd boundary
  //
  if (thisrank % 2 == 1) {
    // send to left
    {
      int chunkid = chunkvec[0]->get_id();
      send_chunk(chunkid, rank_l, 1, +1, 0);
    }
    // recv from left
    {
      int chunkid = chunkvec[0]->get_id() - 1;
      recv_chunk(chunkid, rank_l, 2, -1, 0);
    }
  } else {
    // send to right
    {
      int chunkid = chunkvec[numchunk - 1]->get_id();
      send_chunk(chunkid, rank_r, 2, -1, 0);
    }
    // recv from right
    {
      int chunkid = chunkvec[numchunk - 1]->get_id() + 1;
      recv_chunk(chunkid, rank_r, 1, +1, 0);
    }
  }

  //
  // chunk exchange at even boundary
  //
  if (thisrank % 2 == 1) {
    // send to right
    {
      int chunkid = chunkvec[numchunk - 1]->get_id();
      send_chunk(chunkid, rank_r, 3, -1, 0);
    }
    // recv from right
    {
      int chunkid = chunkvec[numchunk - 1]->get_id() + 1;
      recv_chunk(chunkid, rank_r, 4, +1, 0);
    }
  } else {
    // send to left
    {
      int chunkid = chunkvec[0]->get_id();
      send_chunk(chunkid, rank_l, 4, +1, 0);
    }
    // recv from left
    {
      int chunkid = chunkvec[0]->get_id() - 1;
      recv_chunk(chunkid, rank_l, 3, -1, 0);
    }
  }

  //
  // sort chunkvec and remove unused chunkes
  //
  {
    std::sort(chunkvec.begin(), chunkvec.end(),
              [](const PtrChunk& x, const PtrChunk& y) { return x->get_id() < y->get_id(); });

    // reset numchunk
    numchunk = 0;
    for (int i = 0; i < chunkvec.size(); i++) {
      if (chunkvec[i]->get_id() == Ncmax)
        break;
      numchunk++;
    }

    // resize and discard unused chunks
    chunkvec.resize(numchunk);
  }
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

DEFINE_MEMBER(void, write_chunk_all)(MPI_File& fh, size_t& disp, const int mode)
{
  int allsize = 0;
  int maxsize = 0;
  int bufaddr = 0;
  int bufsize[numchunk];

  // calculate local address
  for (int i = 0; i < numchunk; i++) {
    bufsize[i] = chunkvec[i]->pack_diagnostic(mode, nullptr, 0);
    allsize += bufsize[i];
  }
  MPI_Exscan(&allsize, &bufaddr, 1, MPI_INT32_T, MPI_SUM, MPI_COMM_WORLD);

  // resize buffer if needed
  maxsize = *std::max_element(bufsize, bufsize + numchunk);
  if (sendbuf.size < maxsize) {
    sendbuf.resize(maxsize);
    recvbuf.resize(maxsize);
  }

  // write for each chunk
  {
    MPI_Request req;
    size_t      chunkdisp = disp + bufaddr;
    uint8_t*    sendptr   = sendbuf.get();

    for (int i = 0; i < numchunk; i++) {
      // pack
      assert(bufsize[i] == chunkvec[i]->pack_diagnostic(mode, sendptr, 0));

      // write
      jsonio::write_contiguous_at(&fh, &chunkdisp, sendptr, bufsize[i], 1, &req);
      MPI_Wait(&req, MPI_STATUS_IGNORE);

      chunkdisp += bufsize[i];
    }
  }

  // get total size
  MPI_Allreduce(MPI_IN_PLACE, &allsize, 1, MPI_INT32_T, MPI_SUM, MPI_COMM_WORLD);

  // update pointer
  disp += allsize;
}

DEFINE_MEMBER(void, wait_bc_exchange)(std::set<int>& queue, const int mode)
{
  const float64 deadlock_detection_limit = 60;

  bool    status   = true;
  int     recvmode = RecvMode | mode;
  float64 wclock   = nix::wall_clock();

  while (queue.empty() == false) {
    // find chunk for unpacking
    auto iter = std::find_if(queue.begin(), queue.end(),
                             [&](int i) { return chunkvec[i]->set_boundary_query(recvmode); });

    if (nix::wall_clock() - wclock > deadlock_detection_limit) {
      status = false;
      DEBUGPRINT(std::cerr, "Possible deadlock has been detected at rank = %04d!\n", thisrank);
      DEBUGPRINT(std::cerr, "BoundaryMode = %04d\n", mode);
      DEBUGPRINT(std::cerr, "Remaining chunks:\n");

      for (auto it = queue.begin(); it != queue.end(); ++it) {
        auto mpibuf = chunkvec[*it]->get_mpi_buffer(mode);

        // show all neighbor information
        DEBUGPRINT(std::cerr, "*   ID = %4d\n", chunkvec[*it]->get_id());
        for (int dirz = -1, iz = 0; dirz <= +1; dirz++, iz++) {
          for (int diry = -1, iy = 0; diry <= +1; diry++, iy++) {
            for (int dirx = -1, ix = 0; dirx <= +1; dirx++, ix++) {
              int id   = chunkvec[*it]->get_nb_id(dirz, diry, dirx);
              int rank = chunkvec[*it]->get_nb_rank(dirz, diry, dirx);
              int flag = 0;
              MPI_Test(&mpibuf->recvreq(iz, iy, ix), &flag, MPI_STATUS_IGNORE);
              DEBUGPRINT(std::cerr, "    nb: ID = %4d, rank = %4d, flag = %2d\n", id, rank, flag);
            }
          }
        }
      }
    }

    // not found
    if (iter == queue.end())
      continue;

    // unpack
    chunkvec[*iter]->set_boundary_end(mode);
    queue.erase(*iter);
  }

  // exit on error
  MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_CXX_BOOL, MPI_LAND, MPI_COMM_WORLD);

  if (status == false) {
    ERRORPRINT("Error: exit possibly due to deadlock in wait_bc_exchange()\n");
    finalize(-1);
    exit(-1);
  }
}

DEFINE_MEMBER(void, finalize)(int cleanup)
{
  // save snapshot
  this->save();

  // write log
  this->save_log(true);

  // MPI
  finalize_mpi(cleanup);
}

DEFINE_MEMBER(, Application)() : mpi_init_with_nullptr(false), cleanup(0), periodic{1, 1, 1}
{
}

DEFINE_MEMBER(, Application)
(int argc, char** argv) : mpi_init_with_nullptr(false), cleanup(0), periodic{1, 1, 1}
{
  cl_argc = argc;
  cl_argv = argv;
}

DEFINE_MEMBER(int, main)(std::ostream& out)
{
  //
  // initialize the application
  //
  initialize(cl_argc, cl_argv);

  //
  // set initial condition
  //
  setup();

  // main loop
  while (is_push_needed()) {
    //
    // output log and diagnostics
    //
    log();
    diagnostic(out);

    //
    // advance physical quantities by one step
    //
    push();

    //
    // exit if elapsed time exceeds a limit
    //
    if (get_available_etime() < 0) {
      break;
    }

    //
    // rebuild chankmap if needed
    //
    rebuild_chunkmap();

    //
    // increment step and time
    //
    increment_time();
  }

  //
  // finalize the application
  //
  finalize(cleanup);

  return 0;
}

#undef DEFINE_MEMBER

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
