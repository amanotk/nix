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

///
/// @brief Base Application class
/// @tparam Chunk Chunk type
/// @tparam ChunkMap ChunkMap type
///
template <class Chunk, class ChunkMap>
class Application
{
private:
  using cmdparser = cmdline::parser;
  using json      = nix::json;

protected:
  using PtrBalancer = std::unique_ptr<Balancer>;
  using PtrChunk    = std::unique_ptr<Chunk>;
  using PtrChunkMap = std::unique_ptr<ChunkMap>;
  using FloatVec    = std::vector<float64>;
  using ChunkVec    = std::vector<PtrChunk>;

  int         retcode;  ///< default return code
  int         cl_argc;  ///< command-line argc
  char **     cl_argv;  ///< command-line argv
  std::string cfg_file; ///< configuration file name
  json        cfg_json; ///< configuration json object
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
  float64     delh;     ///< cell size
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
  virtual void parse_cmd(int argc, char **argv);

  ///
  /// @brief parse configuration file
  ///
  virtual void parse_cfg();

  ///
  /// @brief initialize MPI
  /// @param argc number of arguments
  /// @param argv array of arguments
  ///
  void initialize_mpi(int *argc, char ***argv);

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
  virtual void initialize(int argc, char **argv);

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
  /// @brief perform various diagnostics output
  /// @param out output stream to which console message will be printed (if any)
  ///
  virtual void diagnostic(std::ostream &out);

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
  /// @brief calculate work load of chunks
  ///
  virtual void calc_workload();

  ///
  /// @brief initialize chunkmap object
  ///
  virtual void initialize_chunkmap();

  ///
  /// @brief rebuild chunkmap object by performing load balancing
  ///
  virtual void rebuild_chunkmap();

  ///
  /// @brief check the validity of chunkmap object
  /// @return true if the chunkmap is valid and false otherwise
  ///
  virtual bool validate_chunkmap();

  ///
  /// @brief perform MPI send/recv of chunks for load balancing
  /// @param newrank array of ranks to which chunks are assigned
  ///
  virtual void sendrecv_chunk(std::vector<int> &newrank);

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
  virtual void write_chunk_all(MPI_File &fh, size_t &disp, const int mode);

  ///
  /// @brief wait boundary exchange operation in `queue` and perform unpacking
  /// @param queue list of chunk IDs performing boundary exchange
  /// @param mode mode of boundary exchange
  ///
  virtual void wait_bc_exchange(std::set<int> &queue, const int mode);

  ///
  /// @brief print debugging information
  /// @param out output stream
  /// @param verbose level of verbosity
  ///
  virtual void print_info(std::ostream &out, int verbose = 0);

  ///
  /// @brief finalize application
  /// @param cleanup return code
  ///
  virtual void finalize(int cleanup = 0);

public:
  /// @brief default construct (unnecessary?)
  Application();

  ///
  /// @brief constructor
  /// @param argc number of arguments
  /// @param argv array of arguments
  ///
  Application(int argc, char **argv);

  ///
  /// @brief main loop of simulation
  /// @param out output stream
  /// @return return code of application
  ///
  virtual int main(std::ostream &out);
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

DEFINE_MEMBER(void, parse_cmd)(int argc, char **argv)
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

  // delt and delh
  delt = cfg_json["delt"].get<float64>();
  delh = cfg_json["delt"].get<float64>();

  // get dimensions
  int nx = cfg_json["Nx"].get<int>();
  int ny = cfg_json["Ny"].get<int>();
  int nz = cfg_json["Nz"].get<int>();
  int cx = cfg_json["Cx"].get<int>();
  int cy = cfg_json["Cy"].get<int>();
  int cz = cfg_json["Cz"].get<int>();

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
  xlim[1] = delh * ndims[2];
  xlim[2] = xlim[1] - xlim[0];
  ylim[0] = 0;
  ylim[1] = delh * ndims[1];
  ylim[2] = ylim[1] - ylim[0];
  zlim[0] = 0;
  zlim[1] = delh * ndims[0];
  zlim[2] = zlim[1] - zlim[0];
}

DEFINE_MEMBER(void, initialize_mpi)(int *argc, char ***argv)
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
    wclock = MPI_Wtime();
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

DEFINE_MEMBER(void, initialize)(int argc, char **argv)
{
  // parse command line arguments
  parse_cmd(argc, argv);

  // parse configuration file
  parse_cfg();

  // initialize current physical time and time step
  curstep = 0;
  curtime = 0.0;

  // periodic boundary flag
  periodic[0] = 1;
  periodic[1] = 1;
  periodic[2] = 1;

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

DEFINE_MEMBER(void, diagnostic)(std::ostream &out)
{
  out << tfm::format("*** step = %8d (time = %10.5f)\n", curstep, curtime);
  print_info(out, 1);
}

DEFINE_MEMBER(void, push)()
{
  curtime += delt;
  curstep++;
}

DEFINE_MEMBER(bool, is_push_needed)()
{
  if (curtime < tmax) {
    return true;
  }
  return false;
}

DEFINE_MEMBER(float64, get_available_etime)()
{
  float64 etime;

  if (thisrank == 0) {
    etime = MPI_Wtime() - wclock;
  }
  MPI_Bcast(&etime, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return emax - etime;
}

DEFINE_MEMBER(void, calc_workload)()
{
  const int nc = cdims[3];

  // calculate global workload per chunk
  for (int i = 0; i < nc; i++) {
    workload[i] = 0.0;
  }

  // local workload
  for (int i = 0; i < numchunk; i++) {
    int id       = chunkvec[i]->get_id();
    workload[id] = chunkvec[i]->get_total_load();
  }

  // global workload
  MPI_Allreduce(MPI_IN_PLACE, workload.data(), nc, MPI_REAL8, MPI_SUM, MPI_COMM_WORLD);
}

DEFINE_MEMBER(void, initialize_chunkmap)()
{
  const int nc = cdims[3];
  const int mc = nc / nprocess;

  // error check
  if (nc % nprocess != 0) {
    ERRORPRINT("number of chunk   = %8d\n"
               "number of process = %8d\n",
               nc, nprocess);
    finalize(-1);
    exit(-1);
  }

  //
  // initialize global chunkmap
  // (chunkes are equally distributed over all processes)
  //
  chunkmap = std::make_unique<ChunkMap>(cdims);

  for (int id = 0; id < nc; id++) {
    chunkmap->set_rank(id, id / mc);
  }

  //
  // initialize local chunkvec
  //
  {
    int dims[3] = {ndims[0] / cdims[0], ndims[1] / cdims[1], ndims[2] / cdims[2]};
    int id      = thisrank * mc;

    chunkvec.resize(mc);
    for (int i = 0; i < mc; i++, id++) {
      chunkvec[i] = std::make_unique<Chunk>(dims, id);
    }
    numchunk = mc;
  }
  set_chunk_neighbors();

  //
  // allocate workload and initialize
  //
  workload.resize(nc);

  for (int i = 0; i < nc; i++) {
    workload[i] = 0.0;
  }
}

DEFINE_MEMBER(void, rebuild_chunkmap)()
{
  const int        nc = cdims[3];
  std::vector<int> rank(nc);

  // calculate global workload
  calc_workload();

  // calculate new decomposition
  balancer->partition(nprocess, workload, rank);

  //
  // chunk send/recv
  //
  sendrecv_chunk(rank);

  //
  // reset rank
  //
  for (int id = 0; id < nc; id++) {
    chunkmap->set_rank(id, rank[id]);
  }
  set_chunk_neighbors();
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

DEFINE_MEMBER(void, sendrecv_chunk)(std::vector<int> &newrank)
{
  const int dims[3] = {ndims[0] / cdims[0], ndims[1] / cdims[1], ndims[2] / cdims[2]};
  const int ncmax   = Chunk::get_max_id();
  const int nc      = cdims[0] * cdims[1] * cdims[2];

  //
  // check buffer size and reallocate if necessary
  //
  {
    int sendsize_l = 0;
    int sendsize_r = 0;

    for (int i = 0; i < numchunk; i++) {
      int id = chunkvec[i]->get_id();

      if (newrank[id] == thisrank) {
        // no need for transfer
      } else if (newrank[id] == thisrank - 1) {
        // send to left
        sendsize_l += chunkvec[i]->pack(nullptr, 0);
      } else if (newrank[id] == thisrank + 1) {
        // send to right
        sendsize_r += chunkvec[i]->pack(nullptr, 0);
      } else {
        // FIXME; error handling
      }
    }

    // make buffer size the same for all processes
    int bufsize = std::max(sendbuf.size, 2 * std::max(sendsize_l, sendsize_r));
    MPI_Allreduce(MPI_IN_PLACE, &bufsize, 1, MPI_INT32_T, MPI_MAX, MPI_COMM_WORLD);

    sendbuf.resize(bufsize);
    recvbuf.resize(bufsize);
  }

  const int spos_l = 0;
  const int spos_r = sendbuf.size / 2;
  const int rpos_l = 0;
  const int rpos_r = recvbuf.size / 2;

  uint8_t *sbuf_l = sendbuf.get(spos_l);
  uint8_t *sbuf_r = sendbuf.get(spos_r);
  uint8_t *rbuf_l = nullptr;
  uint8_t *rbuf_r = nullptr;

  //
  // pack and calculate message size to be sent
  //
  for (int i = 0; i < numchunk; i++) {
    int id = chunkvec[i]->get_id();

    if (newrank[id] == thisrank - 1) {
      // send to left
      int size = chunkvec[i]->pack(sbuf_l, 0);
      sbuf_l += size;
      chunkvec[i]->set_id(ncmax); // to be removed
    } else if (newrank[id] == thisrank + 1) {
      // send to right
      int size = chunkvec[i]->pack(sbuf_r, 0);
      sbuf_r += size;
      chunkvec[i]->set_id(ncmax); // to be removed
    } else if (newrank[id] == thisrank) {
      // no need to seed
      continue;
    } else {
      // something wrong
      ERRORPRINT("chunk ID     = %4d\n"
                 "current rank = %4d\n"
                 "newrank      = %4d\n",
                 id, thisrank, newrank[id]);
      finalize(-1);
      exit(-1);
    }
  }

  //
  // send/recv chunkes
  //
  {
    MPI_Request request[4];
    MPI_Status  status[4];

    int rbufcnt_l  = 0;
    int rbufcnt_r  = 0;
    int sbufsize_l = sbuf_l - sendbuf.get(spos_l);
    int sbufsize_r = sbuf_r - sendbuf.get(spos_r);
    int rbufsize_l = recvbuf.size / 2;
    int rbufsize_r = recvbuf.size / 2;
    int rank_l     = thisrank > 0 ? thisrank - 1 : MPI_PROC_NULL;
    int rank_r     = thisrank < nprocess - 1 ? thisrank + 1 : MPI_PROC_NULL;

    // send/recv chunks
    sbuf_l = sendbuf.get(spos_l);
    sbuf_r = sendbuf.get(spos_r);
    rbuf_l = recvbuf.get(rpos_l);
    rbuf_r = recvbuf.get(rpos_r);

    MPI_Isend(sbuf_l, sbufsize_l, MPI_BYTE, rank_l, 1, MPI_COMM_WORLD, &request[0]);
    MPI_Isend(sbuf_r, sbufsize_r, MPI_BYTE, rank_r, 2, MPI_COMM_WORLD, &request[1]);
    MPI_Irecv(rbuf_l, rbufsize_l, MPI_BYTE, rank_l, 2, MPI_COMM_WORLD, &request[2]);
    MPI_Irecv(rbuf_r, rbufsize_r, MPI_BYTE, rank_r, 1, MPI_COMM_WORLD, &request[3]);

    MPI_Waitall(4, request, status);
    MPI_Get_count(&status[2], MPI_BYTE, &rbufcnt_l);
    MPI_Get_count(&status[3], MPI_BYTE, &rbufcnt_r);

    // unpack buffer from left
    {
      int      size    = 0;
      uint8_t *rbuf_l0 = rbuf_l;

      while ((rbuf_l - rbuf_l0) < rbufcnt_l) {
        PtrChunk p = std::make_unique<Chunk>(dims, 0);
        size       = p->unpack(rbuf_l, 0);
        chunkvec.push_back(std::move(p));
        rbuf_l += size;
      }
    }

    // unpack buffer from right
    {
      int      size    = 0;
      uint8_t *rbuf_r0 = rbuf_r;

      while ((rbuf_r - rbuf_r0) < rbufcnt_r) {
        PtrChunk p = std::make_unique<Chunk>(dims, 0);
        size       = p->unpack(rbuf_r, 0);
        chunkvec.push_back(std::move(p));
        rbuf_r += size;
      }
    }
  }

  //
  // sort chunkvec and remove unused chunkes
  //
  {
    std::sort(chunkvec.begin(), chunkvec.end(),
              [](const PtrChunk &x, const PtrChunk &y) { return x->get_id() < y->get_id(); });

    // reset numchunk
    numchunk = 0;
    for (int i = 0; i < chunkvec.size(); i++) {
      if (chunkvec[i]->get_id() == ncmax)
        break;
      numchunk++;
    }

    // better to resize if too much memory is used
    if (chunkvec.size() > 2 * numchunk) {
      chunkvec.resize(numchunk);
    }
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

DEFINE_MEMBER(void, write_chunk_all)(MPI_File &fh, size_t &disp, const int mode)
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
    uint8_t *   sendptr   = sendbuf.get();

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

DEFINE_MEMBER(void, wait_bc_exchange)(std::set<int> &queue, const int mode)
{
  int recvmode = nix::RecvMode | mode;

  while (queue.empty() == false) {
    // find chunk for unpacking
    auto iter = std::find_if(queue.begin(), queue.end(),
                             [&](int i) { return chunkvec[i]->set_boundary_query(recvmode); });

    // not found
    if (iter == queue.end())
      continue;

    // unpack
    chunkvec[*iter]->set_boundary_end(mode);
    queue.erase(*iter);
  }
}

DEFINE_MEMBER(void, print_info)(std::ostream &out, int verbose)
{
  LOGPRINT0(out, "\n");
  LOGPRINT0(out, "----- <<< BEGIN INFORMATION >>> -----");
  LOGPRINT0(out, "\n");
  LOGPRINT0(out, "Number of processes : %4d\n", nprocess);
  LOGPRINT0(out, "This rank           : %4d\n", thisrank);

  // local chunk
  if (verbose >= 1) {
    const int nc   = cdims[3];
    float64   gsum = 0.0;
    float64   lsum = 0.0;

    // global workload
    calc_workload();
    for (int i = 0; i < nc; i++) {
      gsum += workload[i];
    }

    LOGPRINT0(out, "\n");
    LOGPRINT0(out, "--- %-8d local chunkes ---\n", numchunk);

    for (int i = 0; i < numchunk; i++) {
      int id = chunkvec[i]->get_id();
      lsum += workload[id];
      LOGPRINT0(out, "   chunk[%.8d]:  workload = %10.4f\n", id, workload[id]);
    }

    LOGPRINT0(out, "\n");
    LOGPRINT0(out, "*** load of %12.8f %% (ideally %12.8f %%)\n", lsum / gsum * 100,
              1.0 / nprocess * 100);
  }

  LOGPRINT0(out, "\n");
  LOGPRINT0(out, "----- <<< END INFORMATION >>> -----\n");
  LOGPRINT0(out, "\n");
  LOGPRINT0(out, "\n");
}

DEFINE_MEMBER(void, finalize)(int cleanup)
{
  // save snapshot
  this->save();

  // MPI
  if (cleanup == 0) {
    cleanup = retcode;
  }
  finalize_mpi(cleanup);
}

DEFINE_MEMBER(, Application)() : mpi_init_with_nullptr(false), retcode(0)
{
}

DEFINE_MEMBER(, Application)(int argc, char **argv) : mpi_init_with_nullptr(false)
{
  cl_argc = argc;
  cl_argv = argv;
}

DEFINE_MEMBER(int, main)(std::ostream &out)
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
    // output diagnostics
    //
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
  }

  //
  // finalize the application
  //
  finalize();

  return 0;
}

#undef DEFINE_MEMBER

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
