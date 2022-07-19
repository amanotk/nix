// -*- C++ -*-
#ifndef _APPLICATION_HPP_
#define _APPLICATION_HPP_

#include "balancer.hpp"
#include "buffer.hpp"
#include "cmdline.hpp"
#include "common.hpp"
#include "json.hpp"
#include "mpistream.hpp"
#include "tinyformat.hpp"

#include "debug.hpp"

///
/// Definition of BaseApplication Class
///
template <class Chunk, class ChunkMap>
class BaseApplication
{
protected:
  using cmdparser = cmdline::parser;
  using json      = nlohmann::ordered_json;

  typedef std::unique_ptr<BaseBalancer> PtrBalancer;
  typedef std::unique_ptr<Chunk>        PtrChunk;
  typedef std::unique_ptr<ChunkMap>     PtrChunkMap;
  typedef std::unique_ptr<char[]>       PtrByte;
  typedef std::unique_ptr<float64[]>    PtrFloat;
  typedef std::vector<PtrChunk>         ChunkVec;

  int         cl_argc;  ///< command-line argc
  char **     cl_argv;  ///< command-line argv
  std::string config;   ///< configuration file name
  cmdparser   parser;   ///< command line parser
  float64     wclock;   ///< wall clock time at initialization
  PtrBalancer balancer; ///< load balancer
  int         numchunk; ///< number of chunkes in current process
  ChunkVec    chunkvec; ///< chunk array
  PtrChunkMap chunkmap; ///< global chunkmap
  PtrFloat    workload; ///< global load array
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
  int    bufsize;               ///< size of buffer
  Buffer sendbuf;               ///< send buffer
  Buffer recvbuf;               ///< recv buffer
  bool   mpi_init_with_nullptr; ///< for testing purspose

  // setup default command-line options
  void setup_cmd_default()
  {
    const float64     etmax = 60 * 60 * 24;
    const float64     ptmax = common::HUGEVAL;
    const std::string fn    = "default.json";

    parser.add<std::string>("config", 'c', "configuration file", false, fn);
    parser.add<std::string>("load", 'l', "load file for restart", false, "");
    parser.add<std::string>("save", 's', "save file for restart", false, "");
    parser.add<float64>("tmax", 't', "maximum physical time", false, ptmax);
    parser.add<float64>("emax", 'e', "maximum elased time [sec]", false, etmax);
  }

  // default command-line parsing
  void parse_cmd_default(int argc, char **argv)
  {
    parser.parse_check(argc, argv);

    tmax     = parser.get<float64>("tmax");
    emax     = parser.get<float64>("emax");
    loadfile = parser.get<std::string>("load");
    savefile = parser.get<std::string>("save");
  }

  // default configuration file parsing
  void parse_cfg_default(std::string cfg)
  {
    json root;

    // read configuration file
    config = cfg;
    {
      std::ifstream f(config.c_str());
      f >> root;
    }

    // delt and delh
    delt = root["delt"].get<float64>();
    delh = root["delt"].get<float64>();

    // get dimensions
    int nx   = root["Nx"].get<int>();
    int ny   = root["Ny"].get<int>();
    int nz   = root["Nz"].get<int>();
    int cx   = root["Cx"].get<int>();
    int cy   = root["Cy"].get<int>();
    int cz   = root["Cz"].get<int>();
    ndims[0] = nz * cz;
    ndims[1] = ny * cy;
    ndims[2] = nx * cx;
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

  // default MPI initialization
  void initialize_mpi_default(int *argc, char ***argv)
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

  // default MPI finalization
  void finalize_mpi_default()
  {
    // release stdout/stderr
    mpistream::finalize();

    MPI_Finalize();
  }

  // get neighbor coordiante with boundary condition taken into account
  int get_nb_coord(const int coord, const int delta, const int dir)
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

public:
  /// default constructor
  BaseApplication() : mpi_init_with_nullptr(false)
  {
  }

  /// Constructor
  BaseApplication(int argc, char **argv) : mpi_init_with_nullptr(false)
  {
    cl_argc = argc;
    cl_argv = argv;
  }

  virtual void initialize(int argc, char **argv)
  {
    LOGPRINT1(std::cout, "Function %s called\n", __func__);

    // setup command line parser
    setup_cmd_default();

    // parse command line
    parse_cmd_default(argc, argv);

    // configuration file
    parse_cfg_default(parser.get<std::string>("config"));

    // initialize current physical time and time step
    curstep = 0;
    curtime = 0.0;

    // periodic boundary flag
    periodic[0] = 1;
    periodic[1] = 1;
    periodic[2] = 1;

    // MPI
    initialize_mpi_default(&argc, &argv);

    // chunkmap
    initialize_chunkmap();

    // load balancer
    balancer.reset(new BaseBalancer());

    // buffer; 16 kB by default
    bufsize = 1024 * 16;
    sendbuf.resize(bufsize);
    recvbuf.resize(bufsize);
  }

  virtual void load()
  {
    if (!loadfile.empty()) {
      LOGPRINT1(std::cout, "Load snapshot from %s\n", loadfile.c_str());
    }
    LOGPRINT1(std::cout, "No load file specified\n");
  }

  virtual void save()
  {
    if (!savefile.empty()) {
      LOGPRINT1(std::cout, "Save snapshot to %s\n", savefile.c_str());
    }
    LOGPRINT1(std::cout, "No save file specified\n");
  }

  virtual void setup()
  {
    LOGPRINT1(std::cout, "Function %s called\n", __func__);

    // load snapshot
    this->load();
  }

  virtual void finalize()
  {
    LOGPRINT1(std::cout, "Function %s called\n", __func__);

    // save snapshot
    this->save();

    // MPI
    finalize_mpi_default();
  }

  virtual void diagnostic()
  {
    LOGPRINT1(std::cout, "Function %s called\n", __func__);

    print_info(std::cout, 1);
  }

  virtual void push()
  {
    curtime += delt;
    curstep++;
  }

  virtual bool is_push_needed()
  {
    if (curtime < tmax) {
      return true;
    }
    return false;
  }

  virtual float64 get_available_etime()
  {
    float64 etime;

    if (thisrank == 0) {
      etime = MPI_Wtime() - wclock;
    }
    MPI_Bcast(&etime, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return emax - etime;
  }

  virtual void calc_workload()
  {
    const int nc = cdims[3];

    // calculate global workload per chunk
    for (int i = 0; i < nc; i++) {
      workload[i] = 0.0;
    }

    // local workload
    for (int i = 0; i < numchunk; i++) {
      int id       = chunkvec[i]->get_id();
      workload[id] = chunkvec[i]->get_load();
    }

    // global workload
    MPI_Allreduce(MPI_IN_PLACE, workload.get(), nc, MPI_REAL8, MPI_SUM, MPI_COMM_WORLD);
  }

  virtual void initialize_chunkmap()
  {
    const int nc = cdims[3];
    const int mc = nc / nprocess;

    // error check
    if (nc % nprocess != 0) {
      ERRORPRINT("number of chunk   = %8d\n"
                 "number of process = %8d\n",
                 nc, nprocess);
      finalize_mpi_default();
      exit(-1);
    }

    //
    // initialize global chunkmap
    // (chunkes are equally distributed over all processes)
    //
    chunkmap.reset(new ChunkMap(cdims));

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
        chunkvec[i].reset(new Chunk(dims, id));
      }
      numchunk = mc;
    }
    set_chunk_neighbors();

    //
    // allocate workload and initialize
    //
    workload.reset(new float64[nc]);

    for (int i = 0; i < nc; i++) {
      workload[i] = 0.0;
    }
  }

  virtual void rebuild_chunkmap()
  {
    const int nc = cdims[3];
    int       rank[nc];

    // calculate global workload
    calc_workload();

    // calculate new decomposition
    balancer->partition(nc, nprocess, workload.get(), rank);

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

  virtual bool validate_chunkmap()
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

  virtual void sendrecv_chunk(int newrank[])
  {
    const int dims[3] = {ndims[0] / cdims[0], ndims[1] / cdims[1], ndims[2] / cdims[2]};
    const int ncmax   = Chunk::get_max_id();
    const int nc      = cdims[0] * cdims[1] * cdims[2];
    const int spos_l  = 0;
    const int spos_r  = sendbuf.size / 2;
    const int rpos_l  = 0;
    const int rpos_r  = recvbuf.size / 2;

    char *sbuf_l = sendbuf.get(spos_l);
    char *sbuf_r = sendbuf.get(spos_r);
    char *rbuf_l = nullptr;
    char *rbuf_r = nullptr;

    //
    // pack and calculate message size to be sent
    //
    for (int i = 0; i < numchunk; i++) {
      int id = chunkvec[i]->get_id();

      if (newrank[id] == thisrank - 1) {
        // send to left
        int size = chunkvec[i]->pack(0, sbuf_l);
        sbuf_l += size;
        chunkvec[i]->set_id(ncmax); // to be removed
      } else if (newrank[id] == thisrank + 1) {
        // send to right
        int size = chunkvec[i]->pack(0, sbuf_r);
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

      // send/recv chunkes
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
        int   size    = 0;
        char *rbuf_l0 = rbuf_l;

        while ((rbuf_l - rbuf_l0) < rbufcnt_l) {
          PtrChunk p(new Chunk(dims, 0));
          size = p->unpack(0, rbuf_l);
          chunkvec.push_back(std::move(p));
          rbuf_l += size;
        }
      }

      // unpack buffer from right
      {
        int   size    = 0;
        char *rbuf_r0 = rbuf_r;

        while ((rbuf_r - rbuf_r0) < rbufcnt_r) {
          PtrChunk p(new Chunk(dims, 0));
          size = p->unpack(0, rbuf_r);
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

  virtual void set_chunk_neighbors()
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

  virtual void print_info(std::ostream &out, int verbose = 0)
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

  virtual int main(std::ostream &out)
  {

    //
    // + initialize object
    // + setup
    // + output diagnostic if needed
    //
    initialize(cl_argc, cl_argv);
    setup();
    diagnostic();

    // main loop
    while (is_push_needed()) {
      out << tfm::format("*** step = %8d (time = %10.5f)\n", curstep, curtime);

      // advance everything by one step
      push();

      // output diagnostic if needed
      diagnostic();

      // exit if elapsed time exceed a limit
      if (get_available_etime() < 0) {
        break;
      }

      // rebuild chankmap if needed
      rebuild_chunkmap();
    }

    //
    // + save current snapshot if needed
    // + finalize MPI
    //
    finalize();

    return 0;
  }
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
