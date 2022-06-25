// -*- C++ -*-
#ifndef _APPLICATION_HPP_
#define _APPLICATION_HPP_

///
/// Main Application Class
///
/// $Id$
///
#include "utils/common.hpp"
#include "utils/cmdparser.hpp"
#include "utils/cfgparser.hpp"
#include "utils/json.hpp"
#include "utils/mpistream.hpp"
#include "balancer.hpp"
#include "buffer.hpp"


///
/// Definition of BaseApplication Class
///
template <class Chunk, class ChunkMap>
class BaseApplication
{
protected:
  using json = nlohmann::json;
  typedef std::unique_ptr<BaseBalancer> PtrBalancer;
  typedef std::unique_ptr<Chunk>        PtrChunk;
  typedef std::unique_ptr<ChunkMap>     PtrChunkMap;
  typedef std::unique_ptr<char[]>       PtrByte;
  typedef std::unique_ptr<float64[]>    PtrFloat;
  typedef std::vector<PtrChunk>         ChunkVec;

  CmdParser    parser;    ///< command line parser
  std::string  config;    ///< configuration file
  float64      wclock;    ///< wall clock time at initialization
  PtrBalancer  balancer;  ///< load balancer
  int          numchunk;  ///< number of chunkes in current process
  ChunkVec     chunkvec;  ///< chunk array
  PtrChunkMap  chunkmap;  ///< global chunkmap
  PtrFloat     workload;  ///< global load array
  float64      tmax;      ///< maximum physical time
  float64      emax;      ///< maximum elapsed time
  std::string  loadfile;  ///< snapshot to be loaded
  std::string  savefile;  ///< snapshot to be saved
  int          ndims[4];  ///< global grid dimensions
  int          cdims[4];  ///< chunk dimensions
  int          curstep;   ///< current iteration step
  float64      curtime;   ///< current time
  float64      delt;      ///< time step
  float64      delh;      ///< cell size
  float64      cc;        ///< speed of light
  float64      xlim[3];   ///< physical domain in x
  float64      ylim[3];   ///< physical domain in y
  float64      zlim[3];   ///< physical domain in z

  // MPI related
  int      nprocess;     ///< number of mpi processes
  int      thisrank;     ///< my rank
  int      bufsize;      ///< size of buffer
  Buffer   sendbuf;      ///< send buffer
  Buffer   recvbuf;      ///< recv buffer


  // default command-line parsing
  void parse_cmd_default(int argc, char ** argv)
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
    int nx    = root["Nx"].get<int>();
    int ny    = root["Ny"].get<int>();
    int nz    = root["Nz"].get<int>();
    int ncx   = root["Ncx"].get<int>();
    int ncy   = root["Ncy"].get<int>();
    int ncz   = root["Ncz"].get<int>();
    ndims[0]  = nz*ncz;
    ndims[1]  = ny*ncy;
    ndims[2]  = nx*ncx;
    ndims[3]  = ndims[0]*ndims[1]*ndims[2];
    cdims[0]  = ncz;
    cdims[1]  = ncy;
    cdims[2]  = ncx;
    cdims[3]  = cdims[0]*cdims[1]*cdims[2];

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
    MPI_Init(argc, argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocess);
    MPI_Comm_rank(MPI_COMM_WORLD, &thisrank);

    // store initial clock
    if( thisrank == 0 ) {
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


public:
  /// Constructor
  BaseApplication(int argc, char **argv)
  {
    // parse command line
    parse_cmd_default(argc, argv);

    // configuration file
    parse_cfg_default(parser.get<std::string>("config"));

    // initialize current physical time and time step
    curstep = 0;
    curtime = 0.0;

    // MPI
    initialize_mpi_default(&argc, &argv);

    // chunkmap
    initialize_chunkmap();

    // load balancer
    balancer.reset(new BaseBalancer());

    // buffer; 16 kB by default
    bufsize = 1024*16;
    sendbuf.resize(bufsize);
    recvbuf.resize(bufsize);
  }


  virtual void load()
  {
    if( ! loadfile.empty() ) {
      std::cout << tfm::format("load snapshot from %s\n",
                               loadfile.c_str());
    }
    std::cout << "no load file specified\n";
  }


  virtual void save()
  {
    if( ! savefile.empty() ) {
      std::cout << tfm::format("save snapshot to %s\n",
                               savefile.c_str());
    }
    std::cout << "no save file specified\n";
  }


  virtual void setup()
  {
    std::cout << "setup() called\n";
  }


  virtual void initialize()
  {
    std::cout << "initialize() called\n";

    // load snapshot
    this->load();
  }


  virtual void finalize()
  {
    std::cout << "finalize() called\n";

    // save snapshot
    this->save();

    // MPI
    finalize_mpi_default();
  }


  virtual void diagnostic()
  {
    return;
  }


  virtual void push()
  {
    curtime += delt;
    curstep++;
  }


  virtual bool need_push()
  {
    if( curtime < tmax ) {
      return true;
    }
    return false;
  }


  virtual float64 available_etime()
  {
    float64 etime;

    if( thisrank == 0 ) {
      etime = MPI_Wtime() - wclock;
    }
    MPI_Bcast(&etime, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return emax - etime;
  }


  virtual void calc_workload()
  {
    const int np = cdims[3];
    float64 sendbuf[np];

    // calculate global workload per chunk
    for(int i=0; i < np ;i++) {
      sendbuf[i]  = 0.0;
      workload[i] = 0.0;
    }

    // local workload
    for(int i=0; i < numchunk ;i++) {
      int pid      = chunkvec[i]->get_id();
      sendbuf[pid] = chunkvec[i]->get_load();
    }

    // global workload
    MPI_Allreduce(sendbuf, workload.get(), np,
                  MPI_REAL8, MPI_SUM, MPI_COMM_WORLD);
  }


  virtual void initialize_chunkmap()
  {
    const int np = cdims[3];
    const int mp = np / nprocess;

    // error check
    if( np % nprocess != 0 ) {
      std::cerr << tfm::format("Error: "
                               "number of chunk %8d, "
                               "number of process %8d\n",
                               np, nprocess);
      finalize_mpi_default();
      exit(-1);
    }

    //
    // initialize global chunkmap
    // (chunkes are equally distributed over all processes)
    //
    chunkmap.reset(new ChunkMap(cdims));

    for(int pid=0; pid < np ;pid++) {
      chunkmap->set_rank(pid, pid / mp);
    }

    //
    // initialize local chunkvec
    //
    {
      int dims[3] = {ndims[0]/cdims[0],
                     ndims[1]/cdims[1],
                     ndims[2]/cdims[2]};
      int pid = thisrank * mp;

      chunkvec.resize(mp);
      for(int i=0; i < mp ;i++, pid++) {
        chunkvec[i].reset(new Chunk(pid, dims));
      }
      numchunk = mp;
    }

    //
    // allocate workload and initialize
    //
    workload.reset(new float64[np]);

    for(int i=0; i < np ;i++) {
      workload[i] = 0.0;
    }
  }


  virtual void rebuild_chunkmap()
  {
    const int np = cdims[0]*cdims[1]*cdims[2];
    int     rank[np];

    // calculate global workload
    calc_workload();

    // calculate new decomposition
    balancer->partition(np, nprocess, workload.get(), rank);

    //
    // chunk send/recv
    //
    sendrecv_chunk(rank);
  }


  virtual void sendrecv_chunk(int newrank[])
  {
    const int dims[3] = {ndims[0]/cdims[0],
                         ndims[1]/cdims[1],
                         ndims[2]/cdims[2]};
    const int npmax   = std::numeric_limits<int>::max();
    const int np      = cdims[0]*cdims[1]*cdims[2];
    const int spos_l  = 0;
    const int spos_r  = sendbuf.size/2;
    const int rpos_l  = 0;
    const int rpos_r  = recvbuf.size/2;

    char *sbuf_l = sendbuf.get(spos_l);
    char *sbuf_r = sendbuf.get(spos_r);
    char *rbuf_l = nullptr;
    char *rbuf_r = nullptr;

    //
    // pack and calculate message size to be sent
    //
    for(int i=0; i < numchunk ;i++) {
      int hi = chunkvec[i]->get_id();

      if( newrank[hi] == thisrank-1 ) {
        // send to left
        int size = chunkvec[i]->pack(0, sbuf_l);
        sbuf_l += size;
        chunkvec[i]->set_id(npmax); // to be removed
      } else if( newrank[hi] == thisrank+1 ) {
        // send to right
        int size = chunkvec[i]->pack(0, sbuf_r);
        sbuf_r += size;
        chunkvec[i]->set_id(npmax); // to be removed
      } else if( newrank[hi] == thisrank ) {
        // no need to seed
        continue;
      } else {
        // something wrong
        std::cerr << tfm::format("Error: chunk ID = %4d; "
                                 "current rank = %4d; "
                                 "newrank = %4d\n",
                                 hi, thisrank, newrank[hi]);
      }
    }

    //
    // send/recv chunkes
    //
    {
      MPI_Request request[4];
      MPI_Status  status[4];

      int rbufcnt_l = 0;
      int rbufcnt_r = 0;
      int sbufsize_l = sbuf_l - sendbuf.get(spos_l);
      int sbufsize_r = sbuf_r - sendbuf.get(spos_r);
      int rbufsize_l = recvbuf.size/2;
      int rbufsize_r = recvbuf.size/2;
      int rank_l = thisrank > 0          ? thisrank-1 : MPI_PROC_NULL;
      int rank_r = thisrank < nprocess-1 ? thisrank+1 : MPI_PROC_NULL;

      // send/recv chunkes
      sbuf_l = sendbuf.get(spos_l);
      sbuf_r = sendbuf.get(spos_r);
      rbuf_l = recvbuf.get(rpos_l);
      rbuf_r = recvbuf.get(rpos_r);

      MPI_Isend(sbuf_l, sbufsize_l, MPI_BYTE, rank_l, 1,
                MPI_COMM_WORLD, &request[0]);
      MPI_Isend(sbuf_r, sbufsize_r, MPI_BYTE, rank_r, 2,
                MPI_COMM_WORLD, &request[1]);
      MPI_Irecv(rbuf_l, rbufsize_l, MPI_BYTE, rank_l, 2,
                MPI_COMM_WORLD, &request[2]);
      MPI_Irecv(rbuf_r, rbufsize_r, MPI_BYTE, rank_r, 1,
                MPI_COMM_WORLD, &request[3]);

      MPI_Waitall(4, request, status);
      MPI_Get_count(&status[2], MPI_BYTE, &rbufcnt_l);
      MPI_Get_count(&status[3], MPI_BYTE, &rbufcnt_r);

      // unpack buffer from left
      {
        int size = 0;
        char *rbuf_l0 = rbuf_l;

        while( (rbuf_l - rbuf_l0) < rbufcnt_l ) {
          PtrChunk p(new Chunk(0, dims));
          size = p->unpack(0, rbuf_l);
          chunkvec.push_back(std::move(p));
          rbuf_l += size;
        }
      }

      // unpack buffer from right
      {
        int size = 0;
        char *rbuf_r0 = rbuf_r;

        while( (rbuf_r - rbuf_r0) < rbufcnt_r ) {
          PtrChunk p(new Chunk(0, dims));
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
                [](const PtrChunk &x, const PtrChunk &y)
                {return x->get_id() < y->get_id();});

      // reset numchunk
      numchunk = 0;
      for(int i=0; i < chunkvec.size() ;i++) {
        if( chunkvec[i]->get_id() == npmax ) break;
        numchunk++;
      }

      // better to resize if too much memory is used
      if( numchunk > 2*chunkvec.size() ) chunkvec.resize(numchunk);
    }
  }


  virtual void print_info(std::ostream &out, int verbose=0)
  {
    out << tfm::format("\n"
                       "----- <<< BEGIN INFORMATION >>> -----"
                       "\n"
                       "number of processes : %4d\n"
                       "this rank           : %4d\n",
                       nprocess,
                       thisrank);


    // local chunk
    if( verbose >= 1 ) {
      const int np = cdims[3];
      float64 gsum = 0.0;
      float64 lsum = 0.0;

      // global workload
      calc_workload();
      for(int i=0; i < np ;i++) {
        gsum += workload[i];
      }

      out << tfm::format("\n--- %-8d local chunkes ---\n", numchunk);

      for(int i=0; i < numchunk ;i++) {
        lsum += chunkvec[i]->get_load();
        out << tfm::format("   chunk[%.8d]:  workload = %10.4f\n",
                           chunkvec[i]->get_id(), chunkvec[i]->get_load());
      }

      out << "\n"
          << tfm::format("*** load of %12.8f %% (ideally %12.8f %%)\n",
                         lsum/gsum * 100, 1.0/nprocess * 100);
    }

    out <<
      "\n"
      "----- <<< END INFORMATION >>> -----"
      "\n\n";
  }


  int main(std::ostream &out)
  {
    //
    // + print initial info
    // + setup solvers
    // + setup initial condition or load a previous snapshot
    // + output diagnostic if needed
    //
    print_info(out, 1);
    setup();
    initialize();
    diagnostic();

    // main loop
    while( need_push() ) {
      out << tfm::format("*** step = %8d (time = %10.5f)\n",
                         curstep, curtime);

      // advance everything by one step
      push();
      // output diagnostic if needed
      diagnostic();

      // exit if elapsed time exceed a limit
      if( available_etime() < 0 ) {
        break;
      }

      rebuild_chunkmap();
      print_info(out, 1);
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
