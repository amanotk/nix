// -*- C++ -*-
#ifndef _APPLICATION_HPP_
#define _APPLICATION_HPP_

///
/// Main Application Class
///
/// $Id$
///
#include "base/common.hpp"
#include "base/cmdparser.hpp"
#include "base/cfgparser.hpp"
#include "base/mpistream.hpp"
#include "balancer.hpp"
#include "buffer.hpp"


///
/// Definition of BaseApplication Class
///
template <class Patch, class PatchMap>
class BaseApplication
{
protected:
  typedef std::unique_ptr<BaseBalancer> PtrBalancer;
  typedef std::unique_ptr<Patch>        PtrPatch;
  typedef std::unique_ptr<PatchMap>     PtrPatchMap;
  typedef std::unique_ptr<char[]>       PtrByte;
  typedef std::unique_ptr<float64[]>    PtrFloat;
  typedef std::vector<PtrPatch>         PatchVec;

  CmdParser    parser;    ///< command line parser
  CfgParser    config;    ///< configuration file parser
  float64      clock0;    ///< wall clock time at initialization
  PtrBalancer  balancer;  ///< load balancer
  int          numpatch;  ///< number of patches in current process
  PatchVec     patchvec;  ///< patch array
  PtrPatchMap  patchmap;  ///< global patchmap
  PtrFloat     workload;  ///< global load array
  float64      tmax;      ///< maximum physical time
  float64      emax;      ///< maximum elapsed time
  std::string  loadfile;  ///< snapshot to be loaded
  std::string  savefile;  ///< snapshot to be saved
  int          ndims[4];  ///< global grid dimensions
  int          pdims[4];  ///< patch dimensions
  int          step;      ///< current iteration step
  float64      tnow;      ///< current time
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
    config.read(cfg.c_str());

    // others
    int nx    = config.getAs<int>("Nx");
    int ny    = config.getAs<int>("Ny");
    int nz    = config.getAs<int>("Nz");
    int px    = config.getAs<int>("Px");
    int py    = config.getAs<int>("Py");
    int pz    = config.getAs<int>("Pz");
    delt      = config.getAs<float64>("delt");
    delh      = config.getAs<float64>("delh");
    cc        = config.getAs<float64>("cc");

    // store dimensions
    ndims[0]  = nz*pz;
    ndims[1]  = ny*py;
    ndims[2]  = nx*px;
    ndims[3]  = ndims[0]*ndims[1]*ndims[2];
    pdims[0]  = pz;
    pdims[1]  = py;
    pdims[2]  = px;
    pdims[3]  = pdims[0]*pdims[1]*pdims[2];

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
      clock0 = MPI_Wtime();
    }
    MPI_Bcast(&clock0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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
    step = 0;
    tnow = 0.0;

    // MPI
    initialize_mpi_default(&argc, &argv);

    // patchmap
    initialize_patchmap();

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
      std::cout << tfm::format("load snapshoft from %s\n",
                               loadfile.c_str());
    }
    std::cout << "no load file specified\n";
  }


  virtual void save()
  {
    if( ! savefile.empty() ) {
      std::cout << tfm::format("save snapshoft to %s\n",
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
    tnow = tnow + delt;
    step++;
  }


  virtual bool need_push()
  {
    if( tnow < tmax ) {
      return true;
    }
    return false;
  }


  virtual float64 available_etime()
  {
    float64 etime;

    if( thisrank == 0 ) {
      etime = MPI_Wtime() - clock0;
    }
    MPI_Bcast(&etime, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return emax - etime;
  }


  virtual void calc_workload()
  {
    const int np = pdims[3];
    float64 sendbuf[np];

    // calculate global workload per patch
    for(int i=0; i < np ;i++) {
      sendbuf[i]  = 0.0;
      workload[i] = 0.0;
    }

    // local workload
    for(int i=0; i < numpatch ;i++) {
      int pid      = patchvec[i]->get_id();
      sendbuf[pid] = patchvec[i]->get_load();
    }

    // global workload
    MPI_Allreduce(sendbuf, workload.get(), np,
                  MPI_REAL8, MPI_SUM, MPI_COMM_WORLD);
  }


  virtual void initialize_patchmap()
  {
    const int np = pdims[3];
    const int mp = np / nprocess;

    // error check
    if( np % nprocess != 0 ) {
      std::cerr << tfm::format("Error: "
                               "number of patch %8d, "
                               "number of process %8d\n",
                               np, nprocess);
      finalize_mpi_default();
      exit(-1);
    }

    //
    // initialize global patchmap
    // (patches are equally distributed over all processes)
    //
    patchmap.reset(new PatchMap(pdims));

    for(int pid=0; pid < np ;pid++) {
      patchmap->set_rank(pid, pid / mp);
    }

    //
    // initialize local patchvec
    //
    {
      int dims[3] = {ndims[0]/pdims[0],
                     ndims[1]/pdims[1],
                     ndims[2]/pdims[2]};
      int pid = thisrank * mp;

      patchvec.resize(mp);
      for(int i=0; i < mp ;i++, pid++) {
        patchvec[i].reset(new Patch(pid, dims));
      }
      numpatch = mp;
    }

    //
    // allocate workload and initialize
    //
    workload.reset(new float64[np]);

    for(int i=0; i < np ;i++) {
      workload[i] = 0.0;
    }
  }


  virtual void rebuild_patchmap()
  {
    const int np = pdims[0]*pdims[1]*pdims[2];
    int     rank[np];

    // calculate global workload
    calc_workload();

    // calculate new decomposition
    balancer->partition(np, nprocess, workload.get(), rank);

    //
    // patch send/recv
    //
    sendrecv_patch(rank);
  }


  virtual void sendrecv_patch(int newrank[])
  {
    const int dims[3] = {ndims[0]/pdims[0],
                         ndims[1]/pdims[1],
                         ndims[2]/pdims[2]};
    const int npmax   = std::numeric_limits<int>::max();
    const int np      = pdims[0]*pdims[1]*pdims[2];
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
    for(int i=0; i < numpatch ;i++) {
      int hi = patchvec[i]->get_id();

      if( newrank[hi] == thisrank-1 ) {
        // send to left
        int size = patchvec[i]->pack(0, sbuf_l);
        sbuf_l += size;
        patchvec[i]->set_id(npmax); // to be removed
      } else if( newrank[hi] == thisrank+1 ) {
        // send to right
        int size = patchvec[i]->pack(0, sbuf_r);
        sbuf_r += size;
        patchvec[i]->set_id(npmax); // to be removed
      } else if( newrank[hi] == thisrank ) {
        // no need to seed
        continue;
      } else {
        // something wrong
        std::cerr << tfm::format("Error: patch ID = %4d; "
                                 "current rank = %4d; "
                                 "newrank = %4d\n",
                                 hi, thisrank, newrank[hi]);
      }
    }

    //
    // send/recv patches
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

      // send/recv patches
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
          PtrPatch p(new Patch(0, dims));
          size = p->unpack(0, rbuf_l);
          patchvec.push_back(std::move(p));
          rbuf_l += size;
        }
      }

      // unpack buffer from right
      {
        int size = 0;
        char *rbuf_r0 = rbuf_r;

        while( (rbuf_r - rbuf_r0) < rbufcnt_r ) {
          PtrPatch p(new Patch(0, dims));
          size = p->unpack(0, rbuf_r);
          patchvec.push_back(std::move(p));
          rbuf_r += size;
        }
      }
    }

    //
    // sort patchvec and remove unused patches
    //
    {
      std::sort(patchvec.begin(), patchvec.end(),
                [](const PtrPatch &x, const PtrPatch &y)
                {return x->get_id() < y->get_id();});

      // reset numpatch
      numpatch = 0;
      for(int i=0; i < patchvec.size() ;i++) {
        if( patchvec[i]->get_id() == npmax ) break;
        numpatch++;
      }

      // better to resize if too much memory is used
      if( numpatch > 2*patchvec.size() ) patchvec.resize(numpatch);
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


    // local patch
    if( verbose >= 1 ) {
      const int np = pdims[3];
      float64 gsum = 0.0;
      float64 lsum = 0.0;

      // global workload
      calc_workload();
      for(int i=0; i < np ;i++) {
        gsum += workload[i];
      }

      out << tfm::format("\n--- %-8d local patches ---\n", numpatch);

      for(int i=0; i < numpatch ;i++) {
        lsum += patchvec[i]->get_load();
        out << tfm::format("   patch[%.8d]:  workload = %10.4f\n",
                           patchvec[i]->get_id(), patchvec[i]->get_load());
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

};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
