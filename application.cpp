// -*- C++ -*-
#include "application.hpp"

#define DEFINE_MEMBER(type, name)                                                                  \
  template <class Chunk, class ChunkMap>                                                           \
  type BaseApplication<Chunk, ChunkMap>::name

DEFINE_MEMBER(void, setup_cmd_default)()
{
  const float64     etmax = 60 * 60 * 24;
  const float64     ptmax = common::HUGEVAL;
  const std::string fn    = "default.json";

  parser.add<std::string>("config", 'c', "configuration file", true, fn);
  parser.add<std::string>("load", 'l', "load file for restart", false, "");
  parser.add<std::string>("save", 's', "save file for restart", false, "");
  parser.add<float64>("tmax", 't', "maximum physical time", false, ptmax);
  parser.add<float64>("emax", 'e', "maximum elased time [sec]", false, etmax);
}

DEFINE_MEMBER(void, parse_cmd_default)(int argc, char **argv)
{
  parser.parse_check(argc, argv);

  tmax     = parser.get<float64>("tmax");
  emax     = parser.get<float64>("emax");
  loadfile = parser.get<std::string>("load");
  savefile = parser.get<std::string>("save");
}

DEFINE_MEMBER(void, parse_cfg_default)(std::string cfg)
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
  int ncx  = root["Ncx"].get<int>();
  int ncy  = root["Ncy"].get<int>();
  int ncz  = root["Ncz"].get<int>();
  ndims[0] = nz * ncz;
  ndims[1] = ny * ncy;
  ndims[2] = nx * ncx;
  ndims[3] = ndims[0] * ndims[1] * ndims[2];
  cdims[0] = ncz;
  cdims[1] = ncy;
  cdims[2] = ncx;
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

DEFINE_MEMBER(void, initialize_mpi_default)(int *argc, char ***argv)
{
  // initialize MPI
  MPI_Init(argc, argv);
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

DEFINE_MEMBER(void, finalize_mpi_default)()
{
  // release stdout/stderr
  mpistream::finalize();

  MPI_Finalize();
}

DEFINE_MEMBER(, BaseApplication)(int argc, char **argv)
{
  // setup command line parser
  setup_cmd_default();

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
  bufsize = 1024 * 16;
  sendbuf.resize(bufsize);
  recvbuf.resize(bufsize);
}

DEFINE_MEMBER(void, load)()
{
  if (!loadfile.empty()) {
    std::cout << tfm::format("load snapshot from %s\n", loadfile.c_str());
  }
  std::cout << tfm::format("no load file specified\n");
}

DEFINE_MEMBER(void, save)()
{
  if (!savefile.empty()) {
    std::cout << tfm::format("save snapshot to %s\n", savefile.c_str());
  }
  std::cout << tfm::format("no save file specified\n");
}

DEFINE_MEMBER(void, setup)()
{
  std::cout << tfm::format("setup() called\n");
}

DEFINE_MEMBER(void, initialize)()
{
  std::cout << tfm::format("initialize() called\n");

  // load snapshot
  this->load();
}

DEFINE_MEMBER(void, finalize)()
{
  std::cout << tfm::format("finalize() called\n");

  // save snapshot
  this->save();

  // MPI
  finalize_mpi_default();
}

DEFINE_MEMBER(void, diagnostic)()
{
  return;
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

DEFINE_MEMBER(float64, available_etime)()
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
  float64   sendbuf[nc];

  // calculate global workload per chunk
  for (int i = 0; i < nc; i++) {
    sendbuf[i]  = 0.0;
    workload[i] = 0.0;
  }

  // local workload
  for (int i = 0; i < numchunk; i++) {
    int id      = chunkvec[i]->get_id();
    sendbuf[id] = chunkvec[i]->get_load();
  }

  // global workload
  MPI_Allreduce(sendbuf, workload.get(), nc, MPI_REAL8, MPI_SUM, MPI_COMM_WORLD);
}

DEFINE_MEMBER(void, initialize_chunkmap)()
{
  const int nc = cdims[3];
  const int mc = nc / nprocess;

  // error check
  if (nc % nprocess != 0) {
    std::cerr << tfm::format("Error: "
                             "number of chunk %8d, "
                             "number of process %8d\n",
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
      chunkvec[i].reset(new Chunk(id, dims));
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

DEFINE_MEMBER(void, rebuild_chunkmap)()
{
  const int nc = cdims[0] * cdims[1] * cdims[2];
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

DEFINE_MEMBER(void, sendrecv_chunk)(int newrank[])
{
  const int dims[3] = {ndims[0] / cdims[0], ndims[1] / cdims[1], ndims[2] / cdims[2]};
  const int ncmax   = std::numeric_limits<int>::max();
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
      std::cerr << tfm::format("Error: chunk ID = %4d; "
                               "current rank = %4d; "
                               "newrank = %4d\n",
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
        PtrChunk p(new Chunk(0, dims));
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
              [](const PtrChunk &x, const PtrChunk &y) { return x->get_id() < y->get_id(); });

    // reset numchunk
    numchunk = 0;
    for (int i = 0; i < chunkvec.size(); i++) {
      if (chunkvec[i]->get_id() == ncmax)
        break;
      numchunk++;
    }

    // better to resize if too much memory is used
    if (numchunk > 2 * chunkvec.size())
      chunkvec.resize(numchunk);
  }
}

DEFINE_MEMBER(void, set_chunk_neighbors)()
{
  for (int i = 0; i < numchunk; i++) {
    int ix, iy, iz;
    int id = chunkvec[i]->get_id();
    chunkmap->get_coordinate(id, iz, iy, ix);

    for (int jz = -1; jz <= +1; jz++) {
      for (int jy = -1; jy <= +1; jy++) {
        for (int jx = -1; jx <= +1; jx++) {
          // neighbor coordiante
          int cz = iz + jz;
          int cy = iy + jy;
          int cx = ix + jx;
          cz     = cz < 0 ? cdims[0] - 1 : cz;
          cz     = cz >= cdims[0] ? cdims[0] - cz : cz;
          cy     = cy < 0 ? cdims[1] - 1 : cy;
          cy     = cy >= cdims[1] ? cdims[1] - cy : cy;
          cx     = cx < 0 ? cdims[2] - 1 : cx;
          cx     = cx >= cdims[2] ? cdims[2] - cx : cx;

          // set neighbor id
          int nbid = chunkmap->get_chunkid(cz, cy, cx);
          chunkvec[i]->set_nb_id(jz, jy, jx, nbid);

          // set neighbor rank
          int nbrank = chunkmap->get_rank(nbid);
          chunkvec[i]->set_nb_rank(jz, jy, jx, nbrank);
        }
      }
    }
  }
}

DEFINE_MEMBER(void, print_info)(std::ostream &out, int verbose)
{
  out << tfm::format("\n"
                     "----- <<< BEGIN INFORMATION >>> -----"
                     "\n"
                     "number of processes : %4d\n"
                     "this rank           : %4d\n",
                     nprocess, thisrank);

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

    out << tfm::format("\n--- %-8d local chunkes ---\n", numchunk);

    for (int i = 0; i < numchunk; i++) {
      lsum += chunkvec[i]->get_load();
      out << tfm::format("   chunk[%.8d]:  workload = %10.4f\n", chunkvec[i]->get_id(),
                         chunkvec[i]->get_load());
    }

    out << "\n"
        << tfm::format("*** load of %12.8f %% (ideally %12.8f %%)\n", lsum / gsum * 100,
                       1.0 / nprocess * 100);
  }

  out << "\n"
         "----- <<< END INFORMATION >>> -----"
         "\n\n";
}

DEFINE_MEMBER(int, main)(std::ostream &out)
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
  while (is_push_needed()) {
    out << tfm::format("*** step = %8d (time = %10.5f)\n", curstep, curtime);

    // advance everything by one step
    push();
    // output diagnostic if needed
    diagnostic();

    // exit if elapsed time exceed a limit
    if (available_etime() < 0) {
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

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
