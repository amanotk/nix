// -*- C++ -*-
#ifndef _APPLICATION_HPP_
#define _APPLICATION_HPP_

///
/// Main Application Class
///
/// $Id$
///
#include "utils/common.hpp"
#include "utils/tinyformat.hpp"
#include "utils/cmdline.hpp"
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
  using cmdparser = cmdline::parser;
  using json = nlohmann::ordered_json;

  typedef std::unique_ptr<BaseBalancer> PtrBalancer;
  typedef std::unique_ptr<Chunk>        PtrChunk;
  typedef std::unique_ptr<ChunkMap>     PtrChunkMap;
  typedef std::unique_ptr<char[]>       PtrByte;
  typedef std::unique_ptr<float64[]>    PtrFloat;
  typedef std::vector<PtrChunk>         ChunkVec;

  std::string  config;    ///< configuration file name
  cmdparser    parser;    ///< command line parser
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


  // setup default command-line options
  void setup_cmd_default();

  // default command-line parsing
  void parse_cmd_default(int argc, char ** argv);

  // default configuration file parsing
  void parse_cfg_default(std::string cfg);

  // default MPI initialization
  void initialize_mpi_default(int *argc, char ***argv);

  // default MPI finalization
  void finalize_mpi_default();

public:
  /// Constructor
  BaseApplication(int argc, char **argv);

  virtual void load();

  virtual void save();

  virtual void setup();

  virtual void initialize();

  virtual void finalize();

  virtual void diagnostic();

  virtual void push();

  virtual bool is_push_needed();

  virtual float64 available_etime();

  virtual void calc_workload();

  virtual void initialize_chunkmap();

  virtual void rebuild_chunkmap();

  virtual void sendrecv_chunk(int newrank[]);

  virtual void set_chunk_neighbors();

  virtual void print_info(std::ostream &out, int verbose=0);

  virtual int main(std::ostream &out);
};


// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
