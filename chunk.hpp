// -*- C++ -*-
#ifndef _CHUNK_HPP_
#define _CHUNK_HPP_

///
/// Chunk
///
/// $Id$
///
#include "utils/common.hpp"
#include "utils/cmdparser.hpp"
#include "utils/cfgparser.hpp"
#include "utils/mpistream.hpp"
#include <bitset>

///
/// BaseChunk class
///
template <int N>
class BaseChunk
{
protected:
  int myid;            ///< chunk ID
  int nbid[3][3][3];   ///< neighboring chunk ID
  int nbrank[3][3][3]; ///< neighboring chunk MPI rank
  float64 load;        ///< current load


  int pack_base(const int flag, char* buffer)
  {
    char *buffer0 = buffer;

    std::memcpy(buffer, &myid, sizeof(int));
    buffer += sizeof(int);

    std::memcpy(buffer, &nbid[0][0][0], sizeof(int)*27);
    buffer += sizeof(int)*27;

    std::memcpy(buffer, &nbrank[0][0][0], sizeof(int)*27);
    buffer += sizeof(int)*27;

    std::cout << tfm::format("pack_base  : %3d (%4d byte)\n", myid, buffer-buffer0);
    return buffer - buffer0;
  }


  int unpack_base(const int flag, char* buffer)
  {
    char *buffer0 = buffer;

    std::memcpy(&myid, buffer, sizeof(int));
    buffer += sizeof(int);

    std::memcpy(&nbid[0][0][0], buffer, sizeof(int)*27);
    buffer += sizeof(int)*27;

    std::memcpy(&nbrank[0][0][0], buffer, sizeof(int)*27);
    buffer += sizeof(int)*27;

    std::cout << tfm::format("unpack_base: %3d (%4d byte)\n", myid, buffer-buffer0);
    return buffer - buffer0;
  }


public:
  BaseChunk() = delete;

  // constructor
  BaseChunk(const int id, const int dim[N])
    : myid(id)
  {
    initialize_load();
  }


  // initialize load
  void initialize_load()
  {
    static std::random_device rd;
    static std::mt19937 mt(rd());
    static std::uniform_real_distribution<float64> rand(0.75, +1.25);

    load = rand(mt);
  }


  // return load
  virtual float64 get_load()
  {
    static std::random_device rd;
    static std::mt19937 mt(rd());
    static std::uniform_real_distribution<float64> rand(0.75, +1.25);

    load = rand(mt);
    return load;
  }


  // pack
  virtual int pack(const int flag, char* buffer)
  {
    return pack_base(flag, buffer);
  }


  // unpack
  virtual int unpack(const int flag, char* buffer)
  {
    return unpack_base(flag, buffer);
  }


  // set id
  void set_id(const int id)
  {
    myid = id;
  }


  // return id
  int get_id()
  {
    return myid;
  }


  // set neighbor chunk id
  void set_nb_id(const int dirz, const int diry, const int dirx,
                 const int id)
  {
    nbid[dirz+1][diry+1][dirx+1] = id;
  }


  // set neighbor chunk MPI rank
  void set_nb_rank(const int dirz, const int diry, const int dirx,
                   const int rank)
  {
    nbrank[dirz+1][diry+1][dirx+1] = rank;
  }


  // return send tag
  int get_sndtag(const int dirz, const int diry, const int dirx)
  {
    return nbid[dirz+1][diry+1][dirx+1];
  }


  // return receive tag
  int get_rcvtag(const int dirz, const int diry, const int dirx)
  {
    return myid;
  }

};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
