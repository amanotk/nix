// -*- C++ -*-
#ifndef _CHUNK_HPP_
#define _CHUNK_HPP_

///
/// Chunk
///
/// $Id$
///
#include "base/common.hpp"
#include "base/cmdparser.hpp"
#include "base/cfgparser.hpp"
#include "base/mpistream.hpp"
#include <bitset>

///
/// BaseChunk class
///
template <int N>
class BaseChunk
{
protected:
  int myid;          ///< ID
  int nbid[3][3][3]; ///< neighboring chunk ID
  float64 load;      ///< current load

public:
  BaseChunk() = delete;

  // constructor
  BaseChunk(const int pid, const int dim[N])
    : myid(pid)
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
    char *buffer0 = buffer;

    std::memcpy(buffer, &myid, sizeof(int));
    buffer += sizeof(int);

    std::memcpy(buffer, &nbid[0][0][0], sizeof(int)*27);
    buffer += sizeof(int)*27;

    std::cout << tfm::format("pack  : %3d (%4d byte)\n", myid, buffer-buffer0);
    return buffer - buffer0;
  }


  // unpack
  virtual int unpack(const int flag, char* buffer)
  {
    char *buffer0 = buffer;

    std::memcpy(&myid, buffer, sizeof(int));
    buffer += sizeof(int);

    std::memcpy(&nbid[0][0][0], buffer, sizeof(int)*27);
    buffer += sizeof(int)*27;

    std::cout << tfm::format("unpack: %3d (%4d byte)\n", myid, buffer-buffer0);
    return buffer - buffer0;
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
  void set_neighbor(const int dirz, const int diry, const int dirx,
                    const int pid)
  {
    nbid[dirz+1][diry+1][dirx+1] = pid;
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
