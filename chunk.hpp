// -*- C++ -*-
#ifndef _CHUNK_HPP_
#define _CHUNK_HPP_

///
/// Chunk
///
/// $Id$
///
#include "utils/common.hpp"
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


  int pack_base(const int flag, char* buffer);

  int unpack_base(const int flag, char* buffer);

public:
  BaseChunk() = delete;

  // constructor
  BaseChunk(const int id, const int dim[N]);

  // initialize load
  virtual void initialize_load();

  // return load
  virtual float64 get_load();

  // pack
  virtual int pack(const int flag, char* buffer);

  // unpack
  virtual int unpack(const int flag, char* buffer);

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
