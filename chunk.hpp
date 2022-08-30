// -*- C++ -*-
#ifndef _CHUNK_HPP_
#define _CHUNK_HPP_

///
/// Chunk
///
#include "common.hpp"

#include <mpi.h>

static constexpr int DIRTAG_BIT  = 5;
static constexpr int DIRTAG_SIZE = 1 << DIRTAG_BIT;
static constexpr int NB_SIZE[3]  = {3, 9, 27};

///
/// Chunk class
///
template <int N>
class Chunk
{
public:
  enum PackMode {
    PackAll = 1,
    PackAllQuery,
  };

protected:
  static int nbsize;              ///< number of neighbors
  static int tagmask;             ///< mask for directional tag
  static int dirtag[DIRTAG_SIZE]; ///< directional tag

  int     myid;                   ///< chunk ID
  int     nbid[NB_SIZE[N - 1]];   ///< neighboring chunk ID
  int     nbrank[NB_SIZE[N - 1]]; ///< neighboring chunk MPI rank
  int     dims[N];                ///< number of grids
  float64 load;                   ///< current load

  int pack_base(const int mode, void *buffer);

  int unpack_base(const int mode, void *buffer);

  virtual void initialize(const int dims[N], const int id);

public:
  // default constructor
  Chunk();

  // constructor
  Chunk(const int dims[N], const int id = 0);

  // destructor
  virtual ~Chunk();

  // initialize load
  virtual void initialize_load();

  // return load
  virtual float64 get_load();

  // pack
  virtual int pack(const int mode, void *buffer);

  // unpack
  virtual int unpack(const int mode, void *buffer);

  // set ID
  void set_id(const int id)
  {
    assert(!(tagmask & id)); // ID must be < 2^(31-DIRTAG_BIT)

    myid = id;
  }

  // return ID
  int get_id()
  {
    return myid;
  }

  // get tag mask
  int get_tagmask()
  {
    return tagmask;
  }

  // get maximum ID
  static int get_max_id()
  {
    int max_int32_t = std::numeric_limits<int32_t>::max();
    return tagmask ^ max_int32_t;
  }

  void set_nb_id(const int dirx, const int id);

  void set_nb_id(const int diry, const int dirx, const int id);

  void set_nb_id(const int dirz, const int diry, const int dirx, const int id);

  int get_nb_id(const int dirx);

  int get_nb_id(const int diry, const int dirx);

  int get_nb_id(const int dirz, const int diry, const int dirx);

  void set_nb_rank(const int dirx, const int rank);

  void set_nb_rank(const int diry, const int dirx, const int rank);

  void set_nb_rank(const int dirz, const int diry, const int dirx, const int rank);

  int get_nb_rank(const int dirx);

  int get_nb_rank(const int diry, const int dirx);

  int get_nb_rank(const int dirz, const int diry, const int dirx);

  int get_sndtag(const int dirx);

  int get_sndtag(const int diry, const int dirx);

  int get_sndtag(const int dirz, const int diry, const int dirx);

  int get_rcvtag(const int dirx);

  int get_rcvtag(const int diry, const int dirx);

  int get_rcvtag(const int dirz, const int diry, const int dirx);
};

template <int N>
int Chunk<N>::nbsize = NB_SIZE[N - 1];
template <int N>
int Chunk<N>::tagmask;
template <int N>
int Chunk<N>::dirtag[DIRTAG_SIZE];

template <>
inline void Chunk<1>::set_nb_id(const int dirx, const int id)
{
  nbid[(dirx + 1)] = id;
}

template <>
inline void Chunk<2>::set_nb_id(const int diry, const int dirx, const int id)
{
  nbid[3 * (diry + 1) + (dirx + 1)] = id;
}

template <>
inline void Chunk<3>::set_nb_id(const int dirz, const int diry, const int dirx, const int id)
{
  nbid[9 * (dirz + 1) + 3 * (diry + 1) + (dirx + 1)] = id;
}

template <>
inline int Chunk<1>::get_nb_id(const int dirx)
{
  return nbid[(dirx + 1)];
}

template <>
inline int Chunk<2>::get_nb_id(const int diry, const int dirx)
{
  return nbid[3 * (diry + 1) + (dirx + 1)];
}

template <>
inline int Chunk<3>::get_nb_id(const int dirz, const int diry, const int dirx)
{
  return nbid[9 * (dirz + 1) + 3 * (diry + 1) + (dirx + 1)];
}

template <>
inline void Chunk<1>::set_nb_rank(const int dirx, const int rank)
{
  nbrank[(dirx + 1)] = rank;
}

template <>
inline void Chunk<2>::set_nb_rank(const int diry, const int dirx, const int rank)
{
  nbrank[3 * (diry + 1) + (dirx + 1)] = rank;
}

template <>
inline void Chunk<3>::set_nb_rank(const int dirz, const int diry, const int dirx, const int rank)
{
  nbrank[9 * (dirz + 1) + 3 * (diry + 1) + (dirx + 1)] = rank;
}

template <>
inline int Chunk<1>::get_nb_rank(const int dirx)
{
  return nbrank[(dirx + 1)];
}

template <>
inline int Chunk<2>::get_nb_rank(const int diry, const int dirx)
{
  return nbrank[3 * (diry + 1) + (dirx + 1)];
}

template <>
inline int Chunk<3>::get_nb_rank(const int dirz, const int diry, const int dirx)
{
  return nbrank[9 * (dirz + 1) + 3 * (diry + 1) + (dirx + 1)];
}

template <>
inline int Chunk<1>::get_sndtag(const int dirx)
{
  int dir = (dirx + 1);
  return dirtag[dir] | nbid[dir];
}

template <>
inline int Chunk<2>::get_sndtag(const int diry, const int dirx)
{
  int dir = 3 * (diry + 1) + (dirx + 1);
  return dirtag[dir] | nbid[dir];
}

template <>
inline int Chunk<3>::get_sndtag(const int dirz, const int diry, const int dirx)
{
  int dir = 9 * (dirz + 1) + 3 * (diry + 1) + (dirx + 1);
  return dirtag[dir] | nbid[dir];
}

template <>
inline int Chunk<1>::get_rcvtag(const int dirx)
{
  int dir = (-dirx + 1);
  return dirtag[dir] | myid;
}

template <>
inline int Chunk<2>::get_rcvtag(const int diry, const int dirx)
{
  int dir = 3 * (-diry + 1) + (-dirx + 1);
  return dirtag[dir] | myid;
}

template <>
inline int Chunk<3>::get_rcvtag(const int dirz, const int diry, const int dirx)
{
  int dir = 9 * (-dirz + 1) + 3 * (-diry + 1) + (-dirx + 1);
  return dirtag[dir] | myid;
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
