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

static constexpr int DIRTAG_BIT = 5;
static constexpr int DIRTAG_SIZE = 1 << DIRTAG_BIT;
static constexpr int NB_SIZE[3] = {3, 9, 27};

///
/// BaseChunk class
///
template <int N>
class BaseChunk
{
protected:
  static int nbsize;              ///< number of neighbors
  static int tagmask;             ///< mask for directional tag
  static int dirtag[DIRTAG_SIZE]; ///< directional tag

  int myid;                   ///< chunk ID
  int nbid[NB_SIZE[N - 1]];   ///< neighboring chunk ID
  int nbrank[NB_SIZE[N - 1]]; ///< neighboring chunk MPI rank
  float64 load;               ///< current load

  int pack_base(const int flag, char *buffer);

  int unpack_base(const int flag, char *buffer);

public:
  // constructor
  BaseChunk(const int id = 0);

  // destructor
  virtual ~BaseChunk();

  // initialize load
  virtual void initialize_load();

  // return load
  virtual float64 get_load();

  // pack
  virtual int pack(const int flag, char *buffer);

  // unpack
  virtual int unpack(const int flag, char *buffer);

  // set id
  void set_id(const int id)
  {
    assert( !(tagmask & id) ); // ID must be < 2^(32-DIRTAG_BIT)

    myid = id;
  }

  // return id
  int get_id()
  {
    return myid;
  }

  // get tag mask
  int get_tagmask()
  {
    return tagmask;
  }

  void set_nb_id(const int dirx, const int id);
  void set_nb_id(const int diry, const int dirx, const int id);
  void set_nb_id(const int dirz, const int diry, const int dirx, const int id);
  void set_nb_rank(const int dirx, const int rank);
  void set_nb_rank(const int diry, const int dirx, const int rank);
  void set_nb_rank(const int dirz, const int diry, const int dirx,
                   const int rank);
  int get_sndtag(const int dirx);
  int get_sndtag(const int diry, const int dirx);
  int get_sndtag(const int dirz, const int diry, const int dirx);
  int get_rcvtag(const int dirx);
  int get_rcvtag(const int diry, const int dirx);
  int get_rcvtag(const int dirz, const int diry, const int dirx);
};

template <int N>
int BaseChunk<N>::nbsize = NB_SIZE[N - 1];
template <int N>
int BaseChunk<N>::tagmask;
template <int N>
int BaseChunk<N>::dirtag[DIRTAG_SIZE];

template <>
inline void BaseChunk<1>::set_nb_id(const int dirx, const int id)
{
  nbid[(dirx + 1)] = id;
}

template <>
inline void BaseChunk<2>::set_nb_id(const int diry, const int dirx,
                                    const int id)
{
  nbid[3 * (diry + 1) + (dirx + 1)] = id;
}

template <>
inline void BaseChunk<3>::set_nb_id(const int dirz, const int diry,
                                    const int dirx, const int id)
{
  nbid[9 * (dirz + 1) + 3 * (diry + 1) + (dirx + 1)] = id;
}

template <>
inline void BaseChunk<1>::set_nb_rank(const int dirx, const int rank)
{
  nbrank[(dirx + 1)] = rank;
}

template <>
inline void BaseChunk<2>::set_nb_rank(const int diry, const int dirx,
                                      const int rank)
{
  nbrank[3 * (diry + 1) + (dirx + 1)] = rank;
}

template <>
inline void BaseChunk<3>::set_nb_rank(const int dirz, const int diry,
                                      const int dirx, const int rank)
{
  nbrank[9 * (dirz + 1) + 3 * (diry + 1) + (dirx + 1)] = rank;
}

template <>
inline int BaseChunk<1>::get_sndtag(const int dirx)
{
  int dir = (dirx + 1);
  return dirtag[dir] | nbid[dir];
}

template <>
inline int BaseChunk<2>::get_sndtag(const int diry, const int dirx)
{
  int dir = 3 * (diry + 1) + (dirx + 1);
  return dirtag[dir] | nbid[dir];
}

template <>
inline int BaseChunk<3>::get_sndtag(const int dirz, const int diry,
                                    const int dirx)
{
  int dir = 9 * (dirz + 1) + 3 * (diry + 1) + (dirx + 1);
  return dirtag[dir] | nbid[dir];
}

template <>
inline int BaseChunk<1>::get_rcvtag(const int dirx)
{
  int dir = (-dirx + 1);
  return dirtag[dir] | myid;
}

template <>
inline int BaseChunk<2>::get_rcvtag(const int diry, const int dirx)
{
  int dir = 3 * (-diry + 1) + (-dirx + 1);
  return dirtag[dir] | myid;
}

template <>
inline int BaseChunk<3>::get_rcvtag(const int dirz, const int diry,
                                    const int dirx)
{
  int dir = 9 * (-dirz + 1) + 3 * (-diry + 1) + (-dirx + 1);
  return dirtag[dir] | myid;
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
