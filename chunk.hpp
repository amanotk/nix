// -*- C++ -*-
#ifndef _CHUNK_HPP_
#define _CHUNK_HPP_

///
/// Chunk
///
#include "common.hpp"

static constexpr int DIRTAG_BIT  = 5;
static constexpr int DIRTAG_SIZE = 1 << DIRTAG_BIT;

// template trick to set number of neighbors
template <int N>
struct NbSize;

///
/// Chunk class
///
template <int N>
class Chunk
{
protected:
  static constexpr int nbsize = NbSize<N>::size; ///< number of neighbors
  static int           tagmask;                  ///< mask for directional tag
  static int           dirtag[DIRTAG_SIZE];      ///< directional tag

  int                  myid;           ///< chunk ID
  int                  nbid[nbsize];   ///< neighboring chunk ID
  int                  nbrank[nbsize]; ///< neighboring chunk MPI rank
  int                  dims[N];        ///< number of grids
  std::vector<float64> load;           ///< load array of chunk

  virtual void initialize(const int dims[N], const int id);

public:
  // default constructor
  Chunk();

  // constructor
  Chunk(const int dims[N], const int id = 0);

  // destructor
  virtual ~Chunk();

  // reset load
  virtual void reset_load();

  // return load vector
  virtual std::vector<float64> get_load();

  // return total load
  virtual float64 get_total_load();

  // pack
  virtual int pack(void *buffer, const int address);

  // unpack
  virtual int unpack(void *buffer, const int address);

  // pack diagnostic information
  virtual int pack_diagnostic(const int mode, void *buffer, const int address);

  // query boundary exhange status
  virtual bool set_boundary_query(const int mode);

  // set physical boundary condition
  virtual void set_boundary_physical(const int mode);

  // begin boundary exchange
  virtual void set_boundary_begin(const int mode);

  // end boundary exchange
  virtual void set_boundary_end(const int mode);

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

template <>
struct NbSize<1> {
  static constexpr int size = 3;
};
template <>
struct NbSize<2> {
  static constexpr int size = 9;
};
template <>
struct NbSize<3> {
  static constexpr int size = 27;
};

template <int N>
constexpr int Chunk<N>::nbsize;
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
