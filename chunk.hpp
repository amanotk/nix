// -*- C++ -*-
#ifndef _CHUNK_HPP_
#define _CHUNK_HPP_

#include "nix.hpp"

NIX_NAMESPACE_BEGIN

static constexpr int DIRTAG_BIT  = 5;
static constexpr int DIRTAG_SIZE = 1 << DIRTAG_BIT;

// template trick to set number of neighbors
template <int N>
struct NbSize;

///
/// @brief Base class for Chunk
/// @tparam Ndim number of dimension
///
template <int Ndim>
class Chunk
{
protected:
  static constexpr int nbsize = NbSize<Ndim>::size; ///< number of neighbors
  static int           tagmask;                     ///< mask for directional tag
  static int           dirtag[DIRTAG_SIZE];         ///< directional tag

  int                  myid;           ///< chunk ID
  int                  nbid[nbsize];   ///< neighboring chunk ID
  int                  nbrank[nbsize]; ///< neighboring chunk MPI rank
  int                  dims[Ndim];     ///< number of grids
  std::vector<float64> load;           ///< load array of chunk

  virtual void initialize(const int dims[Ndim], const int id);

public:
  ///
  /// @brief get maximum Chunk ID allowable
  /// @return maximum Chunk ID
  ///
  static int get_max_id();

  /// @brief default constructor
  Chunk();

  ///
  /// @brief constructor
  /// @param dims number of grids for each direction
  /// @param id Chunk ID
  ///
  Chunk(const int dims[Ndim], const int id = 0);

  ///
  /// @brief reset load of chunk
  ///
  virtual void reset_load();

  ///
  /// @brief return load array with each element representing different types of operation
  /// @return load array
  ///
  virtual std::vector<float64> get_load();

  ///
  /// @brief return sum of loads for different operations
  /// @return total load of Chunk
  ///
  virtual float64 get_total_load();

  ///
  /// @brief pack the content of Chunk into given `buffer`
  /// @param buffer pointer to buffer to pack
  /// @param address first address of buffer to which the data will be packed
  /// @return `address` + (number of bytes packed as a result)
  ///
  virtual int pack(void* buffer, const int address);

  ///
  /// @brief unpack the content of Chunk from given `buffer`
  /// @param buffer point to buffer from unpack
  /// @param address first address of buffer from which the data will be unpacked
  /// @return `address` + (number of bytes unpacked as a result)
  ///
  virtual int unpack(void* buffer, const int address);

  ///
  /// @brief pack diagnostic information
  /// @param mode mode of diagnostic packing
  /// @param buffer pointer to buffer to bpack
  /// @param address  first address of buffer to which the data will be packed
  /// @return `address` + (number of bytes packed as a result)
  ///
  virtual int pack_diagnostic(const int mode, void* buffer, const int address);

  ///
  /// @brief query status of boundary exchange
  /// @param mode mode of boundary exchange
  /// @return true if boundary exchange is finished and false otherwise
  ///
  virtual bool set_boundary_query(const int mode);

  ///
  /// @brief set physical boundary condition
  /// @param mode mode of boundary exchange
  ///
  virtual void set_boundary_physical(const int mode);

  ///
  /// @brief begin boundary exchange
  /// @param mode mode of boundary exchange
  ///
  virtual void set_boundary_begin(const int mode);

  ///
  /// @brief end boundary exchange
  /// @param mode mode of boundary exchange
  ///
  virtual void set_boundary_end(const int mode);

  ///
  /// @brief set Chunk ID
  /// @param id ID to be set
  ///
  void set_id(const int id);

  ///
  /// @brief get Chunk ID
  /// @return Chunk ID
  ///
  int get_id();

  ///
  /// @brief get tag mask
  /// @return tag mask
  ///
  int get_tagmask();

  ///
  /// @brief set neighbor ID for 1D Chunk
  /// @param dirx direction in x
  /// @param id ID of neighbor Chunk
  ///
  void set_nb_id(const int dirx, const int id);

  ///
  /// @brief set neighbor ID for 2D Chunk
  /// @param diry direction in y
  /// @param dirx direction in x
  /// @param id ID of neighbor Chunk
  ///
  void set_nb_id(const int diry, const int dirx, const int id);

  ///
  /// @brief set neighbor ID for 3D Chunk
  /// @param dirz direction in z
  /// @param diry direction in y
  /// @param dirx direction in x
  /// @param id ID of neighbor Chunk
  ///
  void set_nb_id(const int dirz, const int diry, const int dirx, const int id);

  ///
  /// @brief get neighbor Chunk ID for 1D Chunk
  /// @param dirx direction in x
  /// @return neighbor Chunk ID
  ///
  int get_nb_id(const int dirx);

  ///
  /// @brief get neighbor Chunk ID for 2D Chunk
  /// @param diry direction in y
  /// @param dirx direction in x
  /// @return neighbor Chunk ID
  ///
  int get_nb_id(const int diry, const int dirx);

  ///
  /// @brief get neighbor Chunk ID for 3D Chunk
  /// @param dirz direction in z
  /// @param diry direction in y
  /// @param dirx direction in x
  /// @return neighbor Chunk ID
  ///
  int get_nb_id(const int dirz, const int diry, const int dirx);

  ///
  /// @brief set neighbor Chunk rank for 1D Chunk
  /// @param dirx direction in x
  /// @param rank neighbor Chunk rank
  ///
  void set_nb_rank(const int dirx, const int rank);

  ///
  /// @brief set neighbor Chunk rank for 2D Chunk
  /// @param diry direction in y
  /// @param dirx direction in x
  /// @param rank neighbor Chunk rank
  ///
  void set_nb_rank(const int diry, const int dirx, const int rank);

  ///
  /// @brief set neigbhro Chunk rank for 3D Chunk
  /// @param dirz direction in z
  /// @param diry direction in y
  /// @param dirx direction in x
  /// @param rank neighbor Chunk rank
  ///
  void set_nb_rank(const int dirz, const int diry, const int dirx, const int rank);

  ///
  /// @brief get neighbor Chunk rank for 1D Chunk
  /// @param dirx direction x
  /// @return neighbor Chunk rank
  ///
  int get_nb_rank(const int dirx);

  ///
  /// @brief get neighbor Chunk rank for 2D Chunk
  /// @param diry direction in y
  /// @param dirx direction in x
  /// @return neighbor Chunk rank
  ///
  int get_nb_rank(const int diry, const int dirx);

  ///
  /// @brief get neighbor Chunk rank for 3D Chunk
  /// @param dirz direction in z
  /// @param diry direction in y
  /// @param dirx direction in x
  /// @return neighbor Chunk rank
  ///
  int get_nb_rank(const int dirz, const int diry, const int dirx);

  ///
  /// @brief get send tag for 1D Chunk
  /// @param dirx direction in x
  /// @return send tag
  ///
  int get_sndtag(const int dirx);

  ///
  /// @brief get send tag for 2D Chunk
  /// @param diry direction in y
  /// @param dirx direction in x
  /// @return send tag
  ///
  int get_sndtag(const int diry, const int dirx);

  ///
  /// @brief get send tag for 3D Chunk
  /// @param dirz direction in z
  /// @param diry direction in y
  /// @param dirx direction in x
  /// @return send tag
  ///
  int get_sndtag(const int dirz, const int diry, const int dirx);

  ///
  /// @brief get receive tag for 1D Chunk
  /// @param dirx direction in x
  /// @return receive tag
  ///
  int get_rcvtag(const int dirx);

  ///
  /// @brief get receive tag for 2D Chunk
  /// @param diry direction in y
  /// @param dirx direction in x
  /// @return receive tag
  ///
  int get_rcvtag(const int diry, const int dirx);

  ///
  /// @brief get receive tag for 3D Chunk
  /// @param dirz direction in z
  /// @param diry direction in y
  /// @param dirx direction in x
  /// @return receive tag
  ///
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

template <int Ndim>
constexpr int Chunk<Ndim>::nbsize;
template <int Ndim>
int Chunk<Ndim>::tagmask;
template <int Ndim>
int Chunk<Ndim>::dirtag[DIRTAG_SIZE];

//
// implementation of small methods follows
//

template <int Ndim>
inline int Chunk<Ndim>::get_max_id()
{
  int max_int32_t = std::numeric_limits<int32_t>::max();
  return tagmask ^ max_int32_t;
}

template <int Ndim>
inline void Chunk<Ndim>::set_id(const int id)
{
  assert(!(tagmask & id)); // ID must be < 2^(31-DIRTAG_BIT)

  myid = id;
}

template <int Ndim>
inline int Chunk<Ndim>::get_id()
{
  return myid;
}

template <int Ndim>
inline int Chunk<Ndim>::get_tagmask()
{
  return tagmask;
}

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

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
