// -*- C++ -*-
#ifndef _CHUNK3D_HPP_
#define _CHUNK3D_HPP_

#include "buffer.hpp"
#include "chunk.hpp"
#include "debug.hpp"
#include "jsonio.hpp"
#include "nix.hpp"
#include "particle.hpp"
#include "xtensorall.hpp"

///
/// @brief Base class for 3D Chunk
/// @tparam Nb number of boundary margins
///
template <int Nb>
class Chunk3D : public Chunk<3>
{
public:
  using json      = nix::json;
  using T_array3d = xt::xtensor_fixed<int, xt::xshape<3, 3, 3>>;
  using T_request = xt::xtensor_fixed<MPI_Request, xt::xshape<3, 3, 3>>;

  ///
  /// @brief MPI buffer
  ///
  struct MpiBuffer {
    MPI_Comm  comm;
    Buffer    sendbuf;
    Buffer    recvbuf;
    T_array3d bufsize;
    T_array3d bufaddr;
    T_request sendreq;
    T_request recvreq;

    ///
    /// constructor
    ///
    MpiBuffer();

    ///
    /// @brief pack the content into given `buffer`
    /// @param buffer pointer to buffer to pack
    /// @param address first address of buffer to which the data will be packed
    /// @return `address` + (number of bytes packed as a result)
    ///
    int pack(void *buffer, const int address);

    ///
    /// @brief unpack the content from given `buffer`
    /// @param buffer pointer to buffer from unpack
    /// @param address first address of buffer to which the data will be packed
    /// @return `address` + (number of bytes packed as a result)
    ///
    int unpack(void *buffer, const int address);
  };
  using PtrMpiBuffer = std::shared_ptr<MpiBuffer>;
  using MpiBufferVec = std::vector<PtrMpiBuffer>;

  /// boundary margin
  static const int boundary_margin = Nb;

protected:
  bool require_sort; ///< sort flag
  int  ndims[3];     ///< number of global grids
  int  offset[3];    ///< global index offset
  int  Lbx;          ///< lower bound in x
  int  Ubx;          ///< upper bound in x
  int  Lby;          ///< lower bound in y
  int  Uby;          ///< upper bound in y
  int  Lbz;          ///< lower bound in z
  int  Ubz;          ///< upper bound in z
  int  sendlb[3][3]; ///< lower bound for send
  int  sendub[3][3]; ///< upper bound for send
  int  recvlb[3][3]; ///< lower bound for recv
  int  recvub[3][3]; ///< upper bound for recv

  xt::xtensor<float64, 1> xc;        ///< x coordiante
  xt::xtensor<float64, 1> yc;        ///< y coordiante
  xt::xtensor<float64, 1> zc;        ///< z coordiante
  float64                 delh;      ///< grid size
  float64                 xlim[3];   ///< physical domain in x
  float64                 ylim[3];   ///< physical domain in y
  float64                 zlim[3];   ///< physical domain in z
  MpiBufferVec            mpibufvec; ///< MPI buffer vector

  ///
  /// @brief pack diagnostic for load array
  /// @param buffer pointer to buffer to pack
  /// @param address first address of buffer to which the data will be packed
  /// @return `address` + (number of bytes packed as a result)
  ///
  int pack_diagnostic_load(void *buffer, const int address);

  ///
  /// @brief pack diagnostic for coordinate array
  /// @param buffer pointer to buffer to pack
  /// @param address first address of buffer to which the data will be packed
  /// @param dir direction of coordinate
  /// @return `address` + (number of bytes packed as a result)
  ///
  int pack_diagnostic_coord(void *buffer, const int address, const int dir);

  ///
  /// @brief pack diagnostic for field quantity
  /// @param buffer pointer to buffer to pack
  /// @param address first address of buffer to which the data will be packed
  /// @param u field quantity to be packed
  /// @return `address` + (number of bytes packed as a result)
  ///
  int pack_diagnostic_field(void *buffer, const int address, xt::xtensor<float64, 4> &u);

  ///
  /// @brief pack diagnostic for particle (single species)
  /// @param buffer pointer to buffer to pack
  /// @param address first address of buffer to which the data will be packed
  /// @param p particle species
  /// @return `address` + (number of bytes packed as a result)
  ///
  int pack_diagnostic_particle(void *buffer, const int address, PtrParticle p);

  ///
  /// @brief pack and start particle boundary exchange
  /// @param mpibuf MPI buffer to be used
  /// @param particle list of particle species
  ///
  void begin_bc_exchange(PtrMpiBuffer mpibuf, ParticleVec &particle);

  ///
  /// @brief wait particle boundary exchange and unpack
  /// @param mpibuf MPI buffer to be used
  /// @param particle list of particle species
  ///
  void end_bc_exchange(PtrMpiBuffer mpibuf, ParticleVec &particle);

  ///
  /// @brief pack and start field quantity boundary exchange
  /// @tparam T typename for field array
  /// @param mpibuf MPI buffer to be used
  /// @param array array of field quantity
  /// @param moment flag if it is for a moment quantity
  ///
  template <typename T>
  void begin_bc_exchange(PtrMpiBuffer mpibuf, T &array, bool moment = false);

  ///
  /// @brief wait field boundary exchange and unpack
  /// @tparam T typename for field array
  /// @param mpibuf MPI buffer to be used
  /// @param array array of field quantity
  /// @param moment flag if it is for a moment quantity
  ///
  template <typename T>
  void end_bc_exchange(PtrMpiBuffer mpibuf, T &array, bool moment = false);

  ///
  /// @brief setup MPI Buffer
  /// @tparam T typename used to represent number of bytes for each element
  /// @param mpibuf MPI buffer to be setup
  /// @param headbyte number of bytes used for header
  /// @param elembyte number of bytes for each element
  ///
  template <typename T>
  void set_mpi_buffer(PtrMpiBuffer mpibuf, const int headbyte, const T &elembyte);

public:
  ///
  /// @brief constructor
  /// @param dims number of grids for each direction
  /// @param id Chunk ID
  ///
  Chunk3D(const int dims[3], const int id = 0);

  ///
  /// @brief destructor (unnecessary?)
  ///
  virtual ~Chunk3D() override;

  ///
  /// @brief pack the content of Chunk into given `buffer`
  /// @param buffer pointer to buffer to pack
  /// @param address first address of buffer to which the data will be packed
  /// @return `address` + (number of bytes packed as a result)
  ///
  virtual int pack(void *buffer, const int address) override;

  ///
  /// @brief unpack the content of Chunk from given `buffer`
  /// @param buffer point to buffer from unpack
  /// @param address first address of buffer from which the data will be unpacked
  /// @return `address` + (number of bytes unpacked as a result)
  ///
  virtual int unpack(void *buffer, const int address) override;

  ///
  /// @brief set the global context of Chunk
  /// @param offset offset for each direction in global dimensions
  /// @param ndims local number of grids for each direction
  ///
  virtual void set_global_context(const int *offset, const int *ndims);

  ///
  /// @brief set MPI communicator to MpiBuffer of given `mode`
  /// @param mode mode to specify MpiBuffer
  /// @param comm MPI communicator to be set to MpiBuffer
  ///
  virtual void set_mpi_communicator(const int mode, MPI_Comm &comm);

  ///
  /// @brief count particles in cells to prepare for sorting
  /// @param particle particle species
  /// @param Lbp first index of particle array to be counted
  /// @param Ubp last index of particle array to be counted (inclusive)
  /// @param reset reset the count array before counting
  ///
  virtual void count_particle(PtrParticle particle, const int Lbp, const int Ubp,
                              bool reset = true);

  ///
  /// @brief peform particle sorting
  /// @param particle list of particle to be sorted
  ///
  virtual void sort_particle(ParticleVec &particle);

  ///
  /// @brief query status of boundary exchange
  /// @param mode mode of boundary exchange
  /// @return true if boundary exchange is finished and false otherwise
  ///
  virtual bool set_boundary_query(const int mode = 0);

  ///
  /// @brief set physical boundary condition
  /// @param mode mode of boundary exchange
  ///
  virtual void set_boundary_physical(const int mode = 0);

  ///
  /// @brief set boundary condition to particle array
  /// @param particle particle species
  /// @param Lbp first index of particle array
  /// @param Ubp last index of particle array (inclusive)
  ///
  virtual void set_boundary_particle(PtrParticle particle, int Lbp, int Ubp);

  ///
  /// @brief setup initial condition
  /// @param config configuration
  ///
  virtual void setup(json &config) = 0;

  ///
  /// @brief begin boundary exchange
  /// @param mode mode of boundary exchange
  ///
  virtual void set_boundary_begin(const int mode) = 0;

  ///
  /// @brief end boundary exchange
  /// @param mode mode of boundary exchange
  ///
  virtual void set_boundary_end(const int mode) = 0;
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
