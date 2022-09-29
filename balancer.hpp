// -*- C++ -*-
#ifndef _BALANCER_HPP_
#define _BALANCER_HPP_

#include "nix.hpp"

NIX_NAMESPACE_BEGIN

///
/// @brief Load Balancer
///
class Balancer
{
protected:
  ///
  /// @brief move boundary specified by index forward or backward by one
  ///
  /// This tries to move the boundary specified by `index` either forward or backward. It does
  /// so only if the movement improves the local load balancing.
  ///
  /// @param[in] index an index for `boundary` array
  /// @param[in,out] boundary boundary array
  /// @param[in] meanload mean of load (total load divided by number of ranks)
  /// @param[in,out] rankload load for each rank
  /// @param[in] chunkload load for each chunk
  ///
  void move_boundary(const int index, std::vector<int>& boundary, const float64 meanload,
                     std::vector<float64>& rankload, std::vector<float64>& chunkload);

  ///
  /// @brief apply "smoothing" to assignment specified by boundary
  ///
  /// This tries to smooth local load imbalance by applying `move_boundary` to the given `boundary`
  /// in a staggered manner. For each smoothing, it applies `move_boundary` to odd indices and then
  /// even indices. The smoothing will be performed for the number of times specified by `count`.
  ///
  /// @param[in,out] boundary boundary array
  /// @param[in] load load for each chunk
  /// @param[in] count number of smoothing iteration
  ///
  void smooth_load(std::vector<int>& boundary, std::vector<float64>& load, const int count);

  ///
  /// @brief perform naive sequential assignment of chunks
  ///
  /// @param[in] load load of chunks
  /// @param[out] boundary boundary between ranks
  /// @param[in] dir direction of sequential assignment; +1 for forward, -1 for backward.
  ///
  void doit_sequential(std::vector<float64>& load, std::vector<int>& boundary, const int dir);

  ///
  /// @brief perform assignment of chunks according to SMILEI code (Derouillat et al. 2018)
  ///
  /// @param[in] load load of chunks
  /// @param[out] boundary boundary between ranks
  ///
  void doit_smilei(std::vector<float64>& load, std::vector<int>& boundary);

  ///
  /// @brief return if the boundary array gives appropriate assignment of chunks
  ///
  /// @param[in] Nc number of chunks
  /// @param[in] boundary boundary arrays
  /// @return true if the assignment is appropriate and false otherwise
  ///
  bool validate_boundary(const int Nc, const std::vector<int>& boundary);

public:
  ///
  /// @brief perform partition of chunks into ranks
  ///
  /// @param[in] Nr number of ranks
  /// @param[in] load load for each chunk
  /// @param[out] rank rank for each chunk (as a result of assignment)
  ///
  virtual void partition(const int Nr, std::vector<float64>& load, std::vector<int>& rank);

  ///
  /// @brief calculate rank from boundary
  ///
  /// @param[in] boundary boundary between ranks
  /// @param[out] rank rank of chunk
  ///
  void get_rank(std::vector<int>& boundary, std::vector<int>& rank);

  ///
  /// @brief calculate boundary from rank
  ///
  /// @param[in] rank rank of chunk
  /// @param[out] boundary boundary between ranks
  ///
  void get_boundary(std::vector<int>& rank, std::vector<int>& boundary);

  ///
  /// @brief return array of load for each rank
  ///
  /// @param[in] boundary boundary array
  /// @param[in] load load array for each chunk
  /// @return array of load for each rank
  ///
  std::vector<float64> get_rankload(std::vector<int>& boundary, std::vector<float64>& load);

  ///
  /// @brief print summary of load as a result of assignment
  ///
  /// @param[in] out output stream
  /// @param[in] boundary boundary array
  /// @param[in] load load for each chunk
  ///
  void print_assignment(std::ostream& out, std::vector<int>& boundary, std::vector<float64>& load);
};

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
