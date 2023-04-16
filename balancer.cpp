// -*- C++ -*-
#include "balancer.hpp"

#define DEFINE_MEMBER(type, name) type Balancer::name

NIX_NAMESPACE_BEGIN

DEFINE_MEMBER(void, assign)
(std::vector<float64>& load, std::vector<int>& boundary, bool init)
{
  //
  // The size of `load` array should be the number of chunks.
  // The size of `boundary` array should be the number of rank plus one.
  //
  // The chunk specified by `i_chunk` should be assigned to the rank specified by `i_rank` when the
  // following condition is met:
  //
  //     boundary[i_rank] <= i_chunk < boundary[i_rank+1]
  //
  // Therefore, our task is to find an appropriate boundary array first and then use it to calculate
  // `rank` array for output.
  //

  if (init == true) {
    doit_binary_search(load, boundary);
  } else {
    doit_smilei(load, boundary);
  }
}

DEFINE_MEMBER(void, partition)
(int Nr, std::vector<float64>& load, std::vector<int>& boundary)
{
  //
  // The sizes of `load` and `rank` should be the same and correspond to the number of chunks. The
  // size of boundary array should be larger than the number of rank by one.
  //
  // The chunk specified by `i_chunk` should be assigned to the rank specified by `i_rank` when the
  // following condition is met:
  //
  //     boundary[i_rank] <= i_chunk < boundary[i_rank+1]
  //
  // Therefore, our task is to find an appropriate boundary array first and then use it to calculate
  // `rank` array for output.
  //
  doit_smilei(load, boundary);
}

DEFINE_MEMBER(void, get_rank)(std::vector<int>& boundary, std::vector<int>& rank)
{
  const int Nr = boundary.size() - 1;

  for (int r = 0; r < Nr; r++) {
    for (int i = boundary[r]; i < boundary[r + 1]; i++) {
      rank[i] = r;
    }
  }
}

DEFINE_MEMBER(void, get_boundary)(std::vector<int>& rank, std::vector<int>& boundary)
{
  const int Nc = rank.size();
  const int Nr = boundary.size() - 1;

  for (int i = 0, r = 1; i < Nc - 1; i++) {
    if (rank[i + 1] == rank[i])
      continue;
    // found a boundary
    boundary[r] = i + 1;
    r++;
  }
  boundary[0]  = 0;
  boundary[Nr] = Nc;
}

DEFINE_MEMBER(std::vector<float64>, get_rankload)
(std::vector<int>& boundary, std::vector<float64>& load)
{
  const int Nr = boundary.size() - 1;

  std::vector<float64> rankload(Nr);
  for (int r = 0; r < Nr; r++) {
    rankload[r] = 0.0;
    for (int i = boundary[r]; i < boundary[r + 1]; i++) {
      rankload[r] += load[i];
    }
  }

  return rankload;
}

DEFINE_MEMBER(void, print_assignment)
(std::ostream& out, std::vector<int>& boundary, std::vector<float64>& load)
{
  const int Nr = boundary.size() - 1;

  std::vector<float64> rankload = get_rankload(boundary, load);
  float64              meanload = std::accumulate(load.begin(), load.end(), 0.0) / Nr;

  tfm::format(out, "*** mean load = %12.5e ***\n", meanload);
  for (int i = 0; i < Nr; i++) {
    int     numchunk  = boundary[i + 1] - boundary[i];
    float64 deviation = (rankload[i] - meanload) / meanload * 100;
    tfm::format(out, "load[%4d] = %12.5e (%4d : %+7.2f %%)\n", i, rankload[i], numchunk, deviation);
  }
}

DEFINE_MEMBER(bool, validate_boundary)(int Nc, const std::vector<int>& boundary)
{
  const int Nr = boundary.size() - 1;

  bool status = true;

  status = status & (boundary[0] == 0);

  for (int i = 1; i < Nr; i++) {
    status = status & (boundary[i + 1] >= boundary[i]);
  }

  status = status & (boundary[Nr] == Nc);

  return status;
}

DEFINE_MEMBER(void, doit_smilei)(std::vector<float64>& load, std::vector<int>& boundary)
{
  const int Nc = load.size();
  const int Nr = boundary.size() - 1;

  float64              mean_load = 0;
  std::vector<float64> cumload(Nc + 1);
  std::vector<int>     old_boundary(Nr + 1);

  // calculate cumulative load
  cumload[0] = 0;
  for (int i = 0; i < Nc; i++) {
    cumload[i + 1] = cumload[i] + load[i];
  }
  mean_load = cumload[Nc] / Nr;

  // copy original boundary
  std::copy(boundary.begin(), boundary.end(), old_boundary.begin());

  // process for each rank boundary
  for (int i = 1; i < Nr; i++) {
    float64 target  = mean_load * i;
    float64 current = cumload[boundary[i]];

    if (current > target) {
      //
      // possibly move boundary backward
      //
      int index = boundary[i] - 1;

      while (std::abs(current - target) > std::abs(current - target - load[index])) {
        current -= load[index];
        index--;
      }

      // set new boundary
      if (index >= old_boundary[i - 1]) {
        boundary[i] = index + 1;
      } else {
        boundary[i] = old_boundary[i - 1] + 1; // accommodate at least one chunk
      }
    } else {
      //
      // move boundary forward
      //
      int index = boundary[i];

      while (std::abs(current - target) > std::abs(current - target + load[index])) {
        current += load[index];
        index++;
      }

      // set new boundary
      if (index < old_boundary[i + 1]) {
        boundary[i] = index;
      } else {
        boundary[i] = old_boundary[i + 1] - 1; // accommodate at least one chunk
      }
    }
  }
}

DEFINE_MEMBER(void, doit_binary_search)(std::vector<float64>& load, std::vector<int>& boundary)
{
  const int Nc = load.size();
  const int Nr = boundary.size() - 1;

  float64              mean_load = 0;
  std::vector<float64> cumload(Nc + 1);
  std::vector<int>     old_boundary(Nr + 1);

  // calculate cumulative load
  cumload[0] = 0;
  for (int i = 0; i < Nc; i++) {
    cumload[i + 1] = cumload[i] + load[i];
  }
  mean_load = cumload[Nc] / Nr;

  boundary[0]  = 0;
  boundary[Nr] = Nc;

  for (int i = 1; i < Nr; i++) {
    auto it     = std::lower_bound(cumload.begin(), cumload.end(), mean_load * i);
    int  index  = std::distance(cumload.begin(), it);
    boundary[i] = index;
  }
}

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
