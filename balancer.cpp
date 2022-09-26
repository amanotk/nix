// -*- C++ -*-
#include "balancer.hpp"

#define DEFINE_MEMBER(type, name) type Balancer::name

NIX_NAMESPACE_BEGIN

DEFINE_MEMBER(void, assign_rank)(std::vector<int>& boundary, std::vector<int>& rank)
{
  for (int r = 0; r < boundary.size() - 1; r++) {
    for (int i = boundary[r]; i < boundary[r + 1]; i++) {
      rank[i] = r;
    }
  }
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

DEFINE_MEMBER(void, move_boundary)
(const int index, std::vector<int>& boundary, const float64 meanload,
 std::vector<float64>& rankload, std::vector<float64>& chunkload)
{
  if (boundary[index] <= 0 || boundary[index] >= chunkload.size())
    return;

  if (rankload[index - 1] < rankload[index]) {
    // try to move boundary backward
    const int index1 = index - 1;
    const int index2 = index;
    const int cindex = boundary[index] - 1;

    float64 old_load1 = rankload[index1];
    float64 old_load2 = rankload[index2];
    float64 new_load1 = old_load1 - chunkload[cindex];
    float64 new_load2 = old_load2 + chunkload[cindex];
    float64 old_diff  = std::max(std::abs(old_load1 - meanload), std::abs(old_load2 - meanload));
    float64 new_diff  = std::max(std::abs(new_load1 - meanload), std::abs(new_load2 - meanload));
    if (new_diff <= old_diff) {
      // move boundary
      rankload[index1] = new_load1;
      rankload[index2] = new_load2;
      boundary[cindex]--;
    }
  } else {
    // try to move boundary forward
    const int index1 = index - 1;
    const int index2 = index;
    const int cindex = boundary[index];

    float64 old_load1 = rankload[index1];
    float64 old_load2 = rankload[index2];
    float64 new_load1 = old_load1 + chunkload[cindex];
    float64 new_load2 = old_load2 - chunkload[cindex];
    float64 old_diff  = std::max(std::abs(old_load1 - meanload), std::abs(old_load2 - meanload));
    float64 new_diff  = std::max(std::abs(new_load1 - meanload), std::abs(new_load2 - meanload));
    if (new_diff <= old_diff) {
      // move boundary
      rankload[index1] = new_load1;
      rankload[index2] = new_load2;
      boundary[cindex]++;
    }
  }
}

DEFINE_MEMBER(void, smooth_load)
(std::vector<int>& boundary, std::vector<float64>& load, const int count)
{
  const int Nr = boundary.size() - 1;

  std::vector<float64> rankload = get_rankload(boundary, load);
  float64              meanload = std::accumulate(load.begin(), load.end(), 0.0) / Nr;

  for (int n = 0; n < count; n++) {
    // odd sweep
    for (int i = 1; i < Nr; i += 2) {
      move_boundary(i, boundary, meanload, rankload, load);
    }

    // even sweep
    for (int i = 2; i < Nr; i += 2) {
      move_boundary(i, boundary, meanload, rankload, load);
    }
  }
}

DEFINE_MEMBER(void, doit_sequential)
(std::vector<int>& boundary, std::vector<float64>& load, const int dir)
{
  const int Nc = load.size();
  const int Nr = boundary.size() - 1; // boundary size is larger than rank by 1

  boundary[0]  = 0;
  boundary[Nr] = Nc;

  if (dir == +1) { // forward sequential assignment
    int     rank      = 0;
    float64 cur_load  = load[0];
    float64 prv_load  = 0;
    float64 mean_load = std::accumulate(load.begin(), load.end(), 0.0) / Nr;

    for (int i = 0; i < Nc - 1; i++) {
      if (rank >= Nr - 1)
        continue;

      prv_load = cur_load;
      cur_load += load[i + 1];

      if (cur_load >= mean_load) {
        rank++;
        if (std::abs(cur_load - mean_load) < std::abs(prv_load - mean_load)) {
          boundary[rank] = i + 2;
          cur_load       = 0.0;
        } else {
          boundary[rank] = i + 1;
          cur_load       = load[i + 1];
        }
      }
    }

    return;
  }

  if (dir == -1) { // backward sequential assignment
    int     rank     = Nr;
    float64 cur_load = load[Nc - 1];
    float64 prv_load = 0;
    float64 meanload = std::accumulate(load.begin(), load.end(), 0.0) / Nr;

    for (int i = Nc - 1; i > 0; i--) {
      if (rank <= 1)
        continue;

      prv_load = cur_load;
      cur_load += load[i - 1];

      if (cur_load >= meanload) {
        rank--;
        if (std::abs(cur_load - meanload) < std::abs(prv_load - meanload)) {
          boundary[rank] = i - 1;
          cur_load       = 0.0;
        } else {
          boundary[rank] = i;
          cur_load       = load[i - 1];
        }
      }
    }

    return;
  }

  // error
  ERRORPRINT("dir argument should be either +1 or -1\n");
}

DEFINE_MEMBER(void, print_load_summary)
(std::ostream& out, std::vector<int>& boundary, std::vector<float64>& load)
{
  const int Nr = boundary.size() - 1;

  std::vector<float64> rankload = get_rankload(boundary, load);
  float64              meanload = std::accumulate(load.begin(), load.end(), 0.0) / Nr;

  tfm::format(out, "*** mean load = %10.5f\n", meanload);
  for (int i = 0; i < Nr; i++) {
    int     numchunk  = boundary[i + 1] - boundary[i];
    float64 deviation = (rankload[i] - meanload) / meanload * 100;
    tfm::format(out, "load[%4d] = %10.5f (%4d : %+5.2f %%)\n", i, rankload[i], numchunk, deviation);
  }

  for (int i = 0; i < Nr + 1; i++) {
    tfm::format(out, "boundary[%4d] = %8d\n", i, boundary[i]);
  }
}

DEFINE_MEMBER(bool, validate_boundary)(const int Nc, const std::vector<int>& boundary)
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

DEFINE_MEMBER(void, partition)
(const int Nr, std::vector<float64>& load, std::vector<int>& rank)
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
  const int Nc = load.size();

  std::vector<int> boundary1(Nr + 1);
  std::vector<int> boundary2(Nr + 1);

  // forward assignment followed by smoothing
  doit_sequential(boundary1, load, +1);
  smooth_load(boundary1, load, 5);

  // backward assignment followed by smoothing
  doit_sequential(boundary2, load, -1);
  smooth_load(boundary2, load, 5);

  // adopt better one
  std::vector<float64> rankload1 = get_rankload(boundary1, load);
  std::vector<float64> rankload2 = get_rankload(boundary2, load);

  float64 max1 = *std::max_element(rankload1.begin(), rankload1.end());
  float64 max2 = *std::max_element(rankload2.begin(), rankload2.end());

  if (max1 <= max2) {
    // adopt boundary1
    assign_rank(boundary1, rank);
  } else {
    // adopt boundary2
    assign_rank(boundary2, rank);
  }
}

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
