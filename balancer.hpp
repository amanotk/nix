// -*- C++ -*-
#ifndef _BALANCER_HPP_
#define _BALANCER_HPP_

#include "buffer.hpp"
#include "nix.hpp"

NIX_NAMESPACE_BEGIN

///
/// @brief Load Balancer
///
class Balancer
{
protected:
  int                  nchunk;    ///< number of chunks
  std::vector<float64> chunkload; ///< array of chunk load

  static int find_rank(int id, std::vector<int>& boundary)
  {
    auto it = std::upper_bound(boundary.begin(), boundary.end(), id);
    return std::distance(boundary.begin(), it) - 1;
  }

  ///
  /// @brief perform assignment of chunks according to SMILEI code (Derouillat et al. 2018)
  ///
  /// @param[in] load load of chunks
  /// @param[out] boundary boundary between ranks
  /// @return true if the assignment is changed and false otherwise
  ///
  bool assign_smilei(std::vector<float64>& load, std::vector<int>& boundary);

  ///
  /// @brief perform assignment of chunks via binary search for the best
  ///
  /// @param[in] load load of chunks
  /// @param[out] boundary boundary between ranks
  /// @return true if the assignment is successful and false otherwise
  ///
  bool assign_binarysearch(std::vector<float64>& load, std::vector<int>& boundary);

public:
  // default constructor
  Balancer() : Balancer(0)
  {
  }

  // constructor with size
  Balancer(int n) : nchunk(n), chunkload(n, 0.0)
  {
  }

  // accessor to chunkload
  double& load(int i)
  {
    return chunkload[i];
  }

  // const accessor to chunkload
  const double& load(int i) const
  {
    return chunkload[i];
  }

  // fill chunkload with given value
  void fill_load(float64 value)
  {
    std::fill(chunkload.begin(), chunkload.end(), value);
  }

  ///
  /// @brief assignment for the first time (without given boundary)
  /// @return array of boundary as a result of assignment
  ///
  virtual std::vector<int> assign_initial(int nprocess);

  ///
  /// @brief assignment with current assignment boundary
  /// @param[in] boundary array of current boundary
  /// @return array of boundary as a result of assignment
  ///
  virtual std::vector<int> assign(std::vector<int>& boundary);

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
  ///
  void print_assignment(std::ostream& out, std::vector<int>& boundary);

  ///
  /// @brief check if the boundary is in ascending order
  ///
  /// @param[in] boundary array of boundary
  /// @return true if the boundary is in ascending order
  ///
  bool is_boundary_ascending(const std::vector<int>& boundary);

  ///
  /// @brief check if the boundary is optimum
  ///
  /// @param[in] boundary array of boundary
  /// @return true if the boundary is optimum
  ///
  bool is_boundary_optimum(const std::vector<int>& boundary);

  ///
  /// @brief update global chunk load
  ///
  /// @tparam Data Application internal data struct
  ///
  template <typename Data>
  void update_global_load(Data&& data);

  ///
  /// @brief send/recv chunks for load balancing
  ///
  /// @tparam App Application class
  /// @tparam Data Application internal data struct
  /// @param app Application object
  /// @param data Application internal data object
  /// @param boundary array of boundary
  ///
  template <typename App, typename Data>
  void sendrecv_chunk(App&& app, Data&& data, std::vector<int>& boundary);
};

///
/// implementation for template methods follows
///

template <typename Data>
void Balancer::update_global_load(Data&& data)
{
  // clear
  fill_load(0.0);

  // calculate local workload
  {
    for (int i = 0; i < data.chunkvec.size(); i++) {
      int id = data.chunkvec[i]->get_id();

      chunkload[id] = data.chunkvec[i]->get_total_load();
    }
  }

  // synchronize globally
  {
    const int thisrank = data.thisrank;
    const int nprocess = data.nprocess;

    std::vector<int> rcnt(nprocess);
    std::vector<int> disp(nprocess);

    // recv count
    std::fill(rcnt.begin(), rcnt.end(), 0);
    for (int i = 0; i < chunkload.size(); i++) {
      int rank = data.chunkmap->get_rank(i);
      rcnt[rank]++;
    }

    // displacement
    disp[0] = 0;
    for (int r = 0; r < nprocess - 1; r++) {
      disp[r + 1] = disp[r] + rcnt[r];
    }

    MPI_Allgatherv(MPI_IN_PLACE, rcnt[thisrank], MPI_FLOAT64_T, chunkload.data(), rcnt.data(),
                   disp.data(), MPI_FLOAT64_T, MPI_COMM_WORLD);
  }
}

template <typename App, typename Data>
void Balancer::sendrecv_chunk(App&& app, Data&& data, std::vector<int>& boundary)
{
  const int nchunk_global = boundary.back();

  int thisrank = data.thisrank;
  int nprocess = data.nprocess;
  int rankmin  = 0;
  int rankmax  = nprocess - 1;
  int rank_l   = thisrank > rankmin ? thisrank - 1 : MPI_PROC_NULL;
  int rank_r   = thisrank < rankmax ? thisrank + 1 : MPI_PROC_NULL;

  // chunk dimensions
  int dims[3];
  dims[0] = data.ndims[0] / data.cdims[0];
  dims[1] = data.ndims[1] / data.cdims[1];
  dims[2] = data.ndims[2] / data.cdims[2];

  Buffer sendbuf;
  Buffer recvbuf;

  //
  // check buffer size and reallocate if necessary
  //
  {
    int sendsize   = 0;
    int sendsize_l = 0;
    int sendsize_r = 0;
    int recvsize   = 0;
    int recvsize_l = 0;
    int recvsize_r = 0;

    for (int i = 0; i < data.chunkvec.size(); i++) {
      int id   = data.chunkvec[i]->get_id();
      int rank = find_rank(id, boundary);

      if (rank == thisrank - 1) {
        sendsize_l = std::max(sendsize_l, data.chunkvec[i]->pack(nullptr, 0));
      }
      if (rank == thisrank + 1) {
        sendsize_r = std::max(sendsize_r, data.chunkvec[i]->pack(nullptr, 0));
      }
    }

    // get maximum possible chunk size
    {
      MPI_Request request[4];

      MPI_Isend(&sendsize_l, sizeof(int), MPI_BYTE, rank_l, 1, MPI_COMM_WORLD, &request[0]);
      MPI_Isend(&sendsize_r, sizeof(int), MPI_BYTE, rank_r, 2, MPI_COMM_WORLD, &request[1]);
      MPI_Irecv(&recvsize_l, sizeof(int), MPI_BYTE, rank_l, 2, MPI_COMM_WORLD, &request[2]);
      MPI_Irecv(&recvsize_r, sizeof(int), MPI_BYTE, rank_r, 1, MPI_COMM_WORLD, &request[3]);

      MPI_Waitall(4, request, MPI_STATUSES_IGNORE);

      sendsize = std::max(sendsize_l, sendsize_r);
      recvsize = std::max(recvsize_l, recvsize_r);
    }

    // allocate buffer
    sendbuf.resize(sendsize);
    recvbuf.resize(recvsize);
  }

  //
  // function for sending chunk
  //
  auto send_chunk = [&](int chunkid, int rank, int tag, int dir, int pos) {
    MPI_Request request;
    int         size;
    uint8_t*    buf = sendbuf.get(pos);

    if (chunkid < 0 || chunkid >= nchunk_global) {
      return;
    }

    auto it    = std::find_if(data.chunkvec.begin(), data.chunkvec.end(),
                              [&](auto& p) { return p->get_id() == chunkid; });
    int  index = std::distance(data.chunkvec.begin(), it);

    while (find_rank(chunkid, boundary) == rank) {
      // pack
      size = data.chunkvec[index]->pack(buf, 0);
      data.chunkvec[index].reset(); // deallocate memory

      MPI_Isend(buf, size, MPI_BYTE, rank, tag, MPI_COMM_WORLD, &request);
      MPI_Wait(&request, MPI_STATUS_IGNORE);

      chunkid += dir;
      index += dir;
    }
  };

  //
  // function for receiving chunk
  //
  auto recv_chunk = [&](int chunkid, int rank, int tag, int dir, int pos) {
    MPI_Request request;
    int         size = recvbuf.size - pos;
    uint8_t*    buf  = recvbuf.get(pos);

    if (chunkid < 0 || chunkid >= nchunk_global) {
      return;
    }

    while (find_rank(chunkid, boundary) == thisrank) {
      MPI_Irecv(buf, size, MPI_BYTE, rank, tag, MPI_COMM_WORLD, &request);
      MPI_Wait(&request, MPI_STATUS_IGNORE);

      // unpack
      auto p = app.create_chunk(dims, 0);
      p->unpack(buf, 0);
      data.chunkvec.push_back(std::move(p));

      chunkid += dir;
    }
  };

  //
  // chunk exchange at odd boundary
  //
  if (thisrank % 2 == 1) {
    int chunkid_min = (*data.chunkvec.begin())->get_id();

    // send to left
    {
      int chunkid = chunkid_min;
      send_chunk(chunkid, rank_l, 1, +1, 0);
    }
    // recv from left
    {
      int chunkid = chunkid_min - 1;
      recv_chunk(chunkid, rank_l, 2, -1, 0);
    }
  } else {
    int chunkid_max = (*data.chunkvec.rbegin())->get_id();

    // send to right
    {
      int chunkid = chunkid_max;
      send_chunk(chunkid, rank_r, 2, -1, 0);
    }
    // recv from right
    {
      int chunkid = chunkid_max + 1;
      recv_chunk(chunkid, rank_r, 1, +1, 0);
    }
  }

  data.chunkvec.sort_and_shrink();

  //
  // chunk exchange at even boundary
  //
  if (thisrank % 2 == 1) {
    int chunkid_max = (*data.chunkvec.rbegin())->get_id();

    // send to right
    {
      int chunkid = chunkid_max;
      send_chunk(chunkid, rank_r, 3, -1, 0);
    }
    // recv from right
    {
      int chunkid = chunkid_max + 1;
      recv_chunk(chunkid, rank_r, 4, +1, 0);
    }
  } else {
    int chunkid_min = (*data.chunkvec.begin())->get_id();

    // send to left
    {
      int chunkid = chunkid_min;
      send_chunk(chunkid, rank_l, 4, +1, 0);
    }
    // recv from left
    {
      int chunkid = chunkid_min - 1;
      recv_chunk(chunkid, rank_l, 3, -1, 0);
    }
  }

  data.chunkvec.sort_and_shrink();
}

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
