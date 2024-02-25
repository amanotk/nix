// -*- C++ -*-
#ifndef _XTENSOR_HALO3D_HPP_
#define _XTENSOR_HALO3D_HPP_

#include "halo3d.hpp"
#include "nix.hpp"
#include "xtensor_particle.hpp"
#include "xtensorall.hpp"

NIX_NAMESPACE_BEGIN

///
/// @brief Boundary Halo3D class for field
///
template <typename Chunk>
class XtensorHaloField3D : public Halo3D<xt::xtensor<float64, 4>, Chunk>
{
public:
  using Base = Halo3D<xt::xtensor<float64, 4>, Chunk>;
  using Base::Base; // constructor
  using Base::data;
  using Base::chunk;
  using Base::send_buffer;
  using Base::recv_buffer;
  using Base::send_count;
  using Base::recv_count;

  template <typename BufferPtr>
  bool pack(BufferPtr& mpibuf, int iz, int iy, int ix, int send_bound[3][2], int recv_bound[3][2])
  {
    // skip
    if (iz == 1 && iy == 1 && ix == 1)
      return false;

    // packing
    auto Iz   = xt::range(send_bound[0][0], send_bound[0][1] + 1);
    auto Iy   = xt::range(send_bound[1][0], send_bound[1][1] + 1);
    auto Ix   = xt::range(send_bound[2][0], send_bound[2][1] + 1);
    auto view = xt::strided_view(*data, {Iz, Iy, Ix, xt::ellipsis()});

    float64* ptr = static_cast<float64*>(mpibuf->get_send_buffer(iz, iy, ix));
    std::copy(view.begin(), view.end(), ptr);

    // datatype
    mpibuf->sendtype(iz, iy, ix) = MPI_FLOAT64_T;
    mpibuf->recvtype(iz, iy, ix) = MPI_FLOAT64_T;

    // parameters for MPI send/recv
    send_buffer = mpibuf->get_send_buffer(iz, iy, ix);
    recv_buffer = mpibuf->get_recv_buffer(iz, iy, ix);
    send_count  = view.size();
    recv_count  = view.size();

    return true;
  }

  template <typename BufferPtr>
  bool unpack(BufferPtr& mpibuf, int iz, int iy, int ix, int send_bound[3][2], int recv_bound[3][2])
  {
    // skip
    if (iz == 1 && iy == 1 && ix == 1)
      return false;

    if (chunk->get_nb_rank(iz - 1, iy - 1, ix - 1) == MPI_PROC_NULL)
      return false;

    // unpacking
    auto Iz   = xt::range(recv_bound[0][0], recv_bound[0][1] + 1);
    auto Iy   = xt::range(recv_bound[1][0], recv_bound[1][1] + 1);
    auto Ix   = xt::range(recv_bound[2][0], recv_bound[2][1] + 1);
    auto view = xt::strided_view(*data, {Iz, Iy, Ix, xt::ellipsis()});

    float64* ptr = static_cast<float64*>(mpibuf->get_recv_buffer(iz, iy, ix));
    std::copy(ptr, ptr + view.size(), view.begin());

    return true;
  }
};

///
/// @brief Boundary Halo3D class for current
///
template <typename Chunk>
class XtensorHaloCurrent3D : public Halo3D<xt::xtensor<float64, 4>, Chunk>
{
public:
  using Base = Halo3D<xt::xtensor<float64, 4>, Chunk>;
  using Base::Base; // constructor
  using Base::data;
  using Base::chunk;
  using Base::send_buffer;
  using Base::recv_buffer;
  using Base::send_count;
  using Base::recv_count;

  template <typename BufferPtr>
  bool pack(BufferPtr& mpibuf, int iz, int iy, int ix, int send_bound[3][2], int recv_bound[3][2])
  {
    // skip
    if (iz == 1 && iy == 1 && ix == 1)
      return false;

    // packing
    auto Iz   = xt::range(recv_bound[0][0], recv_bound[0][1] + 1);
    auto Iy   = xt::range(recv_bound[1][0], recv_bound[1][1] + 1);
    auto Ix   = xt::range(recv_bound[2][0], recv_bound[2][1] + 1);
    auto view = xt::strided_view(*data, {Iz, Iy, Ix, xt::ellipsis()});

    float64* ptr = static_cast<float64*>(mpibuf->get_send_buffer(iz, iy, ix));
    std::copy(view.begin(), view.end(), ptr);

    // datatype
    mpibuf->sendtype(iz, iy, ix) = MPI_FLOAT64_T;
    mpibuf->recvtype(iz, iy, ix) = MPI_FLOAT64_T;

    // parameters for MPI send/recv
    send_buffer = mpibuf->get_send_buffer(iz, iy, ix);
    recv_buffer = mpibuf->get_recv_buffer(iz, iy, ix);
    send_count  = view.size();
    recv_count  = view.size();

    return true;
  }

  template <typename BufferPtr>
  bool unpack(BufferPtr& mpibuf, int iz, int iy, int ix, int send_bound[3][2], int recv_bound[3][2])
  {
    // skip
    if (iz == 1 && iy == 1 && ix == 1)
      return false;

    if (chunk->get_nb_rank(iz - 1, iy - 1, ix - 1) == MPI_PROC_NULL)
      return false;

    // unpacking
    auto Iz   = xt::range(send_bound[0][0], send_bound[0][1] + 1);
    auto Iy   = xt::range(send_bound[1][0], send_bound[1][1] + 1);
    auto Ix   = xt::range(send_bound[2][0], send_bound[2][1] + 1);
    auto view = xt::strided_view(*data, {Iz, Iy, Ix, xt::ellipsis()});

    float64* ptr = static_cast<float64*>(mpibuf->get_recv_buffer(iz, iy, ix));
    std::transform(ptr, ptr + view.size(), view.begin(), view.begin(), std::plus<float64>());

    return true;
  }
};

///
/// @brief Boundary Halo3D class for moment
///
template <typename Chunk>
class XtensorHaloMoment3D : public Halo3D<xt::xtensor<float64, 5>, Chunk>
{
public:
  using Base = Halo3D<xt::xtensor<float64, 5>, Chunk>;
  using Base::Base; // constructor
  using Base::data;
  using Base::chunk;
  using Base::send_buffer;
  using Base::recv_buffer;
  using Base::send_count;
  using Base::recv_count;

  template <typename BufferPtr>
  bool pack(BufferPtr& mpibuf, int iz, int iy, int ix, int send_bound[3][2], int recv_bound[3][2])
  {
    // skip
    if (iz == 1 && iy == 1 && ix == 1)
      return false;

    // packing
    auto Iz   = xt::range(recv_bound[0][0], recv_bound[0][1] + 1);
    auto Iy   = xt::range(recv_bound[1][0], recv_bound[1][1] + 1);
    auto Ix   = xt::range(recv_bound[2][0], recv_bound[2][1] + 1);
    auto view = xt::strided_view(*data, {Iz, Iy, Ix, xt::ellipsis()});

    float64* ptr = static_cast<float64*>(mpibuf->get_send_buffer(iz, iy, ix));
    std::copy(view.begin(), view.end(), ptr);

    // datatype
    mpibuf->sendtype(iz, iy, ix) = MPI_FLOAT64_T;
    mpibuf->recvtype(iz, iy, ix) = MPI_FLOAT64_T;

    // parameters for MPI send/recv
    send_buffer = mpibuf->get_send_buffer(iz, iy, ix);
    recv_buffer = mpibuf->get_recv_buffer(iz, iy, ix);
    send_count  = view.size();
    recv_count  = view.size();

    return true;
  }

  template <typename BufferPtr>
  bool unpack(BufferPtr& mpibuf, int iz, int iy, int ix, int send_bound[3][2], int recv_bound[3][2])
  {
    // skip
    if (iz == 1 && iy == 1 && ix == 1)
      return false;

    if (chunk->get_nb_rank(iz - 1, iy - 1, ix - 1) == MPI_PROC_NULL)
      return false;

    // unpacking
    auto Iz   = xt::range(send_bound[0][0], send_bound[0][1] + 1);
    auto Iy   = xt::range(send_bound[1][0], send_bound[1][1] + 1);
    auto Ix   = xt::range(send_bound[2][0], send_bound[2][1] + 1);
    auto view = xt::strided_view(*data, {Iz, Iy, Ix, xt::ellipsis()});

    float64* ptr = static_cast<float64*>(mpibuf->get_recv_buffer(iz, iy, ix));
    std::transform(ptr, ptr + view.size(), view.begin(), view.begin(), std::plus<float64>());

    return true;
  }
};

///
/// @brief Boundary Halo3D class for particle
///
template <typename Chunk>
class XtensorHaloParticle3D : public Halo3D<ParticleVec, Chunk>
{
public:
  using Base = Halo3D<ParticleVec, Chunk>;
  using Base::data;
  using Base::chunk;
  using Base::send_buffer;
  using Base::recv_buffer;
  using Base::send_count;
  using Base::recv_count;

  static constexpr int32_t head_byte         = sizeof(int32_t);
  static constexpr int32_t elem_byte         = ParticlePtr::element_type::get_particle_size();
  static constexpr float64 increase_fraction = 0.80;
  static constexpr float64 decrease_fraction = 0.20;

  int32_t                 Ns;
  int32_t                 buffer_flag[3][3][3];
  ParticleVec             particle;
  xt::xtensor<int32_t, 4> snd_count;
  std::vector<int32_t>    num_particle;

  XtensorHaloParticle3D(ParticleVec& data, Chunk& chunk) : Halo3D<ParticleVec, Chunk>(data, chunk)
  {
    Ns       = data.size();
    particle = data;

    snd_count.resize({static_cast<size_t>(Ns + 1), 3ul, 3ul, 3ul});
    num_particle.resize(Ns);
  }

  template <typename BufferPtr>
  void pre_pack(BufferPtr& mpibuf)
  {
    const float64 xmin = chunk->get_xmin();
    const float64 xmax = chunk->get_xmax();
    const float64 ymin = chunk->get_ymin();
    const float64 ymax = chunk->get_ymax();
    const float64 zmin = chunk->get_zmin();
    const float64 zmax = chunk->get_zmax();

    bool status = true;

    // initialize with zero
    snd_count.fill(0);

    //
    // pack out-of-bounds particles
    //
    for (int is = 0; is < Ns; is++) {
      // loop over particles
      auto& xu = particle[is]->xu;
      for (int ip = 0; ip < particle[is]->Np; ip++) {
        int dirz = (xu(ip, 2) >= zmax) - (xu(ip, 2) < zmin);
        int diry = (xu(ip, 1) >= ymax) - (xu(ip, 1) < ymin);
        int dirx = (xu(ip, 0) >= xmax) - (xu(ip, 0) < xmin);

        // skip
        if (dirx == 0 && diry == 0 && dirz == 0)
          continue;

        int iz  = dirz + 1;
        int iy  = diry + 1;
        int ix  = dirx + 1;
        int cnt = elem_byte * snd_count(Ns, iz, iy, ix) + head_byte * (is + 1);
        int pos = mpibuf->bufaddr(iz, iy, ix) + cnt;

        // check buffer size
        if (mpibuf->bufsize(iz, iy, ix) < cnt) {
          status = false;
        }

        // pack
        std::memcpy(mpibuf->sendbuf.get(pos), &xu(ip, 0), elem_byte);
        snd_count(is, iz, iy, ix)++;
        snd_count(Ns, iz, iy, ix)++; // total number of send particles
      }
    }

    if (status == false) {
#pragma omp critical
      {
        ERROR << tfm::format("Chunk[%06d]: insufficient send buffer", chunk->get_id());
        MPI_Abort(MPI_COMM_WORLD, -1);
      }
    }
  }

  template <typename BufferPtr>
  bool pack(BufferPtr& mpibuf, int iz, int iy, int ix, int send_bound[3][2], int recv_bound[3][2])
  {
    // skip
    if (iz == 1 && iy == 1 && ix == 1)
      return false;

    // add header for each species
    int addr = mpibuf->bufaddr(iz, iy, ix);
    int byte = 0;
    for (int is = 0; is < Ns; is++) {
      std::memcpy(mpibuf->sendbuf.get(addr + byte), &snd_count(is, iz, iy, ix), head_byte);
      byte += head_byte + elem_byte * snd_count(is, iz, iy, ix);
    }

    // datatype
    mpibuf->sendtype(iz, iy, ix) = MPI_BYTE;
    mpibuf->recvtype(iz, iy, ix) = MPI_BYTE;

    // parameters for MPI send/recv
    send_buffer = mpibuf->get_send_buffer(iz, iy, ix);
    recv_buffer = mpibuf->get_recv_buffer(iz, iy, ix);
    send_count  = byte;
    recv_count  = mpibuf->bufsize(iz, iy, ix);

    return true;
  }

  template <typename BufferPtr>
  void post_pack(BufferPtr& mpibuf)
  {
    // do nothing
  }

  template <typename BufferPtr>
  void pre_unpack(BufferPtr& mpibuf)
  {
    for (int iz = 0; iz < 3; iz++) {
      for (int iy = 0; iy < 3; iy++) {
        for (int ix = 0; ix < 3; ix++) {
          buffer_flag[iz][iy][ix] = 0;
        }
      }
    }

    for (int is = 0; is < Ns; is++) {
      num_particle[is] = particle[is]->Np;
    }
  }

  template <typename BufferPtr>
  bool unpack(BufferPtr& mpibuf, int iz, int iy, int ix, int send_bound[3][2], int recv_bound[3][2])
  {
    // skip
    if (iz == 1 && iy == 1 && ix == 1)
      return false;

    if (chunk->get_nb_rank(iz - 1, iy - 1, ix - 1) == MPI_PROC_NULL)
      return false;

    int      recvcnt = 0;
    uint8_t* recvptr = mpibuf->recvbuf.get(mpibuf->bufaddr(iz, iy, ix));

    // copy to the end of particle array
    for (int is = 0; is < Ns; is++) {
      // header
      int cnt;
      std::memcpy(&cnt, recvptr, head_byte);
      recvptr += head_byte;

      if (num_particle[is] + cnt > particle[is]->Np_total) {
        // run out of particle buffer and try to reallocate twice the original
        particle[is]->resize(2 * particle[is]->Np_total);
      }

      // particles
      float64* ptcl = &particle[is]->xu(num_particle[is], 0);
      std::memcpy(ptcl, recvptr, elem_byte * cnt);
      recvptr += cnt * elem_byte;
      recvcnt += cnt;

      // increment number of particles
      num_particle[is] += cnt;
    }

    // check received data size
    int recvsize = Ns * head_byte + recvcnt * elem_byte;

    if (recvsize > mpibuf->bufsize(iz, iy, ix)) {
#pragma omp critical
      {
        ERROR << tfm::format("Chunk[%06d]: insufficient recv buffer", chunk->get_id());
        MPI_Abort(MPI_COMM_WORLD, -1);
      }
    } else if (recvsize > increase_fraction * mpibuf->bufsize(iz, iy, ix)) {
      buffer_flag[iz][iy][ix] = +1;
    } else if (recvsize < decrease_fraction * mpibuf->bufsize(iz, iy, ix)) {
      buffer_flag[iz][iy][ix] = -1;
    }

    return true;
  }

  template <typename BufferPtr>
  void post_unpack(BufferPtr& mpibuf)
  {
    //
    // set boundary condition and append count for received particles
    //
    for (int is = 0; is < Ns; is++) {
      chunk->set_boundary_particle_after_sendrecv(particle[is], particle[is]->Np,
                                                  num_particle[is] - 1, is);
      chunk->count_particle(particle[is], particle[is]->Np, num_particle[is] - 1, false);
      // now update number of particles
      particle[is]->Np = num_particle[is];
    }

    //
    // sort (or rearrange) particle array
    //
    for (int is = 0; is < Ns; is++) {
      particle[is]->sort();
    }

    //
    // automatically resize particle buffer
    //
    for (int is = 0; is < Ns; is++) {
      int new_np = particle[is]->Np_total;

      // increase particle buffer
      if (particle[is]->Np > increase_fraction * particle[is]->Np_total) {
        new_np = 2.0 * particle[is]->Np_total;
      }

      // decrease particle buffer
      if (particle[is]->Np < decrease_fraction * particle[is]->Np_total) {
        new_np = 0.5 * particle[is]->Np_total;
      }

      // resize if needed
      particle[is]->resize(new_np);
    }

    //
    // resize MPI buffer if required
    //
    {
      bool increase = false;
      bool decrease = true;

      for (int iz = 0; iz < 3; iz++) {
        for (int iy = 0; iy < 3; iy++) {
          for (int ix = 0; ix < 3; ix++) {
            // if increase flag is true in any directions
            increase = increase | (buffer_flag[iz][iy][ix] == +1);
            // if decrease flag is true in all directions
            decrease = decrease & (buffer_flag[iz][iy][ix] == -1);
          }
        }
      }

      // resize
      if (increase == true || decrease == true) {
        int Nppc = 0;
        for (int is = 0; is < Ns; is++) {
          Nppc += particle[is]->Np / particle[is]->Ng + 1;
        }

        chunk->set_mpi_buffer(mpibuf, 0, Ns * head_byte, Nppc * elem_byte);
      }
    }
  }
};

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
