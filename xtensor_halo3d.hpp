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
class XtensorHaloField3D : public Halo3D<xt::xtensor<float64, 4>, Chunk, true>
{
public:
  using Base = Halo3D<xt::xtensor<float64, 4>, Chunk, true>;
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
class XtensorHaloCurrent3D : public Halo3D<xt::xtensor<float64, 4>, Chunk, true>
{
public:
  using Base = Halo3D<xt::xtensor<float64, 4>, Chunk, true>;
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
class XtensorHaloMoment3D : public Halo3D<xt::xtensor<float64, 5>, Chunk, true>
{
public:
  using Base = Halo3D<xt::xtensor<float64, 5>, Chunk, true>;
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
class XtensorHaloParticle3D : public Halo3D<ParticleVec, Chunk, false>
{
public:
  using Base = Halo3D<ParticleVec, Chunk, false>;
  using Base::data;
  using Base::chunk;
  using Base::send_buffer;
  using Base::recv_buffer;
  using Base::send_count;
  using Base::recv_count;

  static constexpr int32_t head_byte = sizeof(int32_t);
  static constexpr int32_t elem_byte = ParticlePtr::element_type::get_particle_size();

  int32_t              Ns;
  ParticleVec          particle;
  std::vector<int32_t> num_unpacked;

  XtensorHaloParticle3D(ParticleVec& data, Chunk& chunk)
      : Halo3D<ParticleVec, Chunk, false>(data, chunk)
  {
    Ns       = data.size();
    particle = data;
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

    std::array<size_t, 4>   shape = {static_cast<size_t>(Ns + 1), 3ul, 3ul, 3ul};
    xt::xtensor<int32_t, 4> send_count(shape);

    // initialize
    send_count.fill(0);

    //
    // count out-of-bounds particles
    //
    for (int is = 0; is < Ns; is++) {
      auto& xu = particle[is]->xu;
      for (int ip = 0; ip < particle[is]->Np; ip++) {
        int iz = (xu(ip, 2) >= zmax) - (xu(ip, 2) < zmin) + 1;
        int iy = (xu(ip, 1) >= ymax) - (xu(ip, 1) < ymin) + 1;
        int ix = (xu(ip, 0) >= xmax) - (xu(ip, 0) < xmin) + 1;

        // skip
        if (ix == 1 && iy == 1 && iz == 1)
          continue;

        send_count(is, iz, iy, ix)++;
        send_count(Ns, iz, iy, ix)++; // total number of send particles
      }
    }

    //
    // allocate buffer
    //
    {
      int bufsize = 0;

      mpibuf->bufsize.fill(0);
      mpibuf->bufaddr.fill(0);

      for (int iz = 0; iz < 3; iz++) {
        for (int iy = 0; iy < 3; iy++) {
          for (int ix = 0; ix < 3; ix++) {
            // skip
            if (iz == 1 && iy == 1 && ix == 1)
              continue;

            mpibuf->bufsize(iz, iy, ix) = elem_byte * send_count(Ns, iz, iy, ix) + head_byte * Ns;
            mpibuf->bufaddr(iz, iy, ix) = bufsize;
            bufsize += mpibuf->bufsize(iz, iy, ix);
          }
        }
      }

      mpibuf->sendbuf.resize(bufsize);
    }

    //
    // pack header
    //
    {
      for (int iz = 0; iz < 3; iz++) {
        for (int iy = 0; iy < 3; iy++) {
          for (int ix = 0; ix < 3; ix++) {
            // skip
            if (iz == 1 && iy == 1 && ix == 1)
              continue;

            int addr = mpibuf->bufaddr(iz, iy, ix);
            for (int is = 0; is < Ns; is++) {
              std::memcpy(mpibuf->sendbuf.get(addr), &send_count(is, iz, iy, ix), head_byte);
              addr += head_byte + elem_byte * send_count(is, iz, iy, ix);
            }
          }
        }
      }
    }

    //
    // pack out-of-bounds particles
    //
    {
      auto addr = mpibuf->bufaddr;

      for (int is = 0; is < Ns; is++) {
        // skip header
        for (int iz = 0; iz < 3; iz++) {
          for (int iy = 0; iy < 3; iy++) {
            for (int ix = 0; ix < 3; ix++) {
              addr(iz, iy, ix) += head_byte;
            }
          }
        }

        // pack particles
        auto& xu = particle[is]->xu;
        for (int ip = 0; ip < particle[is]->Np; ip++) {
          int iz = (xu(ip, 2) >= zmax) - (xu(ip, 2) < zmin) + 1;
          int iy = (xu(ip, 1) >= ymax) - (xu(ip, 1) < ymin) + 1;
          int ix = (xu(ip, 0) >= xmax) - (xu(ip, 0) < xmin) + 1;

          // skip
          if (ix == 1 && iy == 1 && iz == 1)
            continue;

          // pack
          std::memcpy(mpibuf->sendbuf.get(addr(iz, iy, ix)), &xu(ip, 0), elem_byte);
          addr(iz, iy, ix) += elem_byte;
          send_count(is, iz, iy, ix)--;
        }
      }
    }

    //
    // check if all particles are packed
    //
    {
      bool is_all_packed = true;

      for (int is = 0; is < Ns; is++) {
        for (int iz = 0; iz < 3; iz++) {
          for (int iy = 0; iy < 3; iy++) {
            for (int ix = 0; ix < 3; ix++) {
              is_all_packed = is_all_packed && (send_count(is, iz, iy, ix) == 0);
            }
          }
        }
      }

      if (is_all_packed == false) {
        ERROR << tfm::format("Some particles are not properly packed!");
      }
    }
  }

  template <typename BufferPtr>
  bool pack(BufferPtr& mpibuf, int iz, int iy, int ix, int send_bound[3][2], int recv_bound[3][2])
  {
    // skip
    if (iz == 1 && iy == 1 && ix == 1)
      return false;

    // datatype
    mpibuf->sendtype(iz, iy, ix) = MPI_BYTE;
    mpibuf->recvtype(iz, iy, ix) = MPI_BYTE;

    // only for send buffer (recv is not yet ready)
    send_buffer = mpibuf->get_send_buffer(iz, iy, ix);
    recv_buffer = nullptr;
    send_count  = mpibuf->bufsize(iz, iy, ix);
    recv_count  = 0;

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
    std::array<size_t, 4>   shape = {static_cast<size_t>(Ns + 1), 3ul, 3ul, 3ul};
    xt::xtensor<int32_t, 4> recv_count(shape);

    // initialize
    recv_count.fill(0);

    //
    // unpack header
    //
    for (int iz = 0; iz < 3; iz++) {
      for (int iy = 0; iy < 3; iy++) {
        for (int ix = 0; ix < 3; ix++) {
          // skip null message
          if (mpibuf->bufsize(iz, iy, ix) == 0)
            continue;

          int addr = mpibuf->bufaddr(iz, iy, ix);
          for (int is = 0; is < Ns; is++) {
            std::memcpy(&recv_count(is, iz, iy, ix), mpibuf->recvbuf.get(addr), head_byte);
            addr += head_byte + elem_byte * recv_count(is, iz, iy, ix);
            recv_count(Ns, iz, iy, ix) += recv_count(is, iz, iy, ix);
          }
        }
      }
    }

    //
    // resize particle buffer if needed
    //
    {
      const float64 ratio     = chunk->get_buffer_ratio();
      const float64 increased = 1 + ratio;
      const float64 decreased = 1 - ratio;

      for (int is = 0; is < Ns; is++) {
        int np_next = particle[is]->Np;
        for (int iz = 0; iz < 3; iz++) {
          for (int iy = 0; iy < 3; iy++) {
            for (int ix = 0; ix < 3; ix++) {
              np_next += recv_count(is, iz, iy, ix);
            }
          }
        }

        // expand
        if (np_next > particle[is]->Np_total) {
          particle[is]->resize(increased * np_next);
        }

        // shrink
        if (np_next < particle[is]->Np_total * decreased) {
          particle[is]->resize(decreased * particle[is]->Np_total);
        }
      }
    }

    // number of unpacked particles
    num_unpacked.resize(Ns, 0);
  }

  template <typename BufferPtr>
  bool unpack(BufferPtr& mpibuf, int iz, int iy, int ix, int send_bound[3][2], int recv_bound[3][2])
  {
    // skip
    if (iz == 1 && iy == 1 && ix == 1)
      return false;

    if (chunk->get_nb_rank(iz - 1, iy - 1, ix - 1) == MPI_PROC_NULL)
      return false;

    //
    // copy to the end of particle array
    //
    uint8_t* recvptr = mpibuf->recvbuf.get(mpibuf->bufaddr(iz, iy, ix));

    for (int is = 0; is < Ns; is++) {
      int Np = particle[is]->Np;

      // header
      int rcnt;
      std::memcpy(&rcnt, recvptr, head_byte);
      recvptr += head_byte;

      // particles
      float64* ptcl = &particle[is]->xu(Np + num_unpacked[is], 0);
      std::memcpy(ptcl, recvptr, elem_byte * rcnt);
      recvptr += rcnt * elem_byte;

      // increment number of unpacked particles
      num_unpacked[is] += rcnt;
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
      int np_prev = particle[is]->Np;
      int np_next = particle[is]->Np + num_unpacked[is];
      chunk->set_boundary_particle_after_sendrecv(particle[is], np_prev, np_next - 1, is);
      chunk->count_particle(particle[is], np_prev, np_next - 1, false);
      // now update number of particles
      particle[is]->Np = np_next;
    }

    //
    // sort particle array and discard out-of-range particles
    //
    for (int is = 0; is < Ns; is++) {
      particle[is]->sort();
    }
  }
};

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
