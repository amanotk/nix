// -*- C++ -*-

#include "../application.hpp"
#include "../chunk3d.hpp"
#include "../chunkmap.hpp"

#include "../catch.hpp"

constexpr int Nb = 1;

class TestChunk;

using BaseApp   = Application<TestChunk, ChunkMap<3>>;
using rand_type = std::uniform_real_distribution<float64>;

// class for testing Chunk3D
class TestChunk : public Chunk3D<Nb>
{
private:
  rand_type uniform_rand;

  ParticleList            up;
  xt::xtensor<float64, 4> uf;

public:
  TestChunk(const int dims[3], const int id = 0) : Chunk3D<Nb>(dims, id)
  {
    size_t Nz = this->dims[0] + 2 * Nb;
    size_t Ny = this->dims[1] + 2 * Nb;
    size_t Nx = this->dims[2] + 2 * Nb;

    rand_type::param_type param(-1.0, +1.0);
    uniform_rand.param(param);
  }

  void allocate_memory(const int Nppc, const int Ns)
  {
    size_t Nz = dims[0] + 2 * Nb;
    size_t Ny = dims[1] + 2 * Nb;
    size_t Nx = dims[2] + 2 * Nb;
    size_t Ng = dims[0] * dims[1] * dims[2];

    // memory allocation for field
    uf.resize({Nz, Ny, Nx, 6});
    uf.fill(0);

    // memory allocation for particle
    up.resize(Ns);
    for (int is = 0; is < Ns; is++) {
      up[is]     = std::make_shared<Particle>(2 * Nppc * Ng, Ng);
      up[is]->Np = Nppc * Ng;
    }

    // initialize MPI buffer
    mpibufvec.push_back(std::make_unique<MpiBuffer>());
    set_mpi_buffer(0, sizeof(float64) * 6, mpibufvec[0].get());
    mpibufvec.push_back(std::make_unique<MpiBuffer>());
    set_mpi_buffer(1, sizeof(float64) * 7 * Nppc * 10, mpibufvec[1].get());
  }

  virtual void push(const float64 delt) override
  {
    // push particle position
    for (int is = 0; is < up.size(); is++) {
      for (int ip = 0; ip < up[is]->Np; ip++) {
        up[is]->xu(ip, 0) += +up[is]->xu(ip, 3) * delt;
        up[is]->xu(ip, 1) += +up[is]->xu(ip, 4) * delt;
        up[is]->xu(ip, 2) += +up[is]->xu(ip, 5) * delt;
      }
      // count
      count_particle(up[is], 0, up[is]->Np - 1, true);
    }
  }

  virtual void set_boundary_begin(const int mode) override
  {
    set_boundary_physical(mode);

    switch (mode) {
    case 0:
      begin_bc_exchange(mpibufvec[0].get(), uf);
      break;
    case 1:
      begin_bc_exchange(mpibufvec[1].get(), up);
      break;
    default:
      break;
    }
  }

  virtual void set_boundary_end(const int mode) override
  {
    switch (mode) {
    case 0:
      end_bc_exchange(mpibufvec[0].get(), uf, false);
      break;
    case 1:
      end_bc_exchange(mpibufvec[1].get(), up);
      break;
    default:
      break;
    }
  }

  virtual void set_boundary_particle(ParticlePtr particle, int Lbp, int Ubp) override
  {
    float64 xrange[2] = {0.0, ndims[2] * delh};
    float64 yrange[2] = {0.0, ndims[1] * delh};
    float64 zrange[2] = {0.0, ndims[0] * delh};

    // push particle position
    for (int ip = Lbp; ip <= Ubp; ip++) {
      float64 *xu = &particle->xu(ip, 0);

      // apply periodic boundary condition
      xu[0] = xu[0] < xrange[0] ? xu[0] + xrange[1] : xu[0];
      xu[0] = xu[0] > xrange[1] ? xu[0] - xrange[1] : xu[0];
      xu[1] = xu[1] < yrange[0] ? xu[1] + yrange[1] : xu[1];
      xu[1] = xu[1] > yrange[1] ? xu[1] - yrange[1] : xu[1];
      xu[2] = xu[2] < zrange[0] ? xu[2] + zrange[1] : xu[2];
      xu[2] = xu[2] > zrange[1] ? xu[2] - zrange[1] : xu[2];
    }
  }

  void initialize_field()
  {
    int stride[3] = {ndims[1] * ndims[2], ndims[2], 1};

    std::mt19937 mt;

    for (int iz = Lbz; iz <= Ubz; iz++) {
      int jz = iz - Lbz + offset[0];
      for (int iy = Lby; iy <= Uby; iy++) {
        int jy = iy - Lby + offset[1];
        for (int ix = Lbx; ix <= Ubx; ix++) {
          int jx = ix - Lbx + offset[2];
          int ii = jz * stride[0] + jy * stride[1] + jx * stride[2];
          mt.seed(ii);
          for (int ik = 0; ik < 6; ik++) {
            uf(iz, iy, ix, ik) = uniform_rand(mt);
          }
        }
      }
    }
  }

  bool check_field()
  {
    bool status    = true;
    int  stride[3] = {ndims[1] * ndims[2], ndims[2], 1};

    std::mt19937 mt;

    for (int iz = Lbz - Nb; iz <= Ubz + Nb; iz++) {
      int jz = iz - Lbz + offset[0];
      for (int iy = Lby - Nb; iy <= Uby + Nb; iy++) {
        int jy = iy - Lby + offset[1];
        for (int ix = Lbx - Nb; ix <= Ubx + Nb; ix++) {
          int jx = ix - Lbx + offset[2];

          // periodic boundary
          jz = jz < 0 ? jz + ndims[0] : jz;
          jy = jy < 0 ? jy + ndims[1] : jy;
          jx = jx < 0 ? jx + ndims[2] : jx;
          jz = jz >= ndims[0] ? jz - ndims[0] : jz;
          jy = jy >= ndims[1] ? jy - ndims[0] : jy;
          jx = jx >= ndims[2] ? jx - ndims[2] : jx;

          int ii = jz * stride[0] + jy * stride[1] + jx * stride[2];
          mt.seed(ii);
          for (int ik = 0; ik < 6; ik++) {
            bool check = (std::abs(uf(iz, iy, ix, ik) - uniform_rand(mt)) < 1.0e-14);
            status     = status & check;
          }
        }
      }
    }

    return status;
  }

  void initialize_particle(const int Nppc, const int Ns)
  {
    int stride[3] = {ndims[1] * ndims[2], ndims[2], 1};
    int nps[Ns]   = {0};

    std::mt19937 mt;

    for (int iz = Lbz; iz <= Ubz; iz++) {
      int jz = iz - Lbz + offset[0];
      for (int iy = Lby; iy <= Uby; iy++) {
        int jy = iy - Lby + offset[1];
        for (int ix = Lbx; ix <= Ubx; ix++) {
          int jx = ix - Lbx + offset[2];
          for (int is = 0; is < Ns; is++) {
            for (int ip = nps[is], jp = 0; ip < nps[is] + Nppc; ip++, jp++) {
              int64 id64 = Nppc * (jz * stride[0] + jy * stride[1] + jx * stride[2]) + jp;
              mt.seed(id64);
              // position and velocity
              up[is]->xu(ip, 0) = delh * uniform_rand(mt) * 0.5 + xc(ix);
              up[is]->xu(ip, 1) = delh * uniform_rand(mt) * 0.5 + yc(iy);
              up[is]->xu(ip, 2) = delh * uniform_rand(mt) * 0.5 + zc(iz);
              up[is]->xu(ip, 3) = delh * uniform_rand(mt);
              up[is]->xu(ip, 4) = delh * uniform_rand(mt);
              up[is]->xu(ip, 5) = delh * uniform_rand(mt);
              // ID
              std::memcpy(&up[is]->xu(ip, 6), &id64, sizeof(int64));
            }
            nps[is] += Nppc;
          }
        }
      }
    }

    for (int is = 0; is < Ns; is++) {
      up[is]->Np = nps[is];
    }
  }

  bool check_particle(const int Nppc, const int Ns, const float64 delt)
  {
    bool status = true;

    std::mt19937 mt;

    for (int is = 0; is < up.size(); is++) {
      for (int ip = 0; ip < up[is]->Np; ip++) {
        // ID
        int64 id64 = 0;
        std::memcpy(&id64, &up[is]->xu(ip, 6), sizeof(int64));
        mt.seed(id64);

        float64 xv[6] = {0};
        float64 xu[6] = {0};

        // position and velocity
        xv[0] = up[is]->xu(ip, 0) - up[is]->xu(ip, 3) * delt;
        xv[1] = up[is]->xu(ip, 1) - up[is]->xu(ip, 4) * delt;
        xv[2] = up[is]->xu(ip, 2) - up[is]->xu(ip, 5) * delt;
        xv[3] = up[is]->xu(ip, 3);
        xv[4] = up[is]->xu(ip, 4);
        xv[5] = up[is]->xu(ip, 5);

        // original position and velocity
        int ix = Particle::digitize(xv[0], xlim[0], 1 / delh) + Lbx;
        int iy = Particle::digitize(xv[1], ylim[0], 1 / delh) + Lby;
        int iz = Particle::digitize(xv[2], zlim[0], 1 / delh) + Lbz;

        xu[0] = delh * uniform_rand(mt) * 0.5 + xc(ix);
        xu[1] = delh * uniform_rand(mt) * 0.5 + yc(iy);
        xu[2] = delh * uniform_rand(mt) * 0.5 + zc(iz);
        xu[3] = delh * uniform_rand(mt);
        xu[4] = delh * uniform_rand(mt);
        xu[5] = delh * uniform_rand(mt);

        // check
        for (int ik = 0; ik < 6; ik++) {
          bool check = (std::abs(xv[ik] - xu[ik]) < 1.0e-14);
          status     = status & check;
        }
      }
    }

    return status;
  }

  void add_num_particle(int *nps)
  {
    for (int is = 0; is < up.size(); is++) {
      nps[is] += up[is]->Np;
    }
  }
};

// class for testing send/recv
class TestSendRecv : public BaseApp
{
public:
  TestSendRecv()
  {
    mpi_init_with_nullptr = true;
  }

  bool check_field_sendrecv()
  {
    const int mode = 0;

    bool status = true;

    // initialize and invoke send/recv
    for (int i = 0; i < numchunk; i++) {
      chunkvec[i]->initialize_field();
      chunkvec[i]->set_boundary_begin(mode);
    }

    // wait send/recv and check results
    for (int i = 0; i < numchunk; i++) {
      chunkvec[i]->set_boundary_end(mode);
      status = status & chunkvec[i]->check_field();
    }

    MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_CXX_BOOL, MPI_LAND, MPI_COMM_WORLD);
    if (thisrank == 0) {
      REQUIRE(status == true);
    }

    return status;
  }

  bool check_particle_sendrecv(const int Nppc, const int Ns)
  {
    const int mode = 1;

    int  nps1[Ns] = {0};
    int  nps2[Ns] = {0};
    bool status   = true;

    // initialize
    for (int i = 0; i < numchunk; i++) {
      chunkvec[i]->initialize_particle(Nppc, Ns);
      chunkvec[i]->add_num_particle(nps1);
    }
    MPI_Allreduce(MPI_IN_PLACE, nps1, Ns, MPI_INT32_T, MPI_SUM, MPI_COMM_WORLD);

    for (int step = 0; step < 1; step++) {
      // push and invoke send/recv
      for (int i = 0; i < numchunk; i++) {
        chunkvec[i]->push(delt);
        chunkvec[i]->set_boundary_begin(mode);
      }

      // wait send/recv and check results
      for (int i = 0; i < numchunk; i++) {
        chunkvec[i]->set_boundary_end(mode);
        status = status & chunkvec[i]->check_particle(Nppc, Ns, delt * (step + 1));
        chunkvec[i]->add_num_particle(nps2);
      }

      // check number of particles
      for (int is = 0; is < Ns; is++) {
        nps2[is] = 0;
      }
      for (int i = 0; i < numchunk; i++) {
        chunkvec[i]->add_num_particle(nps2);
      }

      MPI_Allreduce(MPI_IN_PLACE, nps2, Ns, MPI_INT32_T, MPI_SUM, MPI_COMM_WORLD);
      for (int is = 0; is < Ns; is++) {
        status = status & (nps1[is] == nps2[is]);
      }
    }

    MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_CXX_BOOL, MPI_LAND, MPI_COMM_WORLD);
    if (thisrank == 0) {
      REQUIRE(status == true);
    }

    return status;
  }

  void initialize(const int Nppc, const int Ns)
  {
    {
      // command-line arguments
      std::vector<std::string> args = {"TestSendRecv", "-e", "1", "-c", "default.json"};

      cl_argc = args.size();
      cl_argv = new char *[args.size()];
      for (int i = 0; i < args.size(); i++) {
        cl_argv[i] = const_cast<char *>(args[i].c_str());
      }

      BaseApp::initialize(cl_argc, cl_argv);

      delete[] cl_argv;
    }

    // setup offset
    for (int i = 0; i < numchunk; i++) {
      int ix, iy, iz;
      int offset[3];

      chunkmap->get_coordinate(chunkvec[i]->get_id(), iz, iy, ix);
      offset[0] = iz * ndims[0] / cdims[0];
      offset[1] = iy * ndims[1] / cdims[1];
      offset[2] = ix * ndims[2] / cdims[2];
      chunkvec[i]->set_global_context(offset, ndims);
      chunkvec[i]->allocate_memory(Nppc, Ns);
    }
  }
};

//
// test send/recv
//
TEST_CASE("TestSendRecv")
{
  const int Nppc = 10;
  const int Ns   = 2;

  TestSendRecv testapp;

  testapp.initialize(Nppc, Ns);
  testapp.check_field_sendrecv();
  testapp.check_particle_sendrecv(Nppc, Ns);
  testapp.finalize();
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
