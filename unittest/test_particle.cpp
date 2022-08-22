// -*- C++ -*-

#include "../particle.hpp"
#include <iostream>

#include "../catch.hpp"

using uniform_rand = std::uniform_real_distribution<float64>;

// set random particle position
void set_random_particle(Particle &particle, int k, float64 rmin, float64 rmax)
{
  std::random_device seed;
  std::mt19937       engine(seed());
  uniform_rand       rand(rmin, rmax);

  for (int ip = 0; ip < particle.Np; ip++) {
    particle.xu.at(ip, k) = rand(engine);
  }
}

// prepare particle sort for 1D mesh
void prepare_sort1d(Particle &particle, const int Nx)
{
  const float64 delh = 1.0;
  const float64 xmin = 0.0;

  // set random particle position
  set_random_particle(particle, 0, xmin - delh, xmin + delh * (Nx + 1));

  // prepare for sort
  int last = particle.Ng;
  particle.gindex.fill(0);
  particle.count.fill(0);
  for (int ip = 0; ip < particle.Np; ip++) {
    int ix = Particle::digitize(particle.xu.at(ip, 0), xmin, 1 / delh);
    int ii = ix;

    // take care out-of-bounds particles
    ii = (ix < 0 || ix >= Nx) ? last : ii;

    particle.gindex.at(ip) = ii;
    particle.count.at(ii)++;
  }
}

// check particle sort for 1D mesh
bool check_sort1d(Particle &particle, const int Nx)
{
  const float64 delh = 1.0;
  const float64 xmin = 0.0;

  bool status = true;

  for (int ii = 0; ii < particle.Ng; ii++) {
    for (int ip = particle.pindex.at(ii); ip < particle.pindex.at(ii + 1); ip++) {
      int jx = Particle::digitize(particle.xu.at(ip, 0), xmin, 1 / delh);
      int jj = jx;

      status = status & (ii == jj);
    }
  }

  return status;
}

// prepare particle sort for 2D mesh
void prepare_sort2d(Particle &particle, const int Nx, const int Ny)
{
  const float64 delh = 1.0;
  const float64 xmin = 0.0;

  // set random particle position
  set_random_particle(particle, 0, xmin - delh, xmin + delh * (Nx + 1));
  set_random_particle(particle, 1, xmin - delh, xmin + delh * (Ny + 1));

  // prepare for sort
  int last = particle.Ng;
  particle.gindex.fill(0);
  particle.count.fill(0);
  for (int ip = 0; ip < particle.Np; ip++) {
    int ix = Particle::digitize(particle.xu.at(ip, 0), xmin, 1 / delh);
    int iy = Particle::digitize(particle.xu.at(ip, 1), xmin, 1 / delh);
    int ii = ix + iy * Nx;

    // take care out-of-bounds particles
    ii = (ix < 0 || ix >= Nx) ? last : ii;
    ii = (iy < 0 || iy >= Ny) ? last : ii;

    particle.gindex.at(ip) = ii;
    particle.count.at(ii)++;
  }
}

// check particle sort for 2D mesh
bool check_sort2d(Particle &particle, const int Nx, const int Ny)
{
  const float64 delh = 1.0;
  const float64 xmin = 0.0;

  bool status = true;

  for (int ii = 0; ii < particle.Ng; ii++) {
    for (int ip = particle.pindex.at(ii); ip < particle.pindex.at(ii + 1); ip++) {
      int jx = Particle::digitize(particle.xu.at(ip, 0), xmin, 1 / delh);
      int jy = Particle::digitize(particle.xu.at(ip, 1), xmin, 1 / delh);
      int jj = jx + jy * Nx;

      status = status & (ii == jj);
    }
  }

  return status;
}

// prepare particle sort for 3D mesh
void prepare_sort3d(Particle &particle, const int Nx, const int Ny, const int Nz)
{
  const float64 delh = 1.0;
  const float64 xmin = 0.0;

  // set random particle position
  set_random_particle(particle, 0, xmin - delh, xmin + delh * (Nx + 1));
  set_random_particle(particle, 1, xmin - delh, xmin + delh * (Ny + 1));
  set_random_particle(particle, 2, xmin - delh, xmin + delh * (Nz + 1));

  // prepare for sort
  int last = particle.Ng;
  particle.gindex.fill(0);
  particle.count.fill(0);
  for (int ip = 0; ip < particle.Np; ip++) {
    int ix = Particle::digitize(particle.xu.at(ip, 0), xmin, 1 / delh);
    int iy = Particle::digitize(particle.xu.at(ip, 1), xmin, 1 / delh);
    int iz = Particle::digitize(particle.xu.at(ip, 2), xmin, 1 / delh);
    int ii = ix + iy * Nx + iz * Nx * Ny;

    // take care out-of-bounds particles
    ii = (ix < 0 || ix >= Nx) ? last : ii;
    ii = (iy < 0 || iy >= Ny) ? last : ii;
    ii = (iz < 0 || iz >= Nz) ? last : ii;

    particle.gindex.at(ip) = ii;
    particle.count.at(ii)++;
  }
}

// check particle sort for 3D mesh
bool check_sort3d(Particle &particle, const int Nx, const int Ny, const int Nz)
{
  const float64 delh = 1.0;
  const float64 xmin = 0.0;

  bool status = true;

  for (int ii = 0; ii < particle.Ng; ii++) {
    for (int ip = particle.pindex.at(ii); ip < particle.pindex.at(ii + 1); ip++) {
      int jx = Particle::digitize(particle.xu.at(ip, 0), xmin, 1 / delh);
      int jy = Particle::digitize(particle.xu.at(ip, 1), xmin, 1 / delh);
      int jz = Particle::digitize(particle.xu.at(ip, 2), xmin, 1 / delh);
      int jj = jx + jy * Nx + jz * Nx * Ny;

      status = status & (ii == jj);
    }
  }

  return status;
}

// check charge continuity equation
template <int N>
bool check_charge_continuity(const float64 delt, const float64 delh, const float64 rho[N][N][N],
                             const float64 cur[N][N][N][4], const float64 epsilon = 1.0e-14)
{
  bool    status = true;
  float64 errsum = 0.0;
  float64 errnrm = 0.0;

  float64 J[N + 1][N + 1][N + 1][4] = {0};

  for (int jz = 0; jz < N; jz++) {
    for (int jy = 0; jy < N; jy++) {
      for (int jx = 0; jx < N; jx++) {
        J[jz][jy][jx][0] = cur[jz][jy][jx][0];
        J[jz][jy][jx][1] = cur[jz][jy][jx][1];
        J[jz][jy][jx][2] = cur[jz][jy][jx][2];
        J[jz][jy][jx][3] = cur[jz][jy][jx][3];
      }
    }
  }

  for (int jz = 0; jz < N; jz++) {
    for (int jy = 0; jy < N; jy++) {
      for (int jx = 0; jx < N; jx++) {
        errnrm += std::abs(J[jz][jy][jx][0]);
        errsum += std::abs((J[jz][jy][jx][0] - rho[jz][jy][jx]) +
                           delt / delh * (J[jz][jy][jx + 1][1] - J[jz][jy][jx][1]) +
                           delt / delh * (J[jz][jy + 1][jx][2] - J[jz][jy][jx][2]) +
                           delt / delh * (J[jz + 1][jy][jx][3] - J[jz][jy][jx][3]));
      }
    }
  }

  status = status & (errsum < epsilon * errnrm);
  return status;
}

// Esirkepov's scheme in 3D with 1st order shape function
bool esirkepov3d1st(const float64 delt, const float64 delh, float64 xu[7], float64 xv[7],
                    float64 rho[4][4][4], float64 cur[4][4][4][4], const float64 epsilon = 1.0e-14)
{
  const float64 q    = 1.0;
  const float64 rdh  = 1 / delh;
  const float64 dhdt = delh / delt;

  bool    status  = true;
  float64 rhosum0 = 0;
  float64 rhosum1 = 0;
  float64 rhosum2 = 0;

  float64 ss[2][3][4] = {0};

  xv[0] = xu[0];
  xv[1] = xu[1];
  xv[2] = xu[2];
  xu[0] = xu[0] + xu[3] * delt;
  xu[1] = xu[1] + xu[4] * delt;
  xu[2] = xu[2] + xv[5] * delt;

  //
  // before move
  //
  int ix0 = Particle::digitize(xv[0], 0.0, rdh);
  int iy0 = Particle::digitize(xv[1], 0.0, rdh);
  int iz0 = Particle::digitize(xv[2], 0.0, rdh);

  Particle::S1(xv[0], ix0 * delh, rdh, &ss[0][0][1], q);
  Particle::S1(xv[1], iy0 * delh, rdh, &ss[0][1][1], q);
  Particle::S1(xv[2], iz0 * delh, rdh, &ss[0][2][1], q);

  // check charge density
  for (int jz = 0; jz < 4; jz++) {
    for (int jy = 0; jy < 4; jy++) {
      for (int jx = 0; jx < 4; jx++) {
        float64 r = ss[0][0][jx] * ss[0][1][jy] * ss[0][2][jz];
        rhosum0 += cur[jz][jy][jx][0];
        rhosum1 += r;
        rho[jz][jy][jx] += r;
      }
    }
  }

  //
  // after move
  //
  int ix1 = Particle::digitize(xu[0], 0.0, rdh);
  int iy1 = Particle::digitize(xu[1], 0.0, rdh);
  int iz1 = Particle::digitize(xu[2], 0.0, rdh);

  Particle::S1(xu[0], ix1 * delh, rdh, &ss[1][0][1 + ix1 - ix0], q);
  Particle::S1(xu[1], iy1 * delh, rdh, &ss[1][1][1 + iy1 - iy0], q);
  Particle::S1(xu[2], iz1 * delh, rdh, &ss[1][2][1 + iz1 - iz0], q);

  // calculate charge and current density
  Particle::esirkepov3d1(dhdt, ss, cur);

  // check charge density
  for (int jz = 0; jz < 4; jz++) {
    for (int jy = 0; jy < 4; jy++) {
      for (int jx = 0; jx < 4; jx++) {
        rhosum2 += cur[jz][jy][jx][0];
      }
    }
  }

  // contribution to charge density is normalized to unity
  status = status & (std::abs(rhosum1 - 1) < epsilon);

  // charge density increases exactly by one
  status = status & (std::abs(rhosum2 - (rhosum0 + 1)) < epsilon * std::abs(rhosum2));

  return status;
}

//
// create particle
//
TEST_CASE("CreateParticle")
{
  const int Np = 1000;
  const int Ng = 100;

  Particle particle(Np, Ng);
  particle.Np = Np;

  // check array size
  REQUIRE(particle.xu.size() == Np * Particle::Nc);
  REQUIRE(particle.xv.size() == Np * Particle::Nc);
  REQUIRE(particle.gindex.size() == Np);
  REQUIRE(particle.pindex.size() == Ng + 1);
  REQUIRE(particle.count.size() == Ng + 1);

  // check particle data
  REQUIRE(xt::allclose(particle.xu, 0.0));
  REQUIRE(xt::allclose(particle.xv, 0.0));
}

//
// swap particle
//
TEST_CASE("SwapParticle")
{
  const int Np = 1000;
  const int Ng = 100;

  Particle particle(Np, Ng);
  particle.Np = Np;

  float64 *ptr1 = particle.xu.data();
  float64 *ptr2 = particle.xv.data();

  // set random number
  {
    std::random_device seed;
    std::mt19937       engine(seed());

    std::uniform_real_distribution<float64> rand(0.0, 1.0);

    for (int i = 0; i < Np * Particle::Nc; i++) {
      ptr1[i] = rand(engine);
      ptr2[i] = rand(engine);
    }
  }

  // before swap
  REQUIRE(particle.xu.data() == ptr1);
  REQUIRE(std::equal(particle.xu.begin(), particle.xu.end(), ptr1));
  REQUIRE(particle.xv.data() == ptr2);
  REQUIRE(std::equal(particle.xv.begin(), particle.xv.end(), ptr2));

  // swap xu and xv
  particle.swap();

  // after swap
  REQUIRE(particle.xu.data() == ptr2);
  REQUIRE(std::equal(particle.xu.begin(), particle.xu.end(), ptr2));
  REQUIRE(particle.xv.data() == ptr1);
  REQUIRE(std::equal(particle.xv.begin(), particle.xv.end(), ptr1));
}

//
// sort particle 1D
//
TEST_CASE("SortParticle1D")
{
  const int Np = GENERATE(100, 1000, 10000);
  const int Nx = GENERATE(8, 64, 256);

  Particle particle(Np, Nx);
  particle.Np = Np;

  // sort
  prepare_sort1d(particle, Nx);
  particle.sort();

  // check result
  REQUIRE(check_sort1d(particle, Nx) == true);
}

//
// sort particle 2D
//
TEST_CASE("SortParticle2D")
{
  const int Np = GENERATE(100, 1000, 10000);
  const int Nx = GENERATE(8, 16);
  const int Ny = GENERATE(8, 16);

  Particle particle(Np, Nx * Ny);
  particle.Np = Np;

  // sort
  prepare_sort2d(particle, Nx, Ny);
  particle.sort();

  // check result
  REQUIRE(check_sort2d(particle, Nx, Ny) == true);
}

//
// sort particle 3D
//
TEST_CASE("SortParticle3D")
{
  const int Np = GENERATE(100, 1000, 10000);
  const int Nx = GENERATE(8, 16);
  const int Ny = GENERATE(8, 16);
  const int Nz = GENERATE(8, 16);

  Particle particle(Np, Nx * Ny * Nz);
  particle.Np = Np;

  // sort
  prepare_sort3d(particle, Nx, Ny, Nz);
  particle.sort();

  // check result
  REQUIRE(check_sort3d(particle, Nx, Ny, Nz) == true);
}

//
// interpolation of electromagnetic field
//
TEST_CASE("Interp3D1st")
{
  const int     Np   = GENERATE(100, 1000);
  const int     Nx   = GENERATE(8, 16);
  const int     Ny   = GENERATE(8, 16);
  const int     Nz   = GENERATE(8, 16);
  const int     Nb   = 1;
  const float64 delh = 1.0;
  const float64 eps  = 1.0e-14;

  float64 a[6]  = {0};
  float64 bx[6] = {0};
  float64 by[6] = {0};
  float64 bz[6] = {0};
  float64 xmin  = 0.0;
  float64 ymin  = 0.0;
  float64 zmin  = 0.0;
  float64 rdh   = 1 / delh;

  xt::xtensor<float64, 4> eb;
  xt::xtensor<float64, 1> xc;
  xt::xtensor<float64, 1> yc;
  xt::xtensor<float64, 1> zc;

  Particle particle(Np, Nz * Ny * Nx);
  particle.Np = Np;

  // linear functional form
  {
    std::random_device seed;
    std::mt19937       engine(seed());
    uniform_rand       rand(-1.0, +1.0);

    for (int ik = 0; ik < 6; ik++) {
      a[ik]  = rand(engine);
      bx[ik] = rand(engine);
      by[ik] = rand(engine);
      bz[ik] = rand(engine);
    }
  }

  // initialization
  {
    size_t nz = Nz + 2 * Nb;
    size_t ny = Ny + 2 * Nb;
    size_t nx = Nx + 2 * Nb;

    eb.resize({nz, ny, nx, 6});

    xc = xt::linspace(xmin, xmin + (nx - 1) * delh, nx);
    yc = xt::linspace(ymin, ymin + (ny - 1) * delh, ny);
    zc = xt::linspace(zmin, zmin + (nz - 1) * delh, nz);

    for (int iz = 0; iz < nz; iz++) {
      for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
          for (int ik = 0; ik < 6; ik++) {
            eb(iz, iy, ix, ik) = a[ik] + bx[ik] * xc(ix) + by[ik] * yc(iy) + bz[ik] * zc(iz);
          }
        }
      }
    }

    // position
    set_random_particle(particle, 0, xmin + delh, xmin + (nx - 2) * delh);
    set_random_particle(particle, 1, ymin + delh, ymin + (ny - 2) * delh);
    set_random_particle(particle, 2, zmin + delh, zmin + (Nz - 2) * delh);
  }

  // interpolation
  {
    bool status = true;

    for (int ip = 0; ip < particle.Np; ip++) {
      float64 wx[2] = {0};
      float64 wy[2] = {0};
      float64 wz[2] = {0};

      float64 *xu = &particle.xu(ip, 0);

      int ix = Particle::digitize(xu[0], xmin, rdh);
      int iy = Particle::digitize(xu[1], ymin, rdh);
      int iz = Particle::digitize(xu[2], zmin, rdh);

      Particle::S1(xu[0], ix * delh, rdh, wx);
      Particle::S1(xu[1], iy * delh, rdh, wy);
      Particle::S1(xu[2], iz * delh, rdh, wz);

      for (int ik = 0; ik < 6; ik++) {
        float64 val1 = a[ik] + bx[ik] * xu[0] + by[ik] * xu[1] + bz[ik] * xu[2];
        float64 val2 = Particle::interp3d1(eb, iz, iy, ix, ik, wz, wy, wx, 1.0);

        // check absolute error for small values, and relative error otherwise
        status = status & (std::abs(val1 - val2) < std::max(eps, eps*std::abs(val1)));
      }
    }

    REQUIRE(status);
  }
}

//
// Esirkepov's density decomposition scheme 3D for first-order shape function
//
TEST_CASE("Esirkepov3D1st")
{
  const int     Np   = 1000;
  const float64 delt = GENERATE(0.5, 1.0, 2.0);
  const float64 delh = GENERATE(0.5, 1.0, 2.0);
  const float64 delv = delh / delt;
  const float64 eps  = 1.0e-14;

  Particle particle(Np, 1);
  particle.Np = Np;

  // position
  set_random_particle(particle, 0, +1.5 * delh, +2.5 * delh);
  set_random_particle(particle, 1, +1.5 * delh, +2.5 * delh);
  set_random_particle(particle, 2, +1.5 * delh, +2.5 * delh);

  // velocity
  set_random_particle(particle, 3, -1.0 * delv, +1.0 * delv);
  set_random_particle(particle, 4, -1.0 * delv, +1.0 * delv);
  set_random_particle(particle, 5, -1.0 * delv, +1.0 * delv);

  SECTION("check Esirkepov's scheme for individual particles")
  {
    bool status1 = true;
    bool status2 = true;

    for (int ip = 0; ip < particle.Np; ip++) {
      float64 cur[4][4][4][4] = {0};
      float64 rho[4][4][4]    = {0};

      float64 *xv = &particle.xv(ip, 0);
      float64 *xu = &particle.xu(ip, 0);

      status1 = status1 & esirkepov3d1st(delt, delh, xu, xv, rho, cur, eps);
      status2 = status2 & check_charge_continuity(delt, delh, rho, cur, eps);
    }

    REQUIRE(status1); // charge density
    REQUIRE(status2); // charge continuity
  }
  SECTION("check Esirkepov's scheme for group of particles")
  {
    bool status1 = true;
    bool status2 = true;

    float64 cur[4][4][4][4] = {0};
    float64 rho[4][4][4]    = {0};

    for (int ip = 0; ip < particle.Np; ip++) {
      float64 *xv = &particle.xv(ip, 0);
      float64 *xu = &particle.xu(ip, 0);

      status1 = status1 & esirkepov3d1st(delt, delh, xu, xv, rho, cur, eps);
    }
    status2 = status2 & check_charge_continuity(delt, delh, rho, cur, eps);

    REQUIRE(status1); // charge density
    REQUIRE(status2); // charge continuity
  }
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
