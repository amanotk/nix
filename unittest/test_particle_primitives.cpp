// -*- C++ -*-

#include "nix.hpp"
#include "particle_primitives.hpp"
#include <experimental/mdspan>
#include <iostream>

#include "catch.hpp"

using namespace nix::typedefs;
using namespace nix::primitives;
namespace stdex    = std::experimental;
using Array2D      = stdex::mdspan<float64, stdex::dextents<size_t, 2>>;
using Array4D      = stdex::mdspan<float64, stdex::dextents<size_t, 4>>;
using uniform_rand = std::uniform_real_distribution<float64>;

// set random particle position
void set_random_particle(Array2D& xu, int k, float64 rmin, float64 rmax)
{
  std::random_device seed;
  std::mt19937_64    engine(seed());
  uniform_rand       rand(rmin, rmax);

  for (int ip = 0; ip < xu.extent(0); ip++) {
    xu(ip, k) = rand(engine);
  }
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

bool interpolate1st(int N)
{
  const float64 eps = 1.0e-14;

  bool status = true;

  float64 eb_data[2 * 2 * 2 * 6];
  auto    eb = stdex::mdspan(eb_data, 2, 2, 2, 6);

  // initialize field
  {
    std::random_device seed;
    std::mt19937_64    engine(seed());
    uniform_rand       rand(-1.0, +1.0);

    for (int iz = 0; iz < 2; iz++) {
      for (int iy = 0; iy < 2; iy++) {
        for (int ix = 0; ix < 2; ix++) {
          for (int ik = 0; ik < 6; ik++) {
            eb(iz, iy, ix, ik) = rand(engine);
          }
        }
      }
    }
  }

  // interpolation
  {
    int     jx    = 0;
    int     jy    = 0;
    int     jz    = 0;
    float64 wx[2] = {0};
    float64 wy[2] = {0};
    float64 wz[2] = {0};
    float64 rdh   = 1.0;

    for (int iz = 0; iz < N; iz++) {
      for (int iy = 0; iy < N; iy++) {
        for (int ix = 0; ix < N; ix++) {
          float64 x = static_cast<float64>(ix) / (N - 1);
          float64 y = static_cast<float64>(iy) / (N - 1);
          float64 z = static_cast<float64>(iz) / (N - 1);
          shape1(x, 0.0, rdh, wx);
          shape1(y, 0.0, rdh, wy);
          shape1(z, 0.0, rdh, wz);

          for (int ik = 0; ik < 6; ik++) {
            // interpolation
            float64 val1 = interp3d1(eb, 0, 0, 0, ik, wz, wy, wx, 1.0);

            // check
            float64 val2 = 0;
            for (int jz = 0; jz < 2; jz++) {
              for (int jy = 0; jy < 2; jy++) {
                for (int jx = 0; jx < 2; jx++) {
                  val2 += eb(jz, jy, jx, ik) * wz[jz] * wy[jy] * wx[jx];
                }
              }
            }

            status = status & (std::abs(val1 - val2) < std::max(eps, eps * std::abs(val1)));
          }
        }
      }
    }

    REQUIRE(status);
  }

  return status;
}

bool interpolate2nd(int N)
{
  const float64 eps = 1.0e-14;

  bool status = true;

  float64 eb_data[3 * 3 * 3 * 6];
  auto    eb = stdex::mdspan(eb_data, 3, 3, 3, 6);

  // initialize field
  {
    std::random_device seed;
    std::mt19937_64    engine(seed());
    uniform_rand       rand(-1.0, +1.0);

    for (int iz = 0; iz < 3; iz++) {
      for (int iy = 0; iy < 3; iy++) {
        for (int ix = 0; ix < 3; ix++) {
          for (int ik = 0; ik < 6; ik++) {
            eb(iz, iy, ix, ik) = rand(engine);
          }
        }
      }
    }
  }

  // interpolation
  {
    float64 wx[3] = {0};
    float64 wy[3] = {0};
    float64 wz[3] = {0};
    float64 rdh   = 1.0;

    for (int iz = 0; iz < N; iz++) {
      for (int iy = 0; iy < N; iy++) {
        for (int ix = 0; ix < N; ix++) {
          float64 x = static_cast<float64>(ix) / (N - 1) - 0.5;
          float64 y = static_cast<float64>(iy) / (N - 1) - 0.5;
          float64 z = static_cast<float64>(iz) / (N - 1) - 0.5;
          shape2(x, 0.0, rdh, wx);
          shape2(y, 0.0, rdh, wy);
          shape2(z, 0.0, rdh, wz);

          for (int ik = 0; ik < 6; ik++) {
            // interpolation
            float64 val1 = interp3d2(eb, 1, 1, 1, ik, wz, wy, wx, 1.0);

            // check
            float64 val2 = 0;
            for (int jz = 0; jz < 3; jz++) {
              for (int jy = 0; jy < 3; jy++) {
                for (int jx = 0; jx < 3; jx++) {
                  val2 += eb(jz, jy, jx, ik) * wz[jz] * wy[jy] * wx[jx];
                }
              }
            }

            status = status & (std::abs(val1 - val2) < std::max(eps, eps * std::abs(val1)));
          }
        }
      }
    }

    REQUIRE(status);
  }

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
  int ix0 = digitize(xv[0], 0.0, rdh);
  int iy0 = digitize(xv[1], 0.0, rdh);
  int iz0 = digitize(xv[2], 0.0, rdh);

  shape1(xv[0], ix0 * delh, rdh, &ss[0][0][1], q);
  shape1(xv[1], iy0 * delh, rdh, &ss[0][1][1], q);
  shape1(xv[2], iz0 * delh, rdh, &ss[0][2][1], q);

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
  int ix1 = digitize(xu[0], 0.0, rdh);
  int iy1 = digitize(xu[1], 0.0, rdh);
  int iz1 = digitize(xu[2], 0.0, rdh);

  shape1(xu[0], ix1 * delh, rdh, &ss[1][0][1 + ix1 - ix0], q);
  shape1(xu[1], iy1 * delh, rdh, &ss[1][1][1 + iy1 - iy0], q);
  shape1(xu[2], iz1 * delh, rdh, &ss[1][2][1 + iz1 - iz0], q);

  // calculate charge and current density
  esirkepov3d1(dhdt, dhdt, dhdt, ss, cur);

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

// Esirkepov's scheme in 3D with 2nd order shape function
bool esirkepov3d2nd(const float64 delt, const float64 delh, float64 xu[7], float64 xv[7],
                    float64 rho[5][5][5], float64 cur[5][5][5][4], const float64 epsilon = 1.0e-14)
{
  const float64 q    = 1.0;
  const float64 rdh  = 1 / delh;
  const float64 dh2  = delh / 2;
  const float64 dhdt = delh / delt;

  bool    status  = true;
  float64 rhosum0 = 0;
  float64 rhosum1 = 0;
  float64 rhosum2 = 0;

  float64 ss[2][3][5] = {0};

  xv[0] = xu[0];
  xv[1] = xu[1];
  xv[2] = xu[2];
  xu[0] = xu[0] + xu[3] * delt;
  xu[1] = xu[1] + xu[4] * delt;
  xu[2] = xu[2] + xv[5] * delt;

  //
  // before move
  //
  int ix0 = digitize(xv[0], -dh2, rdh);
  int iy0 = digitize(xv[1], -dh2, rdh);
  int iz0 = digitize(xv[2], -dh2, rdh);

  shape2(xv[0], ix0 * delh, rdh, &ss[0][0][1], q);
  shape2(xv[1], iy0 * delh, rdh, &ss[0][1][1], q);
  shape2(xv[2], iz0 * delh, rdh, &ss[0][2][1], q);

  // check charge density
  for (int jz = 0; jz < 5; jz++) {
    for (int jy = 0; jy < 5; jy++) {
      for (int jx = 0; jx < 5; jx++) {
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
  int ix1 = digitize(xu[0], -dh2, rdh);
  int iy1 = digitize(xu[1], -dh2, rdh);
  int iz1 = digitize(xu[2], -dh2, rdh);

  shape2(xu[0], ix1 * delh, rdh, &ss[1][0][1 + ix1 - ix0], q);
  shape2(xu[1], iy1 * delh, rdh, &ss[1][1][1 + iy1 - iy0], q);
  shape2(xu[2], iz1 * delh, rdh, &ss[1][2][1 + iz1 - iz0], q);

  // calculate charge and current density
  esirkepov3d2(dhdt, dhdt, dhdt, ss, cur);

  // check charge density
  for (int jz = 0; jz < 5; jz++) {
    for (int jy = 0; jy < 5; jy++) {
      for (int jx = 0; jx < 5; jx++) {
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

TEST_CASE("Interpolation with first-order shape function")
{
  const int N = GENERATE(8, 16, 32);

  REQUIRE(interpolate1st(N) == true);
}

TEST_CASE("Interpolation with second-order shape function")
{
  const int N = GENERATE(8, 16, 32);

  REQUIRE(interpolate2nd(N) == true);
}

TEST_CASE("Esirkepov scheme 3D with first-order shape function")
{
  const int     Np   = 1000;
  const float64 delt = GENERATE(0.5, 1.0, 2.0);
  const float64 delh = GENERATE(0.5, 1.0, 2.0);
  const float64 delv = delh / delt;
  const float64 eps  = 1.0e-14;

  float64 xv_data[Np * 7];
  float64 xu_data[Np * 7];
  auto    xv = stdex::mdspan(xv_data, Np, 7);
  auto    xu = stdex::mdspan(xu_data, Np, 7);

  // position
  set_random_particle(xu, 0, +1.5 * delh, +2.5 * delh);
  set_random_particle(xu, 1, +1.5 * delh, +2.5 * delh);
  set_random_particle(xu, 2, +1.5 * delh, +2.5 * delh);

  // velocity
  set_random_particle(xu, 3, -1.0 * delv, +1.0 * delv);
  set_random_particle(xu, 4, -1.0 * delv, +1.0 * delv);
  set_random_particle(xu, 5, -1.0 * delv, +1.0 * delv);

  SECTION("check Esirkepov's scheme for individual particles")
  {
    bool status1 = true;
    bool status2 = true;

    for (int ip = 0; ip < Np; ip++) {
      float64 cur[4][4][4][4] = {0};
      float64 rho[4][4][4]    = {0};

      float64* xv_ptr = &xv(ip, 0);
      float64* xu_ptr = &xu(ip, 0);

      status1 = status1 & esirkepov3d1st(delt, delh, xu_ptr, xv_ptr, rho, cur, eps);
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

    for (int ip = 0; ip < Np; ip++) {
      float64* xv_ptr = &xv(ip, 0);
      float64* xu_ptr = &xu(ip, 0);

      status1 = status1 & esirkepov3d1st(delt, delh, xu_ptr, xv_ptr, rho, cur, eps);
    }
    status2 = status2 & check_charge_continuity(delt, delh, rho, cur, eps);

    REQUIRE(status1); // charge density
    REQUIRE(status2); // charge continuity
  }
}

TEST_CASE("Esirkepov scheme 3D with second-order shape function")
{
  const int     Np   = 1000;
  const float64 delt = GENERATE(0.5, 1.0, 2.0);
  const float64 delh = GENERATE(0.5, 1.0, 2.0);
  const float64 delv = delh / delt;
  const float64 eps  = 1.0e-14;

  float64 xv_data[Np * 7];
  float64 xu_data[Np * 7];
  auto    xv = stdex::mdspan(xv_data, Np, 7);
  auto    xu = stdex::mdspan(xu_data, Np, 7);

  // position
  set_random_particle(xu, 0, +1.5 * delh, +2.5 * delh);
  set_random_particle(xu, 1, +1.5 * delh, +2.5 * delh);
  set_random_particle(xu, 2, +1.5 * delh, +2.5 * delh);

  // velocity
  set_random_particle(xu, 3, -1.0 * delv, +1.0 * delv);
  set_random_particle(xu, 4, -1.0 * delv, +1.0 * delv);
  set_random_particle(xu, 5, -1.0 * delv, +1.0 * delv);

  SECTION("check Esirkepov's scheme for individual particles")
  {
    bool status1 = true;
    bool status2 = true;

    for (int ip = 0; ip < Np; ip++) {
      float64 cur[5][5][5][4] = {0};
      float64 rho[5][5][5]    = {0};

      float64* xv_ptr = &xv(ip, 0);
      float64* xu_ptr = &xu(ip, 0);

      status1 = status1 & esirkepov3d2nd(delt, delh, xu_ptr, xv_ptr, rho, cur, eps);
      status2 = status2 & check_charge_continuity(delt, delh, rho, cur, eps);
    }

    REQUIRE(status1); // charge density
    REQUIRE(status2); // charge continuity
  }
  SECTION("check Esirkepov's scheme for group of particles")
  {
    bool status1 = true;
    bool status2 = true;

    float64 cur[5][5][5][4] = {0};
    float64 rho[5][5][5]    = {0};

    for (int ip = 0; ip < Np; ip++) {
      float64* xv_ptr = &xv(ip, 0);
      float64* xu_ptr = &xu(ip, 0);

      status1 = status1 & esirkepov3d2nd(delt, delh, xu_ptr, xv_ptr, rho, cur, eps);
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
