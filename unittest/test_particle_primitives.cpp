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

void set_random_particle(Array2D& xu, float64 delh, float64 delv)
{
  std::random_device seed;
  std::mt19937_64    engine(seed());
  uniform_rand       rand(0, 1);

  for (int ip = 0; ip < xu.extent(0); ip++) {
    xu(ip, 0) = rand(engine) * delh;
    xu(ip, 1) = rand(engine) * delh;
    xu(ip, 2) = rand(engine) * delh;
    xu(ip, 3) = rand(engine) * delv * 2 - delv;
    xu(ip, 4) = rand(engine) * delv * 2 - delv;
    xu(ip, 5) = rand(engine) * delv * 2 - delv;
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
          shape<1>(x, 0.0, rdh, wx);
          shape<1>(y, 0.0, rdh, wy);
          shape<1>(z, 0.0, rdh, wz);

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
          shape<2>(x, 0.0, rdh, wx);
          shape<2>(y, 0.0, rdh, wy);
          shape<2>(z, 0.0, rdh, wz);

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

template <int Order>
bool test_esirkepov3d(const float64 delt, const float64 delh, float64 xu[7], float64 xv[7],
                      float64       rho[Order + 3][Order + 3][Order + 3],
                      float64       cur[Order + 3][Order + 3][Order + 3][4],
                      const float64 epsilon = 1.0e-14)
{
  const float64 rdh  = 1 / delh;
  const float64 dhdt = delh / delt;

  bool    status  = true;
  float64 rhosum0 = 0;
  float64 rhosum1 = 0;
  float64 rhosum2 = 0;

  float64 ss[2][3][Order + 3] = {0};

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

  shape<Order>(xv[0], ix0 * delh, rdh, &ss[0][0][1]);
  shape<Order>(xv[1], iy0 * delh, rdh, &ss[0][1][1]);
  shape<Order>(xv[2], iz0 * delh, rdh, &ss[0][2][1]);

  // check charge density
  for (int jz = 0; jz < Order + 3; jz++) {
    for (int jy = 0; jy < Order + 3; jy++) {
      for (int jx = 0; jx < Order + 3; jx++) {
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

  shape<Order>(xu[0], ix1 * delh, rdh, &ss[1][0][1 + ix1 - ix0]);
  shape<Order>(xu[1], iy1 * delh, rdh, &ss[1][1][1 + iy1 - iy0]);
  shape<Order>(xu[2], iz1 * delh, rdh, &ss[1][2][1 + iz1 - iz0]);

  // calculate charge and current density
  esirkepov3d<Order>(dhdt, dhdt, dhdt, ss, cur);

  // check charge density
  for (int jz = 0; jz < Order + 3; jz++) {
    for (int jy = 0; jy < Order + 3; jy++) {
      for (int jx = 0; jx < Order + 3; jx++) {
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

TEST_CASE("First-order shape function")
{
  const int     N        = 100;
  const float64 epsilon  = 1.0e-14;
  const float64 xmin     = 0;
  const float64 xmax     = 1;
  const float64 delx     = xmax - xmin;
  const float64 xeval[2] = {xmin, xmax};

  // analytic form
  auto W = [](float64 x) { return std::abs(x) < 1 ? 1 - std::abs(x) : 0; };

  // test
  bool status = true;
  for (int i = 0; i < N; i++) {
    float64 s[2];
    float64 x = xmin + (xmax - xmin) * i / (N - 1);

    shape<1>(x, xmin, 1 / delx, s);
    status = status & (std::abs(s[0] - W(xeval[0] - x)) < epsilon);
    status = status & (std::abs(s[1] - W(xeval[1] - x)) < epsilon);
  }
  REQUIRE(status == true);
}

TEST_CASE("Second-order shape function")
{
  const int     N        = 100;
  const float64 epsilon  = 1.0e-14;
  const float64 xmin     = -0.5;
  const float64 xmax     = +0.5;
  const float64 xmid     = 0;
  const float64 delx     = xmax - xmin;
  const float64 xeval[3] = {xmid - delx, xmid, xmid + delx};

  // analytic form
  auto W = [](float64 x) {
    float64 abs_x = std::abs(x);
    if (abs_x < 0.5) {
      return 0.75 - std::pow(x, 2);
    } else if (abs_x < 1.5) {
      return std::pow(3 - 2 * abs_x, 2) / 8;
    } else {
      return 0.0;
    }
  };

  // test
  bool status = true;
  for (int i = 0; i < N; i++) {
    float64 s[3];
    float64 x = xmin + (xmax - xmin) * i / (N - 1);

    shape<2>(x, xmid, 1 / delx, s);
    status = status & (std::abs(s[0] - W(xeval[0] - x)) < epsilon);
    status = status & (std::abs(s[1] - W(xeval[1] - x)) < epsilon);
    status = status & (std::abs(s[2] - W(xeval[2] - x)) < epsilon);
  }
  REQUIRE(status == true);
}

TEST_CASE("Third-order shape function")
{
  const int     N        = 100;
  const float64 epsilon  = 1.0e-14;
  const float64 xmin     = 0;
  const float64 xmax     = 1;
  const float64 delx     = xmax - xmin;
  const float64 xeval[4] = {xmin - delx, xmin, xmax, xmax + delx};

  // analytic form
  auto W = [](float64 x) {
    float64 abs_x = std::abs(x);
    if (abs_x < 1.0) {
      return 2 / 3.0 - std::pow(x, 2) + std::pow(abs_x, 3) / 2;
    } else if (abs_x < 2.0) {
      return std::pow(2 - abs_x, 3) / 6;
    } else {
      return 0.0;
    }
  };

  // test
  bool status = true;
  for (int i = 0; i < N; i++) {
    float64 s[4];
    float64 x = xmin + (xmax - xmin) * i / (N - 1);

    shape<3>(x, xmin, 1 / delx, s);
    status = status & (std::abs(s[0] - W(xeval[0] - x)) < epsilon);
    status = status & (std::abs(s[1] - W(xeval[1] - x)) < epsilon);
    status = status & (std::abs(s[2] - W(xeval[2] - x)) < epsilon);
    status = status & (std::abs(s[3] - W(xeval[3] - x)) < epsilon);
  }
  REQUIRE(status == true);
}

TEST_CASE("Fourth-order shape function")
{
  const int     N        = 100;
  const float64 epsilon  = 1.0e-14;
  const float64 xmin     = -0.5;
  const float64 xmax     = +0.5;
  const float64 xmid     = 0;
  const float64 delx     = xmax - xmin;
  const float64 xeval[5] = {xmid - 2 * delx, xmid - delx, xmid, xmid + delx, xmid + 2 * delx};

  // analytic form
  auto W = [](float64 x) {
    float64 abs_x = std::abs(x);
    if (abs_x < 0.5) {
      return 115 / 192.0 - 5 * std::pow(x, 2) / 8 + std::pow(x, 4) / 4;
    } else if (abs_x < 1.5) {
      return (55 + 20 * abs_x - 120 * std::pow(x, 2) + 80 * std::pow(abs_x, 3) -
              16 * std::pow(x, 4)) /
             96;
    } else if (abs_x < 2.5) {
      return std::pow(5 - 2 * abs_x, 4) / 384;
    } else {
      return 0.0;
    }
  };

  // test
  bool status = true;
  for (int i = 0; i < N; i++) {
    float64 s[5];
    float64 x = xmin + (xmax - xmin) * i / (N - 1);

    shape<4>(x, xmid, 1 / delx, s);
    status = status & (std::abs(s[0] - W(xeval[0] - x)) < epsilon);
    status = status & (std::abs(s[1] - W(xeval[1] - x)) < epsilon);
    status = status & (std::abs(s[2] - W(xeval[2] - x)) < epsilon);
    status = status & (std::abs(s[3] - W(xeval[3] - x)) < epsilon);
    status = status & (std::abs(s[4] - W(xeval[4] - x)) < epsilon);
  }
  REQUIRE(status == true);
}

TEST_CASE("First-order shape function for WT scheme")
{
  const int     N        = 100;
  const float64 epsilon  = 1.0e-14;
  const float64 xmin     = 0;
  const float64 xmax     = 1;
  const float64 delx     = xmax - xmin;
  const float64 xeval[2] = {xmin, xmax};
  const float64 delt     = GENERATE(0.1, 0.2, 0.3, 0.4, 0.5);

  // analytic form
  auto W = [](float64 x, float64 dt) {
    float64 abs_x = std::abs(x);
    if (0.5 - dt < abs_x && abs_x <= 0.5 + dt) {
      return (1 + 2 * dt - 2 * abs_x) / (4 * dt);
    } else if (abs_x <= 0.5 - dt) {
      return 1.0;
    } else {
      return 0.0;
    }
  };

  // test
  bool status = true;
  for (int i = 0; i < N; i++) {
    float64 s[2];
    float64 x = xmin + (xmax - xmin) * i / (N - 1);

    shape_wt<1>(x, xmin, 1 / delx, delt, 1 / delt, s);
    status = status & (std::abs(s[0] - W(xeval[0] - x, delt)) < epsilon);
    status = status & (std::abs(s[1] - W(xeval[1] - x, delt)) < epsilon);
  }
  REQUIRE(status == true);
}

TEST_CASE("Second-order shape function for WT scheme")
{
  const int     N        = 100;
  const float64 epsilon  = 1.0e-14;
  const float64 xmin     = -0.5;
  const float64 xmax     = +0.5;
  const float64 delx     = xmax - xmin;
  const float64 xmid     = 0;
  const float64 xeval[3] = {xmid - delx, xmid, xmid + delx};
  const float64 delt     = GENERATE(0.1, 0.2, 0.3, 0.4, 0.5);

  // analytic form
  auto W = [](float64 x, float64 dt) {
    float64 abs_x = std::abs(x);
    if (1 - dt < abs_x && abs_x <= 1 + dt) {
      return (dt + 1 - abs_x) * (dt + 1 - abs_x) / (4 * dt);
    } else if (dt < abs_x && abs_x <= 1 - dt) {
      return 1.0 - abs_x;
    } else if (abs_x <= dt) {
      return (2 * dt - dt * dt - abs_x * abs_x) / (2 * dt);
    } else {
      return 0.0;
    }
  };

  // test
  bool status = true;
  for (int i = 0; i < N; i++) {
    float64 s[3];
    float64 x = xmin + (xmax - xmin) * i / (N - 1);

    shape_wt<2>(x, xmid, 1 / delx, delt, 1 / delt, s);
    status = status & (std::abs(s[0] - W(xeval[0] - x, delt)) < epsilon);
    status = status & (std::abs(s[1] - W(xeval[1] - x, delt)) < epsilon);
    status = status & (std::abs(s[2] - W(xeval[2] - x, delt)) < epsilon);
  }
  REQUIRE(status == true);
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

TEST_CASE("Esirkepov scheme in 3D")
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

  set_random_particle(xu, delh, delv);

  //
  // first order
  //
  SECTION("First-order Esirkepov scheme for individual particles")
  {
    const int size    = 4;
    bool      status1 = true;
    bool      status2 = true;

    for (int ip = 0; ip < Np; ip++) {
      float64 cur[size][size][size][4] = {0};
      float64 rho[size][size][size]    = {0};

      float64* xv_ptr = &xv(ip, 0);
      float64* xu_ptr = &xu(ip, 0);

      status1 = status1 & test_esirkepov3d<1>(delt, delh, xu_ptr, xv_ptr, rho, cur, eps);
      status2 = status2 & check_charge_continuity(delt, delh, rho, cur, eps);
    }

    REQUIRE(status1); // charge density
    REQUIRE(status2); // charge continuity
  }
  SECTION("First-order Esirkepov scheme for group of particles")
  {
    const int size    = 4;
    bool      status1 = true;
    bool      status2 = true;

    float64 cur[size][size][size][4] = {0};
    float64 rho[size][size][size]    = {0};

    for (int ip = 0; ip < Np; ip++) {
      float64* xv_ptr = &xv(ip, 0);
      float64* xu_ptr = &xu(ip, 0);

      status1 = status1 & test_esirkepov3d<1>(delt, delh, xu_ptr, xv_ptr, rho, cur, eps);
    }
    status2 = status2 & check_charge_continuity(delt, delh, rho, cur, eps);

    REQUIRE(status1); // charge density
    REQUIRE(status2); // charge continuity
  }
  //
  // second order
  //
  SECTION("Second-order Esirkepov scheme for individual particles")
  {
    const int size    = 5;
    bool      status1 = true;
    bool      status2 = true;

    for (int ip = 0; ip < Np; ip++) {
      float64 cur[size][size][size][4] = {0};
      float64 rho[size][size][size]    = {0};

      float64* xv_ptr = &xv(ip, 0);
      float64* xu_ptr = &xu(ip, 0);

      status1 = status1 & test_esirkepov3d<2>(delt, delh, xu_ptr, xv_ptr, rho, cur, eps);
      status2 = status2 & check_charge_continuity(delt, delh, rho, cur, eps);
    }

    REQUIRE(status1); // charge density
    REQUIRE(status2); // charge continuity
  }
  SECTION("Second-order Esirkepov scheme for group of particles")
  {
    const int size    = 5;
    bool      status1 = true;
    bool      status2 = true;

    float64 cur[size][size][size][4] = {0};
    float64 rho[size][size][size]    = {0};

    for (int ip = 0; ip < Np; ip++) {
      float64* xv_ptr = &xv(ip, 0);
      float64* xu_ptr = &xu(ip, 0);

      status1 = status1 & test_esirkepov3d<2>(delt, delh, xu_ptr, xv_ptr, rho, cur, eps);
    }
    status2 = status2 & check_charge_continuity(delt, delh, rho, cur, eps);

    REQUIRE(status1); // charge density
    REQUIRE(status2); // charge continuity
  }
  //
  // third order
  //
  SECTION("Third-order Esirkepov scheme for individual particles")
  {
    const int size    = 6;
    bool      status1 = true;
    bool      status2 = true;

    for (int ip = 0; ip < Np; ip++) {
      float64 cur[size][size][size][4] = {0};
      float64 rho[size][size][size]    = {0};

      float64* xv_ptr = &xv(ip, 0);
      float64* xu_ptr = &xu(ip, 0);

      status1 = status1 & test_esirkepov3d<3>(delt, delh, xu_ptr, xv_ptr, rho, cur, eps);
      status2 = status2 & check_charge_continuity(delt, delh, rho, cur, eps);
    }

    REQUIRE(status1); // charge density
    REQUIRE(status2); // charge continuity
  }
  SECTION("Third-order Esirkepov scheme for group of particles")
  {
    const int size    = 6;
    bool      status1 = true;
    bool      status2 = true;

    float64 cur[size][size][size][4] = {0};
    float64 rho[size][size][size]    = {0};

    for (int ip = 0; ip < Np; ip++) {
      float64* xv_ptr = &xv(ip, 0);
      float64* xu_ptr = &xu(ip, 0);

      status1 = status1 & test_esirkepov3d<3>(delt, delh, xu_ptr, xv_ptr, rho, cur, eps);
    }
    status2 = status2 & check_charge_continuity(delt, delh, rho, cur, eps);

    REQUIRE(status1); // charge density
    REQUIRE(status2); // charge continuity
  }
  //
  // forth order
  //
  SECTION("Fourth-order Esirkepov scheme for individual particles")
  {
    const int size    = 7;
    bool      status1 = true;
    bool      status2 = true;

    for (int ip = 0; ip < Np; ip++) {
      float64 cur[size][size][size][4] = {0};
      float64 rho[size][size][size]    = {0};

      float64* xv_ptr = &xv(ip, 0);
      float64* xu_ptr = &xu(ip, 0);

      status1 = status1 & test_esirkepov3d<4>(delt, delh, xu_ptr, xv_ptr, rho, cur, eps);
      status2 = status2 & check_charge_continuity(delt, delh, rho, cur, eps);
    }

    REQUIRE(status1); // charge density
    REQUIRE(status2); // charge continuity
  }
  SECTION("Fourth-order Esirkepov scheme for group of particles")
  {
    const int size    = 7;
    bool      status1 = true;
    bool      status2 = true;

    float64 cur[size][size][size][4] = {0};
    float64 rho[size][size][size]    = {0};

    for (int ip = 0; ip < Np; ip++) {
      float64* xv_ptr = &xv(ip, 0);
      float64* xu_ptr = &xu(ip, 0);

      status1 = status1 & test_esirkepov3d<4>(delt, delh, xu_ptr, xv_ptr, rho, cur, eps);
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
