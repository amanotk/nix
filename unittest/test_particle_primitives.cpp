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

//
// forward declarations of helper functions
//

void set_random_particle(Array2D& xu, float64 delh, float64 delv);

template <int N>
bool test_charge_continuity(const float64 delt, const float64 delh, const float64 rho[N][N][N],
                            const float64 cur[N][N][N][4], const float64 epsilon);

template <int Order>
bool esirkepov3d_scalar(const float64 delt, const float64 delh, float64 xu[7], float64 xv[7],
                        float64 rho[Order + 3][Order + 3][Order + 3],
                        float64 cur[Order + 3][Order + 3][Order + 3][4], const float64 epsilon);

template <int Order, typename T_float>
bool esirkepov3d_xsimd(const T_float delt, const T_float delh, T_float xu[7], T_float xv[7],
                       T_float rho[Order + 3][Order + 3][Order + 3],
                       T_float cur[Order + 3][Order + 3][Order + 3][4], const float64 epsilon);

template <int Order, typename T_array>
bool test_esirkepov3d_scalar(T_array& xu, T_array& xv, const int Np, float64 delt, float64 delh,
                             const float64 epsilon);

template <int Order, typename T_array>
bool test_esirkepov3d_xsimd(T_array& xu, T_array& xv, const int Np, float64 delt, float64 delh,
                            const float64 epsilon);

template <int Order, typename T_array>
bool test_append_current3d_scalar(T_array& uj, int iz0, int iy0, int ix0, float64 q,
                                  const float64 epsilon);

template <int Order, typename T_array>
bool test_append_current3d_xsimd(T_array& uj, T_array& vj, int iz0, int iy0, int ix0, float64 q,
                                 const float64 epsilon);

template <int Order, typename T_array>
bool test_interpolate3d_scalar(T_array eb, int iz0, int iy0, int ix0, float64 delt,
                               float64 epsilon);

template <int Order, typename T_array, typename T_int>
bool test_interpolate3d_xsimd(T_array eb, T_int iz0, T_int iy0, T_int ix0, float64 delt,
                              float64 epsilon);

//
// test cases
//

TEST_CASE("digitize")
{
  const int     N    = 10;
  const float64 xmin = GENERATE(-1.0, 0.0, +.0);
  const float64 delx = GENERATE(0.5, 1.0, 1.5);

  std::random_device seed;
  std::mt19937_64    engine(seed());
  uniform_rand       rand(0, 1);

  SECTION("scalar")
  {
    bool status = true;
    for (int i = 0; i < N; i++) {
      float64 x = xmin + delx * i + rand(engine) * delx;
      int     j = digitize(x, xmin, 1 / delx);
      status    = status & (i == j);
    }

    REQUIRE(status == true);
  }
  SECTION("xsimd")
  {
    using simd::simd_f64;

    // initialize
    std::vector<float64> x(simd_f64::size);

    for (int i = 0; i < simd_f64::size; i++) {
      x[i] = xmin + delx * i + rand(engine) * delx;
    }

    // SIMD version
    simd_f64 x_simd    = xsimd::load_unaligned(x.data());
    simd_f64 xmin_simd = xmin;
    simd_f64 delx_simd = delx;
    auto     j_simd    = digitize(x_simd, xmin_simd, 1 / delx_simd);

    bool status = true;
    for (int i = 0; i < simd_f64::size; i++) {
      int j  = digitize(x[i], xmin, 1 / delx);
      status = status & (j == j_simd.get(i));
    }

    REQUIRE(status == true);
  }
}

TEST_CASE("sign")
{
  const float64 epsilon = 1.0e-14;

  SECTION("scalar")
  {
    bool status = true;
    status      = status & (std::abs(sign(+0.0) - 1) < epsilon);
    status      = status & (std::abs(sign(-0.0) + 1) < epsilon);
    status      = status & (std::abs(sign(+1.0) - 1) < epsilon);
    status      = status & (std::abs(sign(-1.0) + 1) < epsilon);

    REQUIRE(status == true);
  }
  SECTION("xsimd")
  {
    using simd::simd_f64;

    // initialize
    std::vector<float64> x(simd_f64::size);

    for (int i = 0; i < simd_f64::size; i++) {
      x[i] = (i % 2 == 0) ? +1.0 : -1.0;
    }

    // SIMD version
    simd_f64 x_simd = xsimd::load_unaligned(x.data());
    auto     s_simd = sign(x_simd);

    bool status = true;
    for (int i = 0; i < simd_f64::size; i++) {
      float64 s = sign(x[i]);
      status    = status & (std::abs(s - s_simd.get(i)) < epsilon);
    }

    REQUIRE(status == true);
  }
}

TEST_CASE("lorentz_factor")
{
  const int     N       = 10;
  const float64 epsilon = 1.0e-14;
  float64       cc      = GENERATE(0.5, 1.0, 2.0);

  std::random_device seed;
  std::mt19937_64    engine(seed());
  uniform_rand       rand(-1, +1);

  SECTION("scalar")
  {
    bool status = true;
    for (int i = 0; i < N; i++) {
      float64 ux = rand(engine);
      float64 uy = rand(engine);
      float64 uz = rand(engine);
      float64 gm = sqrt(1 + (ux * ux + uy * uy + uz * uz) / (cc * cc));
      status     = status & (std::abs(gm - lorentz_factor(ux, uy, uz, 1 / cc)) < epsilon);
    }

    REQUIRE(status == true);
  }
  SECTION("xsimd")
  {
    using simd::simd_f64;

    // initialize
    std::vector<float64> ux(simd_f64::size);
    std::vector<float64> uy(simd_f64::size);
    std::vector<float64> uz(simd_f64::size);

    for (int i = 0; i < simd_f64::size; i++) {
      ux[i] = rand(engine);
      uy[i] = rand(engine);
      uz[i] = rand(engine);
    }

    // SIMD version
    simd_f64 ux_simd = xsimd::load_unaligned(ux.data());
    simd_f64 uy_simd = xsimd::load_unaligned(uy.data());
    simd_f64 uz_simd = xsimd::load_unaligned(uz.data());
    simd_f64 cc_simd = cc;
    auto     gm_simd = lorentz_factor(ux_simd, uy_simd, uz_simd, 1 / cc_simd);

    bool status = true;
    for (int i = 0; i < simd_f64::size; i++) {
      float64 gm = lorentz_factor(ux[i], uy[i], uz[i], 1 / cc);
      status     = status & (std::abs(gm - gm_simd.get(i)) < epsilon);
    }

    REQUIRE(status == true);
  }
}

TEST_CASE("push_boris")
{
  const float64 epsilon = 1.0e-14;
  const float64 u0      = 1.0;
  const float64 e0      = GENERATE(0.25, 0.5, 1.0);
  const float64 b0      = GENERATE(0.25, 1.0, 4.0);

  std::random_device seed;
  std::mt19937_64    engine(seed());
  uniform_rand       rand(-1, 1);

  SECTION("scalar")
  {
    // How do we do this?
  }
  SECTION("xsimd")
  {
    using simd::simd_f64;

    std::vector<float64> ux(simd_f64::size);
    std::vector<float64> uy(simd_f64::size);
    std::vector<float64> uz(simd_f64::size);
    std::vector<float64> ex(simd_f64::size);
    std::vector<float64> ey(simd_f64::size);
    std::vector<float64> ez(simd_f64::size);
    std::vector<float64> bx(simd_f64::size);
    std::vector<float64> by(simd_f64::size);
    std::vector<float64> bz(simd_f64::size);

    for (int i = 0; i < simd_f64::size; i++) {
      ux[i] = u0 * rand(engine);
      uy[i] = u0 * rand(engine);
      uz[i] = u0 * rand(engine);
      ex[i] = e0 * rand(engine);
      ey[i] = e0 * rand(engine);
      ez[i] = e0 * rand(engine);
      bx[i] = b0 * rand(engine);
      by[i] = b0 * rand(engine);
      bz[i] = b0 * rand(engine);
    }

    // SIMD version
    simd_f64 ux_simd = xsimd::load_unaligned(ux.data());
    simd_f64 uy_simd = xsimd::load_unaligned(uy.data());
    simd_f64 uz_simd = xsimd::load_unaligned(uz.data());
    simd_f64 ex_simd = xsimd::load_unaligned(ex.data());
    simd_f64 ey_simd = xsimd::load_unaligned(ey.data());
    simd_f64 ez_simd = xsimd::load_unaligned(ez.data());
    simd_f64 bx_simd = xsimd::load_unaligned(bx.data());
    simd_f64 by_simd = xsimd::load_unaligned(by.data());
    simd_f64 bz_simd = xsimd::load_unaligned(bz.data());

    push_boris(ux_simd, uy_simd, uz_simd, ex_simd, ey_simd, ez_simd, bx_simd, by_simd, bz_simd);

    bool status = true;
    for (int i = 0; i < simd_f64::size; i++) {
      push_boris(ux[i], uy[i], uz[i], ex[i], ey[i], ez[i], bx[i], by[i], bz[i]);
      status = status & (std::abs(ux[i] - ux_simd.get(i)) < epsilon);
      status = status & (std::abs(uy[i] - uy_simd.get(i)) < epsilon);
      status = status & (std::abs(uz[i] - uz_simd.get(i)) < epsilon);
    }

    REQUIRE(status == true);
  }
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

  SECTION("scalar")
  {
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
  SECTION("xsimd")
  {
    using simd::simd_f64;

    // initialize
    std::vector<float64> x(simd_f64::size);

    for (int i = 0; i < simd_f64::size; i++) {
      x[i] = xmin + (xmax - xmin) * i / (simd_f64::size - 1);
    }

    // SIMD version
    simd_f64 x_simd = xsimd::load_unaligned(x.data());
    simd_f64 s_simd[2];
    simd_f64 xmin_simd = xmin;
    simd_f64 delx_simd = delx;

    shape<1>(x_simd, xmin_simd, 1 / delx_simd, s_simd);

    bool status = true;
    for (int i = 0; i < simd_f64::size; i++) {
      float64 s[2];
      shape<1>(x[i], xmin, 1 / delx, s);
      status = status & (std::abs(s[0] - s_simd[0].get(i)) < epsilon);
      status = status & (std::abs(s[1] - s_simd[1].get(i)) < epsilon);
    }

    REQUIRE(status == true);
  }
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

  SECTION("scalar")
  {
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
  SECTION("xsimd")
  {
    using simd::simd_f64;

    // initialize
    std::vector<float64> x(simd_f64::size);

    for (int i = 0; i < simd_f64::size; i++) {
      x[i] = xmin + (xmax - xmin) * i / (simd_f64::size - 1);
    }

    // SIMD version
    simd_f64 x_simd = xsimd::load_unaligned(x.data());
    simd_f64 s_simd[3];
    simd_f64 xmin_simd = xmin;
    simd_f64 delx_simd = delx;

    shape<2>(x_simd, xmin_simd, 1 / delx_simd, s_simd);

    bool status = true;
    for (int i = 0; i < simd_f64::size; i++) {
      float64 s[3];
      shape<2>(x[i], xmin, 1 / delx, s);
      status = status & (std::abs(s[0] - s_simd[0].get(i)) < epsilon);
      status = status & (std::abs(s[1] - s_simd[1].get(i)) < epsilon);
      status = status & (std::abs(s[2] - s_simd[2].get(i)) < epsilon);
    }

    REQUIRE(status == true);
  }
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

  SECTION("scalar")
  {
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
  SECTION("xsimd")
  {
    using simd::simd_f64;

    // initialize
    std::vector<float64> x(simd_f64::size);

    for (int i = 0; i < simd_f64::size; i++) {
      x[i] = xmin + (xmax - xmin) * i / (simd_f64::size - 1);
    }

    // SIMD version
    simd_f64 x_simd = xsimd::load_unaligned(x.data());
    simd_f64 s_simd[4];
    simd_f64 xmin_simd = xmin;
    simd_f64 delx_simd = delx;

    shape<3>(x_simd, xmin_simd, 1 / delx_simd, s_simd);

    bool status = true;
    for (int i = 0; i < simd_f64::size; i++) {
      float64 s[4];
      shape<3>(x[i], xmin, 1 / delx, s);
      status = status & (std::abs(s[0] - s_simd[0].get(i)) < epsilon);
      status = status & (std::abs(s[1] - s_simd[1].get(i)) < epsilon);
      status = status & (std::abs(s[2] - s_simd[2].get(i)) < epsilon);
      status = status & (std::abs(s[3] - s_simd[3].get(i)) < epsilon);
    }

    REQUIRE(status == true);
  }
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

  SECTION("scalar")
  {
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
  SECTION("xsimd")
  {
    using simd::simd_f64;

    // initialize
    std::vector<float64> x(simd_f64::size);

    for (int i = 0; i < simd_f64::size; i++) {
      x[i] = xmin + (xmax - xmin) * i / (simd_f64::size - 1);
    }

    // SIMD version
    simd_f64 x_simd = xsimd::load_unaligned(x.data());
    simd_f64 s_simd[5];
    simd_f64 xmin_simd = xmin;
    simd_f64 delx_simd = delx;

    shape<4>(x_simd, xmin_simd, 1 / delx_simd, s_simd);

    bool status = true;
    for (int i = 0; i < simd_f64::size; i++) {
      float64 s[5];
      shape<4>(x[i], xmin, 1 / delx, s);
      status = status & (std::abs(s[0] - s_simd[0].get(i)) < epsilon);
      status = status & (std::abs(s[1] - s_simd[1].get(i)) < epsilon);
      status = status & (std::abs(s[2] - s_simd[2].get(i)) < epsilon);
      status = status & (std::abs(s[3] - s_simd[3].get(i)) < epsilon);
      status = status & (std::abs(s[4] - s_simd[4].get(i)) < epsilon);
    }

    REQUIRE(status == true);
  }
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

  SECTION("scalar")
  {
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
  SECTION("xsimd")
  {
    using simd::simd_f64;

    // initialize
    std::vector<float64> x(simd_f64::size);

    for (int i = 0; i < simd_f64::size; i++) {
      x[i] = xmin + (xmax - xmin) * i / (simd_f64::size - 1);
    }

    // SIMD version
    simd_f64 x_simd = xsimd::load_unaligned(x.data());
    simd_f64 s_simd[2];
    simd_f64 xmin_simd = xmin;
    simd_f64 delx_simd = delx;
    simd_f64 delt_simd = delt;

    shape_wt<1>(x_simd, xmin_simd, 1 / delx_simd, delt_simd, 1 / delt_simd, s_simd);

    bool status = true;
    for (int i = 0; i < simd_f64::size; i++) {
      float64 s[2];
      shape_wt<1>(x[i], xmin, 1 / delx, delt, 1 / delt, s);
      status = status & (std::abs(s[0] - s_simd[0].get(i)) < epsilon);
      status = status & (std::abs(s[1] - s_simd[1].get(i)) < epsilon);
    }

    REQUIRE(status == true);
  }
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
      return std::pow(dt + 1 - abs_x, 2) / (4 * dt);
    } else if (dt < abs_x && abs_x <= 1 - dt) {
      return 1.0 - abs_x;
    } else if (abs_x <= dt) {
      return (2 * dt - dt * dt - abs_x * abs_x) / (2 * dt);
    } else {
      return 0.0;
    }
  };

  SECTION("scalar")
  {
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
  SECTION("xsimd")
  {
    using simd::simd_f64;

    // initialize
    std::vector<float64> x(simd_f64::size);

    for (int i = 0; i < simd_f64::size; i++) {
      x[i] = xmin + (xmax - xmin) * i / (simd_f64::size - 1);
    }

    // SIMD version
    simd_f64 x_simd = xsimd::load_unaligned(x.data());
    simd_f64 s_simd[3];
    simd_f64 xmin_simd = xmin;
    simd_f64 delx_simd = delx;
    simd_f64 delt_simd = delt;

    shape_wt<2>(x_simd, xmin_simd, 1 / delx_simd, delt_simd, 1 / delt_simd, s_simd);

    bool status = true;
    for (int i = 0; i < simd_f64::size; i++) {
      float64 s[3];
      shape_wt<2>(x[i], xmin, 1 / delx, delt, 1 / delt, s);
      status = status & (std::abs(s[0] - s_simd[0].get(i)) < epsilon);
      status = status & (std::abs(s[1] - s_simd[1].get(i)) < epsilon);
      status = status & (std::abs(s[2] - s_simd[2].get(i)) < epsilon);
    }

    REQUIRE(status == true);
  }
}

TEST_CASE("Third-order shape function for WT scheme")
{
  const int     N        = 100;
  const float64 epsilon  = 1.0e-14;
  const float64 xmin     = 0;
  const float64 xmax     = 1;
  const float64 delx     = xmax - xmin;
  const float64 xeval[4] = {xmin - delx, xmin, xmax, xmax + delx};
  const float64 delt     = GENERATE(0.1, 0.2, 0.3, 0.4, 0.5);

  // analytic form
  auto W = [](float64 x, float64 dt) {
    float64 abs_x = std::abs(x);
    if (1.5 - dt < abs_x && abs_x <= 1.5 + dt) {
      return std::pow(3 + 2 * dt - 2 * abs_x, 3) / (96 * dt);
    } else if (0.5 + dt < abs_x && abs_x <= 1.5 - dt) {
      return (4 * dt * dt + 3 * std::pow(3 - 2 * abs_x, 2)) / 24;
    } else if (0.5 - dt < abs_x && abs_x <= 0.5 + dt) {
      return (-8 * dt * dt * dt - 36 * dt * dt * (1 - 2 * abs_x) - 3 * std::pow(1 - 2 * abs_x, 3) +
              6 * dt * (15 - 12 * abs_x - 4 * x * x)) /
             (96 * dt);
    } else if (abs_x <= 0.5 - dt) {
      return (9 - 4 * dt * dt - 12 * x * x) / 12;
    } else {
      return 0.0;
    }
  };

  SECTION("scalar")
  {
    bool status = true;
    for (int i = 0; i < N; i++) {
      float64 s[4];
      float64 x = xmin + (xmax - xmin) * i / (N - 1);

      shape_wt<3>(x, xmin, 1 / delx, delt, 1 / delt, s);
      status = status & (std::abs(s[0] - W(xeval[0] - x, delt)) < epsilon);
      status = status & (std::abs(s[1] - W(xeval[1] - x, delt)) < epsilon);
      status = status & (std::abs(s[2] - W(xeval[2] - x, delt)) < epsilon);
      status = status & (std::abs(s[3] - W(xeval[3] - x, delt)) < epsilon);
    }

    REQUIRE(status == true);
  }
  SECTION("xsimd")
  {
    using simd::simd_f64;

    // initialize
    std::vector<float64> x(simd_f64::size);

    for (int i = 0; i < simd_f64::size; i++) {
      x[i] = xmin + (xmax - xmin) * i / (simd_f64::size - 1);
    }

    // SIMD version
    simd_f64 x_simd = xsimd::load_unaligned(x.data());
    simd_f64 s_simd[4];
    simd_f64 xmin_simd = xmin;
    simd_f64 delx_simd = delx;
    simd_f64 delt_simd = delt;

    shape_wt<3>(x_simd, xmin_simd, 1 / delx_simd, delt_simd, 1 / delt_simd, s_simd);

    bool status = true;
    for (int i = 0; i < simd_f64::size; i++) {
      float64 s[4];
      shape_wt<3>(x[i], xmin, 1 / delx, delt, 1 / delt, s);
      status = status & (std::abs(s[0] - s_simd[0].get(i)) < epsilon);
      status = status & (std::abs(s[1] - s_simd[1].get(i)) < epsilon);
      status = status & (std::abs(s[2] - s_simd[2].get(i)) < epsilon);
      status = status & (std::abs(s[3] - s_simd[3].get(i)) < epsilon);
    }

    REQUIRE(status == true);
  }
}

TEST_CASE("Fourth-order shape function for WT scheme")
{
  const int     N        = 100;
  const float64 epsilon  = 1.0e-14;
  const float64 xmin     = -0.5;
  const float64 xmax     = +0.5;
  const float64 delx     = xmax - xmin;
  const float64 xmid     = 0;
  const float64 xeval[5] = {xmid - 2 * delx, xmid - delx, xmid, xmid + delx, xmid + 2 * delx};
  const float64 delt     = GENERATE(0.1, 0.2, 0.3, 0.4, 0.5);

  // analytic form
  auto W = [](float64 x, float64 dt) {
    float64 abs_x = std::abs(x);
    if (2 - dt < abs_x && abs_x <= 2 + dt) {
      return std::pow(dt + 2 - abs_x, 4) / (48 * dt);
    } else if (1 + dt < abs_x && abs_x <= 2 - dt) {
      return (2 - abs_x) * (std::pow(2 - abs_x, 2) + dt * dt) / 6;
    } else if (1 - dt < abs_x && abs_x <= 1 + dt) {
      return (-std::pow(1 - abs_x, 4) + 2 * dt * (6 - 6 * abs_x + std::pow(abs_x, 3)) -
              6 * dt * dt * std::pow(1 - abs_x, 2) + 2 * dt * dt * dt * abs_x - dt * dt * dt * dt) /
             (12 * dt);
    } else if (dt < abs_x && abs_x <= 1 - dt) {
      return (4 - 6 * x * x + 3 * std::pow(abs_x, 3) - dt * dt * (2 - 3 * abs_x)) / 6;
    } else if (abs_x <= dt) {
      return (3 * std::pow(x, 4) + dt * (16 - 24 * x * x) + 18 * dt * dt * x * x -
              8 * dt * dt * dt + 3 * dt * dt * dt * dt) /
             (24 * dt);
    } else {
      return 0.0;
    }
  };

  SECTION("scalar")
  {
    bool status = true;
    for (int i = 0; i < N; i++) {
      float64 s[5];
      float64 x = xmin + (xmax - xmin) * i / (N - 1);

      shape_wt<4>(x, xmid, 1 / delx, delt, 1 / delt, s);
      status = status & (std::abs(s[0] - W(xeval[0] - x, delt)) < epsilon);
      status = status & (std::abs(s[1] - W(xeval[1] - x, delt)) < epsilon);
      status = status & (std::abs(s[2] - W(xeval[2] - x, delt)) < epsilon);
      status = status & (std::abs(s[3] - W(xeval[3] - x, delt)) < epsilon);
      status = status & (std::abs(s[4] - W(xeval[4] - x, delt)) < epsilon);
    }

    REQUIRE(status == true);
  }
  SECTION("xsimd")
  {
    using simd::simd_f64;

    // initialize
    std::vector<float64> x(simd_f64::size);

    for (int i = 0; i < simd_f64::size; i++) {
      x[i] = xmin + (xmax - xmin) * i / (simd_f64::size - 1);
    }

    // SIMD version
    simd_f64 x_simd = xsimd::load_unaligned(x.data());
    simd_f64 s_simd[5];
    simd_f64 xmin_simd = xmin;
    simd_f64 delx_simd = delx;
    simd_f64 delt_simd = delt;

    shape_wt<4>(x_simd, xmin_simd, 1 / delx_simd, delt_simd, 1 / delt_simd, s_simd);

    bool status = true;
    for (int i = 0; i < simd_f64::size; i++) {
      float64 s[5];
      shape_wt<4>(x[i], xmin, 1 / delx, delt, 1 / delt, s);
      status = status & (std::abs(s[0] - s_simd[0].get(i)) < epsilon);
      status = status & (std::abs(s[1] - s_simd[1].get(i)) < epsilon);
      status = status & (std::abs(s[2] - s_simd[2].get(i)) < epsilon);
      status = status & (std::abs(s[3] - s_simd[3].get(i)) < epsilon);
      status = status & (std::abs(s[4] - s_simd[4].get(i)) < epsilon);
    }

    REQUIRE(status == true);
  }
}

TEST_CASE("Esirkepov scheme in 3D")
{
  const int     Np   = 1024;
  const float64 delt = GENERATE(0.5, 1.0, 2.0);
  const float64 delh = GENERATE(0.5, 1.0, 2.0);
  const float64 delv = delh / delt;
  const float64 eps  = 1.0e-13;

  float64 xv_data[Np * 7];
  float64 xu_data[Np * 7];
  auto    xv = stdex::mdspan(xv_data, Np, 7);
  auto    xu = stdex::mdspan(xu_data, Np, 7);

  set_random_particle(xu, delh, delv);

  //
  // first order
  //
  SECTION("First-order Esirkepov scheme : scalar")
  {
    REQUIRE(test_esirkepov3d_scalar<1>(xu, xv, Np, delt, delh, eps) == true);
  }
  SECTION("First-order Esirkepov scheme : xsimd")
  {
    REQUIRE(test_esirkepov3d_xsimd<1>(xu, xv, Np, delt, delh, eps) == true);
  }
  //
  // second order
  //
  SECTION("Second-order Esirkepov scheme : scalar")
  {
    REQUIRE(test_esirkepov3d_scalar<2>(xu, xv, Np, delt, delh, eps) == true);
  }
  SECTION("Second-order Esirkepov scheme : xsimd")
  {
    REQUIRE(test_esirkepov3d_xsimd<2>(xu, xv, Np, delt, delh, eps) == true);
  }
  //
  // third order
  //
  SECTION("Third-order Esirkepov scheme : scalar")
  {
    REQUIRE(test_esirkepov3d_scalar<3>(xu, xv, Np, delt, delh, eps) == true);
  }
  SECTION("Third-order Esirkepov scheme : xsimd")
  {
    REQUIRE(test_esirkepov3d_xsimd<3>(xu, xv, Np, delt, delh, eps) == true);
  }
  //
  // forth order
  //
  SECTION("Fourth-order Esirkepov scheme : scalar")
  {
    REQUIRE(test_esirkepov3d_scalar<4>(xu, xv, Np, delt, delh, eps) == true);
  }
  SECTION("Fourth-order Esirkepov scheme : xsimd")
  {
    REQUIRE(test_esirkepov3d_xsimd<4>(xu, xv, Np, delt, delh, eps) == true);
  }
}

TEST_CASE("Current append to global array 3D")
{
  const int     Nz  = 16;
  const int     Ny  = 16;
  const int     Nx  = 16;
  const float64 eps = 1.0e-14;
  const float64 q   = 1.0;

  std::vector<float64> uj_data1(Nz * Ny * Nx * 4);
  std::vector<float64> uj_data2(Nz * Ny * Nx * 4);
  auto                 uj1 = stdex::mdspan(uj_data1.data(), Nz, Ny, Nx, 4);
  auto                 uj2 = stdex::mdspan(uj_data2.data(), Nz, Ny, Nx, 4);

  //
  // first order
  //
  SECTION("First-order current append to global array : scalar")
  {
    REQUIRE(test_append_current3d_scalar<1>(uj1, 2, 2, 2, q, eps) == true);
    REQUIRE(test_append_current3d_scalar<1>(uj2, 2, 4, 8, q, eps) == true);
  }
  SECTION("First-order current append to global array : xsimd with scalar index")
  {
    REQUIRE(test_append_current3d_xsimd<1>(uj1, uj2, 2, 4, 8, q, eps) == true);
  }
  //
  // second order
  //
  SECTION("Second-order current append to global array : scalar")
  {
    REQUIRE(test_append_current3d_scalar<2>(uj1, 2, 2, 2, q, eps) == true);
    REQUIRE(test_append_current3d_scalar<2>(uj2, 2, 4, 8, q, eps) == true);
  }
  SECTION("Second-order current append to global array : xsimd with scalar index")
  {
    REQUIRE(test_append_current3d_xsimd<2>(uj1, uj2, 2, 4, 8, q, eps) == true);
  }
  //
  // third order
  //
  SECTION("Third-order current append to global array : scalar")
  {
    REQUIRE(test_append_current3d_scalar<3>(uj1, 2, 2, 2, q, eps) == true);
    REQUIRE(test_append_current3d_scalar<3>(uj2, 2, 4, 8, q, eps) == true);
  }
  SECTION("Third-order current append to global array : xsimd with scalar index")
  {
    REQUIRE(test_append_current3d_xsimd<3>(uj1, uj2, 2, 4, 8, q, eps) == true);
  }
  //
  // fourth order
  //
  SECTION("Fourth-order current append to global array : scalar")
  {
    REQUIRE(test_append_current3d_scalar<4>(uj1, 2, 2, 2, q, eps) == true);
    REQUIRE(test_append_current3d_scalar<4>(uj2, 2, 4, 8, q, eps) == true);
  }
  SECTION("Fourth-order current append to global array : xsimd with scalar index")
  {
    REQUIRE(test_append_current3d_xsimd<4>(uj1, uj2, 2, 4, 8, q, eps) == true);
  }
}

TEST_CASE("Interpolation 3D")
{
  const int     Nz   = 16;
  const int     Ny   = 16;
  const int     Nx   = 16;
  const float64 delt = 0.5;
  const float64 eps  = 1.0e-14;

  std::random_device seed;
  std::mt19937_64    engine(seed());
  uniform_rand       rand(0.0, +1.0);

  // field array
  std::vector<float64> eb_data(Nz * Ny * Nx * 6);
  auto                 eb = stdex::mdspan(eb_data.data(), Nz, Ny, Nx, 6);
  std::transform(eb_data.begin(), eb_data.end(), eb_data.begin(),
                 [&](float64 x) { return rand(engine); });

  // vector index
  std::vector<int64> iz0_data = {2, 3, 4, 5, 2, 3, 4, 5};
  std::vector<int64> iy0_data = {2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<int64> ix0_data = {5, 4, 3, 2, 5, 4, 3, 2};

  //
  // first order
  //
  SECTION("First-order interpolation : scalar")
  {
    REQUIRE(test_interpolate3d_scalar<1>(eb, 2, 2, 2, delt, eps) == true);
    REQUIRE(test_interpolate3d_scalar<1>(eb, 2, 4, 8, delt, eps) == true);
  }
  SECTION("First-order interpolation : xsimd")
  {
    // scalar index
    REQUIRE(test_interpolate3d_xsimd<1>(eb, 2, 2, 2, delt, eps) == true);
    REQUIRE(test_interpolate3d_xsimd<1>(eb, 2, 4, 8, delt, eps) == true);

    // vector index
    using simd::simd_i64;
    simd_i64 iz0 = xsimd::load_unaligned(iz0_data.data());
    simd_i64 iy0 = xsimd::load_unaligned(iy0_data.data());
    simd_i64 ix0 = xsimd::load_unaligned(ix0_data.data());
    REQUIRE(test_interpolate3d_xsimd<1>(eb, iz0, iy0, ix0, delt, eps) == true);
  }
  //
  // second order
  //
  SECTION("Second-order interpolation : scalar")
  {
    REQUIRE(test_interpolate3d_scalar<2>(eb, 2, 2, 2, delt, eps) == true);
    REQUIRE(test_interpolate3d_scalar<2>(eb, 2, 4, 8, delt, eps) == true);
  }
  SECTION("Second-order interpolation : xsimd")
  {
    // scalar index
    REQUIRE(test_interpolate3d_xsimd<2>(eb, 2, 2, 2, delt, eps) == true);
    REQUIRE(test_interpolate3d_xsimd<2>(eb, 2, 4, 8, delt, eps) == true);

    // vector index
    using simd::simd_i64;
    simd_i64 iz0 = xsimd::load_unaligned(iz0_data.data());
    simd_i64 iy0 = xsimd::load_unaligned(iy0_data.data());
    simd_i64 ix0 = xsimd::load_unaligned(ix0_data.data());
    REQUIRE(test_interpolate3d_xsimd<2>(eb, iz0, iy0, ix0, delt, eps) == true);
  }
  //
  // third order
  //
  SECTION("Third-order interpolation : scalar")
  {
    REQUIRE(test_interpolate3d_scalar<3>(eb, 2, 2, 2, delt, eps) == true);
    REQUIRE(test_interpolate3d_scalar<3>(eb, 2, 4, 8, delt, eps) == true);
  }
  SECTION("Third-order interpolation : xsimd")
  {
    // scalar index
    REQUIRE(test_interpolate3d_xsimd<3>(eb, 2, 2, 2, delt, eps) == true);
    REQUIRE(test_interpolate3d_xsimd<3>(eb, 2, 4, 8, delt, eps) == true);

    // vector index
    using simd::simd_i64;
    simd_i64 iz0 = xsimd::load_unaligned(iz0_data.data());
    simd_i64 iy0 = xsimd::load_unaligned(iy0_data.data());
    simd_i64 ix0 = xsimd::load_unaligned(ix0_data.data());
    REQUIRE(test_interpolate3d_xsimd<3>(eb, iz0, iy0, ix0, delt, eps) == true);
  }
  //
  // fourth order
  //
  SECTION("Fourth-order interpolation : scalar")
  {
    REQUIRE(test_interpolate3d_scalar<4>(eb, 2, 2, 2, delt, eps) == true);
    REQUIRE(test_interpolate3d_scalar<4>(eb, 2, 4, 8, delt, eps) == true);
  }
  SECTION("Fourth-order interpolation : xsimd")
  {
    // scalar index
    REQUIRE(test_interpolate3d_xsimd<4>(eb, 2, 2, 2, delt, eps) == true);
    REQUIRE(test_interpolate3d_xsimd<4>(eb, 2, 4, 8, delt, eps) == true);

    // vector index
    using simd::simd_i64;
    simd_i64 iz0 = xsimd::load_unaligned(iz0_data.data());
    simd_i64 iy0 = xsimd::load_unaligned(iy0_data.data());
    simd_i64 ix0 = xsimd::load_unaligned(ix0_data.data());
    REQUIRE(test_interpolate3d_xsimd<4>(eb, iz0, iy0, ix0, delt, eps) == true);
  }
}

//
// implementation of helper functions
//

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
bool test_charge_continuity(const float64 delt, const float64 delh, const float64 rho[N][N][N],
                            const float64 cur[N][N][N][4], const float64 epsilon)
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

template <int Order>
bool esirkepov3d_scalar(const float64 delt, const float64 delh, float64 xu[7], float64 xv[7],
                        float64 rho[Order + 3][Order + 3][Order + 3],
                        float64 cur[Order + 3][Order + 3][Order + 3][4], const float64 epsilon)
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
  xu[2] = xu[2] + xu[5] * delt;

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

template <int Order, typename T_float>
bool esirkepov3d_xsimd(const T_float delt, const T_float delh, T_float xu[7], T_float xv[7],
                       T_float rho[Order + 3][Order + 3][Order + 3],
                       T_float cur[Order + 3][Order + 3][Order + 3][4], const float64 epsilon)
{
  const T_float zero = 0;
  const T_float rdh  = 1 / delh;
  const T_float dhdt = delh / delt;

  bool    status  = true;
  T_float rhosum0 = 0;
  T_float rhosum1 = 0;
  T_float rhosum2 = 0;

  T_float ss[2][3][Order + 3] = {0};

  xv[0] = xu[0];
  xv[1] = xu[1];
  xv[2] = xu[2];
  xu[0] = xu[0] + xu[3] * delt;
  xu[1] = xu[1] + xu[4] * delt;
  xu[2] = xu[2] + xu[5] * delt;

  //
  // before move
  //
  auto ix0 = digitize(xv[0], zero, rdh);
  auto iy0 = digitize(xv[1], zero, rdh);
  auto iz0 = digitize(xv[2], zero, rdh);

  shape<Order>(xv[0], xsimd::to_float(ix0) * delh, rdh, &ss[0][0][1]);
  shape<Order>(xv[1], xsimd::to_float(iy0) * delh, rdh, &ss[0][1][1]);
  shape<Order>(xv[2], xsimd::to_float(iz0) * delh, rdh, &ss[0][2][1]);

  // check charge density
  for (int jz = 0; jz < Order + 3; jz++) {
    for (int jy = 0; jy < Order + 3; jy++) {
      for (int jx = 0; jx < Order + 3; jx++) {
        T_float r = ss[0][0][jx] * ss[0][1][jy] * ss[0][2][jz];
        rhosum0 += cur[jz][jy][jx][0];
        rhosum1 += r;
        rho[jz][jy][jx] += r;
      }
    }
  }

  //
  // after move
  //
  auto ix1 = digitize(xu[0], zero, rdh);
  auto iy1 = digitize(xu[1], zero, rdh);
  auto iz1 = digitize(xu[2], zero, rdh);

  shape<Order>(xu[0], xsimd::to_float(ix1) * delh, rdh, &ss[1][0][1]);
  shape<Order>(xu[1], xsimd::to_float(iy1) * delh, rdh, &ss[1][1][1]);
  shape<Order>(xu[2], xsimd::to_float(iz1) * delh, rdh, &ss[1][2][1]);

  //
  // in-place shift of ss[1] according to particle movement
  //
  xsimd::batch<int64_t> shift[3] = {ix1 - ix0, iy1 - iy0, iz1 - iz0};
  esirkepov3d_shift_weights_after_movement<Order>(shift, ss[1]);

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

  // error check
  {
    float64 rho0 = xsimd::reduce_add(rhosum0) / T_float::size;
    float64 rho1 = xsimd::reduce_add(rhosum1) / T_float::size;
    float64 rho2 = xsimd::reduce_add(rhosum2) / T_float::size;

    // contribution to charge density is normalized to unity
    status = status & (std::abs(rho1 - 1) < epsilon);

    // charge density increases exactly by one
    status = status & (std::abs(rho2 - (rho0 + 1)) < epsilon * std::abs(rho2));
  }

  return status;
}

template <int Order, typename T_array>
bool test_esirkepov3d_scalar(T_array& xu, T_array& xv, const int Np, float64 delt, float64 delh,
                             const float64 epsilon)
{
  const int size    = Order + 3;
  bool      status1 = true;
  bool      status2 = true;

  float64 cur[size][size][size][4] = {0};
  float64 rho[size][size][size]    = {0};

  for (int ip = 0; ip < Np; ip++) {
    float64* xu_ptr = &xu(ip, 0);
    float64* xv_ptr = &xv(ip, 0);

    status1 = status1 & esirkepov3d_scalar<Order>(delt, delh, xu_ptr, xv_ptr, rho, cur, epsilon);
  }
  status2 = status2 & test_charge_continuity(delt, delh, rho, cur, epsilon);

  return status1 & status2;
}

template <int Order, typename T_array>
bool test_esirkepov3d_xsimd(T_array& xu, T_array& xv, const int Np, float64 delt, float64 delh,
                            const float64 epsilon)
{
  using simd::simd_f64;
  using simd::simd_i64;

  const int size    = Order + 3;
  bool      status1 = true;
  bool      status2 = true;

  float64  cur[size][size][size][4]      = {0};
  float64  rho[size][size][size]         = {0};
  simd_f64 cur_simd[size][size][size][4] = {0};
  simd_f64 rho_simd[size][size][size]    = {0};
  simd_f64 delt_simd                     = delt;
  simd_f64 delh_simd                     = delh;
  simd_f64 xu_simd[7];
  simd_f64 xv_simd[7];
  simd_i64 index_simd = xsimd::detail::make_sequence_as_batch<simd_i64>() * 7;

  // SIMD version
  for (int ip = 0; ip < Np - Np % simd_f64::size; ip += simd_f64::size) {
    //  load data
    for (int k = 0; k < 7; k++) {
      xu_simd[k] = simd_f64::gather(&xu(ip, k), index_simd);
      xv_simd[k] = simd_f64::gather(&xv(ip, k), index_simd);
    }

    status1 = status1 & esirkepov3d_xsimd<Order>(delt_simd, delh_simd, xu_simd, xv_simd, rho_simd,
                                                 cur_simd, epsilon);
  }

  // scalar version
  for (int ip = 0; ip < Np - Np % simd_f64::size; ip++) {
    float64* xu_ptr = &xu(ip, 0);
    float64* xv_ptr = &xv(ip, 0);

    status1 = status1 & esirkepov3d_scalar<Order>(delt, delh, xu_ptr, xv_ptr, rho, cur, epsilon);
  }

  // compare scalar and SIMD results
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      for (int k = 0; k < size; k++) {
        for (int l = 0; l < 4; l++) {
          float64 cur_sum = xsimd::reduce_add(cur_simd[i][j][k][l]);
          float64 cur_err = std::abs(cur[i][j][k][l] - cur_sum);
          status2 = status2 & ((cur_err <= epsilon) || (cur_err <= std::abs(cur_sum) * epsilon));
        }
        float64 rho_sum = xsimd::reduce_add(rho_simd[i][j][k]);
        float64 rho_err = std::abs(rho[i][j][k] - rho_sum);
        status2 = status2 & ((rho_err <= epsilon) || (rho_err <= std::abs(rho_sum) * epsilon));
      }
    }
  }

  return status1 & status2;
}

template <int Order, typename T_array>
bool test_append_current3d_scalar(T_array& uj, int iz0, int iy0, int ix0, float64 q,
                                  const float64 epsilon)
{
  const int size = Order + 3;

  float64 cur[size][size][size][4] = {0};

  // test data
  for (int jz = 0; jz < size; jz++) {
    for (int jy = 0; jy < size; jy++) {
      for (int jx = 0; jx < size; jx++) {
        for (int k = 0; k < 4; k++) {
          cur[jz][jy][jx][k] = k + 1;
        }
      }
    }
  }

  // append
  append_current3d<Order>(uj, iz0, iy0, ix0, cur, q);

  // check
  bool status = true;
  for (int jz = 0, iz = iz0; jz < size; jz++, iz++) {
    for (int jy = 0, iy = iy0; jy < size; jy++, iy++) {
      for (int jx = 0, ix = ix0; jx < size; jx++, ix++) {
        for (int k = 0; k < 4; k++) {
          status = status & (std::abs(uj(iz, iy, ix, k) - q * (k + 1)) < epsilon);
        }
      }
    }
  }

  return status;
}

template <int Order, typename T_array>
bool test_append_current3d_xsimd(T_array& uj, T_array& vj, int iz0, int iy0, int ix0, float64 q,
                                 const float64 epsilon)
{
  using simd::simd_f64;
  using simd::simd_i64;

  const int size   = Order + 3;
  const int stride = size * size * size * 4;
  const int Nz     = uj.extent(0);
  const int Ny     = uj.extent(1);
  const int Nx     = uj.extent(2);

  float64  cur[simd_f64::size][size][size][size][4] = {0};
  simd_f64 cur_simd[size][size][size][4]            = {0};
  simd_i64 index_simd = xsimd::detail::make_sequence_as_batch<simd_i64>() * stride;

  std::random_device seed;
  std::mt19937_64    engine(seed());
  uniform_rand       rand(-1, +1);

  // test data
  for (int ip = 0; ip < simd_f64::size; ip++) {
    for (int jz = 0; jz < size; jz++) {
      for (int jy = 0; jy < size; jy++) {
        for (int jx = 0; jx < size; jx++) {
          for (int k = 0; k < 4; k++) {
            cur[ip][jz][jy][jx][k] = rand(engine);
          }
        }
      }
    }
  }
  for (int jz = 0; jz < size; jz++) {
    for (int jy = 0; jy < size; jy++) {
      for (int jx = 0; jx < size; jx++) {
        for (int k = 0; k < 4; k++) {
          cur_simd[jz][jy][jx][k] = simd_f64::gather(&cur[0][jz][jy][jx][k], index_simd);
        }
      }
    }
  }

  // scalar version
  for (int ip = 0; ip < simd_f64::size; ip++) {
    append_current3d<Order>(uj, iz0, iy0, ix0, cur[ip], q);
  }

  // SIMD version
  append_current3d<Order>(vj, iz0, iy0, ix0, cur_simd, q);

  // compare scalar and SIMD results
  bool status = true;
  for (int iz = 0; iz < Nz; iz++) {
    for (int iy = 0; iy < Ny; iy++) {
      for (int ix = 0; ix < Nx; ix++) {
        for (int k = 0; k < 4; k++) {
          status = status & (std::abs(uj(iz, iy, ix, k) - vj(iz, iy, ix, k) < epsilon));
        }
      }
    }
  }

  return status;
}

template <int Order, typename T_array>
bool test_interpolate3d_scalar(T_array eb, int iz0, int iy0, int ix0, float64 delt, float64 epsilon)
{
  constexpr int size = Order + 2;

  std::random_device seed;
  std::mt19937_64    engine(seed());
  uniform_rand       rand(0.0, +1.0);

  // test data
  std::vector<float64> result1(6);
  std::vector<float64> result2(6);
  std::vector<float64> wx_data(size);
  std::vector<float64> wy_data(size);
  std::vector<float64> wz_data(size);
  std::transform(wx_data.begin(), wx_data.end(), wx_data.begin(),
                 [&](float64) { return rand(engine); });
  std::transform(wy_data.begin(), wy_data.end(), wy_data.begin(),
                 [&](float64) { return rand(engine); });
  std::transform(wz_data.begin(), wz_data.end(), wz_data.begin(),
                 [&](float64) { return rand(engine); });

  // scalar version
  {
    float64* wx = wx_data.data();
    float64* wy = wy_data.data();
    float64* wz = wz_data.data();

    for (int ik = 0; ik < 6; ik++) {
      result1[ik] = interpolate3d<Order>(eb, iz0, iy0, ix0, ik, wz, wy, wx, delt);
    }
  }

  // naive calculation
  {
    float64* wx = wx_data.data();
    float64* wy = wy_data.data();
    float64* wz = wz_data.data();

    for (int jz = 0, iz = iz0; jz < size; jz++, iz++) {
      for (int jy = 0, iy = iy0; jy < size; jy++, iy++) {
        for (int jx = 0, ix = ix0; jx < size; jx++, ix++) {
          for (int ik = 0; ik < 6; ik++) {
            result2[ik] += eb(iz, iy, ix, ik) * wz[jz] * wy[jy] * wx[jx] * delt;
          }
        }
      }
    }
  }

  // compare results
  bool status = true;
  for (int i = 0; i < result1.size(); i++) {
    float64 err = std::abs(result1[i] - result2[i]);
    status      = status & (err <= std::abs(result1[1]) * epsilon);
  }

  return status;
}

template <int Order, typename T_array, typename T_int>
bool test_interpolate3d_xsimd(T_array eb, T_int iz0, T_int iy0, T_int ix0, float64 delt,
                              float64 epsilon)
{
  using simd::simd_f64;
  using simd::simd_i64;
  constexpr int size = Order + 2;
  static_assert(std::is_integral_v<T_int> || std::is_same_v<T_int, simd_i64>,
                "T_int must be either int or an appropriate SIMD type");

  // index for scalar version
  int iz[simd_f64::size];
  int iy[simd_f64::size];
  int ix[simd_f64::size];

  if constexpr (std::is_integral_v<T_int>) {
    // scalar index
    for (int i = 0; i < simd_f64::size; i++) {
      iz[i] = iz0;
      iy[i] = iy0;
      ix[i] = ix0;
    }
  } else if constexpr (std::is_same_v<T_int, simd_i64>) {
    // vector index
    iz0.store_unaligned(iz);
    iy0.store_unaligned(iy);
    ix0.store_unaligned(ix);
  }

  std::random_device seed;
  std::mt19937_64    engine(seed());
  uniform_rand       rand(0.0, +1.0);

  // test data
  std::vector<float64> result1(simd_f64::size * 6);
  std::vector<float64> result2(simd_f64::size * 6);
  std::vector<float64> wx_data(size * simd_f64::size);
  std::vector<float64> wy_data(size * simd_f64::size);
  std::vector<float64> wz_data(size * simd_f64::size);
  std::transform(wx_data.begin(), wx_data.end(), wx_data.begin(),
                 [&](float64) { return rand(engine); });
  std::transform(wy_data.begin(), wy_data.end(), wy_data.begin(),
                 [&](float64) { return rand(engine); });
  std::transform(wz_data.begin(), wz_data.end(), wz_data.begin(),
                 [&](float64) { return rand(engine); });

  // SIMD version
  {
    simd_f64 wx_simd[size];
    simd_f64 wy_simd[size];
    simd_f64 wz_simd[size];

    // load weights
    for (int i = 0; i < size; i++) {
      wx_simd[i] = xsimd::load_unaligned(wx_data.data() + i * simd_f64::size);
      wy_simd[i] = xsimd::load_unaligned(wy_data.data() + i * simd_f64::size);
      wz_simd[i] = xsimd::load_unaligned(wz_data.data() + i * simd_f64::size);
    }

    for (int ik = 0; ik < 6; ik++) {
      // interpolate
      simd_f64 val = interpolate3d<Order>(eb, iz0, iy0, ix0, ik, wz_simd, wy_simd, wx_simd, delt);
      // store
      val.store_unaligned(result1.data() + ik * simd_f64::size);
    }
  }

  // scalar version
  {
    float64 wx[size];
    float64 wy[size];
    float64 wz[size];

    for (int ik = 0; ik < 6; ik++) {
      for (int ip = 0; ip < simd_f64::size; ip++) {
        for (int i = 0; i < size; i++) {
          wx[i] = wx_data[i * simd_f64::size + ip];
          wy[i] = wy_data[i * simd_f64::size + ip];
          wz[i] = wz_data[i * simd_f64::size + ip];
        }
        // interpolate and store
        result2[ik * simd_f64::size + ip] =
            interpolate3d<Order>(eb, iz[ip], iy[ip], ix[ip], ik, wz, wy, wx, delt);
      }
    }
  }

  // compare SIMD and scalar results
  bool status = true;
  for (int i = 0; i < result1.size(); i++) {
    float64 err = std::abs(result1[i] - result2[i]);
    status      = status & (err <= std::abs(result1[1]) * epsilon);
  }

  return status;
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
