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
template <typename T>
using aligned_vector = std::vector<T, xsimd::aligned_allocator<T, 64>>;

//
// forward declarations of helper functions
//

template <int Order, typename T_array>
bool test_append_current3d_scalar(T_array& uj, int iz0, int iy0, int ix0, const float64 epsilon);

template <int Order, typename T_array, typename T_int>
bool test_append_current3d_xsimd(T_array& uj, T_array& vj, T_int iz0, T_int iy0, T_int ix0,
                                 const float64 epsilon);

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
    // initialize
    aligned_vector<float64> x(simd_f64::size);

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
    // initialize
    aligned_vector<float64> x(simd_f64::size);

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
    // initialize
    aligned_vector<float64> ux(simd_f64::size);
    aligned_vector<float64> uy(simd_f64::size);
    aligned_vector<float64> uz(simd_f64::size);

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
    aligned_vector<float64> ux(simd_f64::size);
    aligned_vector<float64> uy(simd_f64::size);
    aligned_vector<float64> uz(simd_f64::size);
    aligned_vector<float64> ex(simd_f64::size);
    aligned_vector<float64> ey(simd_f64::size);
    aligned_vector<float64> ez(simd_f64::size);
    aligned_vector<float64> bx(simd_f64::size);
    aligned_vector<float64> by(simd_f64::size);
    aligned_vector<float64> bz(simd_f64::size);

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
    // initialize
    aligned_vector<float64> x(simd_f64::size);

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
    // initialize
    aligned_vector<float64> x(simd_f64::size);

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
    // initialize
    aligned_vector<float64> x(simd_f64::size);

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
    // initialize
    aligned_vector<float64> x(simd_f64::size);

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
    // initialize
    aligned_vector<float64> x(simd_f64::size);

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
    // initialize
    aligned_vector<float64> x(simd_f64::size);

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
    // initialize
    aligned_vector<float64> x(simd_f64::size);

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
    // initialize
    aligned_vector<float64> x(simd_f64::size);

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

TEST_CASE("Current append to global array 3D")
{
  const int     Nz  = 16;
  const int     Ny  = 16;
  const int     Nx  = 16;
  const float64 eps = 1.0e-14;
  const float64 q   = 1.0;

  // current array
  aligned_vector<float64> uj_data1(Nz * Ny * Nx * 4);
  aligned_vector<float64> uj_data2(Nz * Ny * Nx * 4);
  auto                    uj1 = stdex::mdspan(uj_data1.data(), Nz, Ny, Nx, 4);
  auto                    uj2 = stdex::mdspan(uj_data2.data(), Nz, Ny, Nx, 4);

  // vector index
  aligned_vector<int64> iz0_data = {2, 3, 4, 5, 2, 3, 4, 5};
  aligned_vector<int64> iy0_data = {2, 2, 2, 2, 2, 2, 2, 2};
  aligned_vector<int64> ix0_data = {5, 4, 3, 2, 5, 4, 3, 2};

  //
  // first order
  //
  SECTION("First-order current append to global array : scalar")
  {
    REQUIRE(test_append_current3d_scalar<1>(uj1, 2, 2, 2, eps) == true);
    REQUIRE(test_append_current3d_scalar<1>(uj2, 2, 4, 8, eps) == true);
  }
  SECTION("First-order current append to global array : xsimd")
  {
    // scalar index
    REQUIRE(test_append_current3d_xsimd<1>(uj1, uj2, 2, 4, 8, eps) == true);

    // vector index
    simd_i64 iz0 = xsimd::load_unaligned(iz0_data.data());
    simd_i64 iy0 = xsimd::load_unaligned(iy0_data.data());
    simd_i64 ix0 = xsimd::load_unaligned(ix0_data.data());
    REQUIRE(test_append_current3d_xsimd<1>(uj1, uj2, iz0, iy0, ix0, eps) == true);
  }
  //
  // second order
  //
  SECTION("Second-order current append to global array : scalar")
  {
    REQUIRE(test_append_current3d_scalar<2>(uj1, 2, 2, 2, eps) == true);
    REQUIRE(test_append_current3d_scalar<2>(uj2, 2, 4, 8, eps) == true);
  }
  SECTION("Second-order current append to global array : xsimd")
  {
    // scalar index
    REQUIRE(test_append_current3d_xsimd<2>(uj1, uj2, 2, 4, 8, eps) == true);

    // vector index
    simd_i64 iz0 = xsimd::load_unaligned(iz0_data.data());
    simd_i64 iy0 = xsimd::load_unaligned(iy0_data.data());
    simd_i64 ix0 = xsimd::load_unaligned(ix0_data.data());
    REQUIRE(test_append_current3d_xsimd<2>(uj1, uj2, iz0, iy0, ix0, eps) == true);
  }
  //
  // third order
  //
  SECTION("Third-order current append to global array : scalar")
  {
    REQUIRE(test_append_current3d_scalar<3>(uj1, 2, 2, 2, eps) == true);
    REQUIRE(test_append_current3d_scalar<3>(uj2, 2, 4, 8, eps) == true);
  }
  SECTION("Third-order current append to global array : xsimd")
  {
    // scalar index
    REQUIRE(test_append_current3d_xsimd<3>(uj1, uj2, 2, 4, 8, eps) == true);

    // vector index
    simd_i64 iz0 = xsimd::load_unaligned(iz0_data.data());
    simd_i64 iy0 = xsimd::load_unaligned(iy0_data.data());
    simd_i64 ix0 = xsimd::load_unaligned(ix0_data.data());
    REQUIRE(test_append_current3d_xsimd<3>(uj1, uj2, iz0, iy0, ix0, eps) == true);
  }
  //
  // fourth order
  //
  SECTION("Fourth-order current append to global array : scalar")
  {
    REQUIRE(test_append_current3d_scalar<4>(uj1, 2, 2, 2, eps) == true);
    REQUIRE(test_append_current3d_scalar<4>(uj2, 2, 4, 8, eps) == true);
  }
  SECTION("Fourth-order current append to global array : xsimd")
  {
    // scalar index
    REQUIRE(test_append_current3d_xsimd<4>(uj1, uj2, 2, 4, 8, eps) == true);

    // vector index
    simd_i64 iz0 = xsimd::load_unaligned(iz0_data.data());
    simd_i64 iy0 = xsimd::load_unaligned(iy0_data.data());
    simd_i64 ix0 = xsimd::load_unaligned(ix0_data.data());
    REQUIRE(test_append_current3d_xsimd<4>(uj1, uj2, iz0, iy0, ix0, eps) == true);
  }
}

//
// implementation of helper functions
//

template <int Order, typename T_array>
bool test_append_current3d_scalar(T_array& uj, int iz0, int iy0, int ix0, const float64 epsilon)
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
  append_current3d<Order>(uj, iz0, iy0, ix0, cur);

  // check
  bool status = true;
  for (int jz = 0, iz = iz0; jz < size; jz++, iz++) {
    for (int jy = 0, iy = iy0; jy < size; jy++, iy++) {
      for (int jx = 0, ix = ix0; jx < size; jx++, ix++) {
        for (int k = 0; k < 4; k++) {
          status = status & (std::abs(uj(iz, iy, ix, k) - (k + 1)) < epsilon);
        }
      }
    }
  }

  return status;
}

template <int Order, typename T_array, typename T_int>
bool test_append_current3d_xsimd(T_array& uj, T_array& vj, T_int iz0, T_int iy0, T_int ix0,
                                 const float64 epsilon)
{
  const int size   = Order + 3;
  const int stride = size * size * size * 4;
  const int Nz     = uj.extent(0);
  const int Ny     = uj.extent(1);
  const int Nx     = uj.extent(2);

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

  // SIMD version
  append_current3d<Order>(vj, iz0, iy0, ix0, cur_simd);

  // scalar version
  for (int ip = 0; ip < simd_f64::size; ip++) {
    append_current3d<Order>(uj, iz[ip], iy[ip], ix[ip], cur[ip]);
  }

  // compare scalar and SIMD results
  bool status = true;
  for (int iz = 0; iz < Nz; iz++) {
    for (int iy = 0; iy < Ny; iy++) {
      for (int ix = 0; ix < Nx; ix++) {
        for (int k = 0; k < 4; k++) {
          status = status & (std::abs(uj(iz, iy, ix, k) - vj(iz, iy, ix, k)) < epsilon);
        }
      }
    }
  }

  return status;
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
