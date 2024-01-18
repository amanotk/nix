// -*- C++ -*-
#ifndef _PARTICLE_PRIMITIVES_HPP_
#define _PARTICLE_PRIMITIVES_HPP_

#include "nix.hpp"
#include "xsimd/xsimd.hpp"

NIX_NAMESPACE_BEGIN

namespace primitives
{

/// SIMD types
namespace simd
{
using simd_f32 = xsimd::batch<nix::typedefs::float32>;
using simd_f64 = xsimd::batch<nix::typedefs::float64>;
using simd_i32 = xsimd::batch<nix::typedefs::int32>;
using simd_i64 = xsimd::batch<nix::typedefs::int64>;
} // namespace simd

/// utility to get internal data pointer of array
template <typename T_array>
static auto get_data_pointer(T_array& array)
{
  if constexpr (is_xtensor<T_array>::value == true) {
    return array.data();
  } else if constexpr (is_mdspan<T_array>::value == true) {
    return array.data_handle();
  }
}

/// utility to get stride of array for a given dimension
template <typename T_array>
static auto get_stride(T_array& array, int dim)
{
  if constexpr (is_xtensor<T_array>::value == true) {
    return array.strides()[dim];
  } else if constexpr (is_mdspan<T_array>::value == true) {
    return array.stride(dim);
  }
}

/// digitize for calculating grid index for particles
template <typename T_float>
static auto digitize(T_float x, T_float xmin, T_float rdx)
{
  using namespace simd;
  constexpr bool is_scalar = std::is_floating_point_v<T_float>;
  constexpr bool is_vector = std::is_same_v<T_float, simd_f32> || std::is_same_v<T_float, simd_f64>;
  static_assert(is_scalar || is_vector, "Only scalar or vector of floating point is allowed");

  if constexpr (is_scalar) {
    return static_cast<int>(floor((x - xmin) * rdx));
  } else if constexpr (is_vector) {
    return xsimd::to_int(xsimd::floor((x - xmin) * rdx));
  }
}

/// return sign of argument
template <typename T_float>
static auto sign(T_float x)
{
  using namespace simd;
  constexpr bool is_scalar = std::is_floating_point_v<T_float>;
  constexpr bool is_vector = std::is_same_v<T_float, simd_f32> || std::is_same_v<T_float, simd_f64>;
  static_assert(is_scalar || is_vector, "Only scalar or vector of floating point is allowed");

  if constexpr (is_scalar) {
    return copysign(1.0, x);
  } else if constexpr (is_vector) {
    return xsimd::copysign(T_float(1.0), x);
  }
}

/// return minimum of two arguments
template <typename T_float>
static auto min(T_float x, T_float y)
{
  using namespace simd;
  constexpr bool is_scalar = std::is_floating_point_v<T_float>;
  constexpr bool is_vector = std::is_same_v<T_float, simd_f32> || std::is_same_v<T_float, simd_f64>;
  static_assert(is_scalar || is_vector, "Only scalar or vector of floating point is allowed");

  if constexpr (is_scalar) {
    return std::min(x, y);
  } else if constexpr (is_vector) {
    return xsimd::min(x, y);
  }
}

/// return maximum of two arguments
template <typename T_float>
static auto max(T_float x, T_float y)
{
  using namespace simd;
  constexpr bool is_scalar = std::is_floating_point_v<T_float>;
  constexpr bool is_vector = std::is_same_v<T_float, simd_f32> || std::is_same_v<T_float, simd_f64>;
  static_assert(is_scalar || is_vector, "Only scalar or vector of floating point is allowed");

  if constexpr (is_scalar) {
    return std::max(x, y);
  } else if constexpr (is_vector) {
    return xsimd::max(x, y);
  }
}

/// return absolute value of argument
template <typename T_float>
static auto abs(T_float x)
{
  using namespace simd;
  constexpr bool is_scalar = std::is_floating_point_v<T_float>;
  constexpr bool is_vector = std::is_same_v<T_float, simd_f32> || std::is_same_v<T_float, simd_f64>;
  static_assert(is_scalar || is_vector, "Only scalar or vector of floating point is allowed");

  if constexpr (is_scalar) {
    return std::abs(x);
  } else if constexpr (is_vector) {
    return xsimd::abs(x);
  }
}

/// ternary operator
template <typename T_float, typename T_bool>
static auto ifthenelse(T_bool cond, T_float x, T_float y)
{
  using namespace simd;
  constexpr bool is_scalar = std::is_floating_point_v<T_float>;
  constexpr bool is_vector = std::is_same_v<T_float, simd_f32> || std::is_same_v<T_float, simd_f64>;
  static_assert(is_scalar || is_vector, "Only scalar or vector of floating point is allowed");

  if constexpr (is_scalar) {
    return cond ? x : y;
  } else if constexpr (is_vector) {
    return xsimd::select(cond, x, y);
  }
}

///
/// @brief return Lorentz factor
///
/// @param[in] ux X-component of four velocity
/// @param[in] uy Y-component of four velocity
/// @param[in] uz Z-component of four velocity
/// @param[in] rc 1/c
///
template <typename T_float>
static auto lorentz_factor(T_float ux, T_float uy, T_float uz, T_float rc)
{
  using namespace simd;
  constexpr bool is_scalar = std::is_floating_point_v<T_float>;
  constexpr bool is_vector = std::is_same_v<T_float, simd_f32> || std::is_same_v<T_float, simd_f64>;
  static_assert(is_scalar || is_vector, "Only scalar or vector of floating point is allowed");

  if constexpr (is_scalar) {
    return sqrt(1 + (ux * ux + uy * uy + uz * uz) * rc * rc);
  } else if constexpr (is_vector) {
    return xsimd::sqrt(1 + (ux * ux + uy * uy + uz * uz) * rc * rc);
  }
}

///
/// @brief Boris pusher for equation of motion
///
/// @param[in,out] ux, uy, uz  Velocity in x, y, z directions
/// @param[in]     ex, ey, ez  Electric field in x, y, z directions multiplied by time step
/// @param[in]     bx, by, bz  Magnetic field in x, y, z directions multiplied by time step
///
template <typename T_float>
static void push_boris(T_float& ux, T_float& uy, T_float& uz, T_float ex, T_float ey, T_float ez,
                       T_float bx, T_float by, T_float bz)
{
  T_float tt, vx, vy, vz;

  ux += ex;
  uy += ey;
  uz += ez;

  tt = 2.0 / (1.0 + bx * bx + by * by + bz * bz);

  vx = ux + (uy * bz - uz * by);
  vy = uy + (uz * bx - ux * bz);
  vz = uz + (ux * by - uy * bx);

  ux += (vy * bz - vz * by) * tt + ex;
  uy += (vz * bx - vx * bz) * tt + ey;
  uz += (vx * by - vy * bx) * tt + ez;
}

/// implementation of first-order particle shape function
template <typename T_float>
static void shape1(T_float x, T_float X, T_float rdx, T_float s[2])
{
  T_float delta = (x - X) * rdx;

  s[0] = 1 - delta;
  s[1] = delta;
}

/// implementation of second-order particle shape function
template <typename T_float>
static void shape2(T_float x, T_float X, T_float rdx, T_float s[3])
{
  T_float delta = (x - X) * rdx;

  T_float w0 = delta;
  T_float w1 = 0.5 - w0;
  T_float w2 = 0.5 + w0;

  s[0] = 0.50 * w1 * w1;
  s[1] = 0.75 - w0 * w0;
  s[2] = 0.50 * w2 * w2;
}

/// implementation of third-order particle shape function
template <typename T_float>
static void shape3(T_float x, T_float X, T_float rdx, T_float s[4])
{
  const T_float a     = 1 / 6.0;
  T_float       delta = (x - X) * rdx;

  T_float w1      = delta;
  T_float w2      = 1 - delta;
  T_float w1_pow2 = w1 * w1;
  T_float w2_pow2 = w2 * w2;
  T_float w1_pow3 = w1_pow2 * w1;
  T_float w2_pow3 = w2_pow2 * w2;

  s[0] = a * w2_pow3;
  s[1] = a * (4 - 6 * w1_pow2 + 3 * w1_pow3);
  s[2] = a * (4 - 6 * w2_pow2 + 3 * w2_pow3);
  s[3] = a * w1_pow3;
}

/// implementation of fourth-order particle shape function
template <typename T_float>
static void shape4(T_float x, T_float X, T_float rdx, T_float s[5])
{
  const T_float a     = 1 / 384.0;
  const T_float b     = 1 / 96.0;
  const T_float c     = 115 / 192.0;
  const T_float d     = 1 / 8.0;
  T_float       delta = (x - X) * rdx;

  T_float w1      = 1 + delta;
  T_float w2      = 1 - delta;
  T_float w3      = 1 + delta * 2;
  T_float w4      = 1 - delta * 2;
  T_float w0_pow2 = delta * delta;
  T_float w1_pow2 = w1 * w1;
  T_float w2_pow2 = w2 * w2;
  T_float w1_pow3 = w1_pow2 * w1;
  T_float w2_pow3 = w2_pow2 * w2;
  T_float w1_pow4 = w1_pow3 * w1;
  T_float w2_pow4 = w2_pow3 * w2;
  T_float w3_pow4 = w3 * w3 * w3 * w3;
  T_float w4_pow4 = w4 * w4 * w4 * w4;

  s[0] = a * w4_pow4;
  s[1] = b * (55 + 20 * w1 - 120 * w1_pow2 + 80 * w1_pow3 - 16 * w1_pow4);
  s[2] = c + d * w0_pow2 * (2 * w0_pow2 - 5);
  s[3] = b * (55 + 20 * w2 - 120 * w2_pow2 + 80 * w2_pow3 - 16 * w2_pow4);
  s[4] = a * w3_pow4;
}

/// implementation of first-order particle shape function for WT scheme
template <typename T_float>
static void shape_wt1(T_float x, T_float X, T_float rdx, T_float dt, T_float rdt, T_float s[2])
{
  const T_float zero  = 0.0;
  const T_float one   = 1.0;
  T_float       delta = (x - X) * rdx;

  T_float ss = min(one, max(zero, 0.25 * rdt * (1 + 2 * dt - 2 * delta)));

  s[0] = ss;
  s[1] = 1 - ss;
}

/// implementation of second-order particle shape function for WT scheme
template <typename T_float>
static void shape_wt2(T_float x, T_float X, T_float rdx, T_float dt, T_float rdt, T_float s[3])
{
  const T_float zero  = 0.0;
  const T_float one   = 1.0;
  T_float       delta = (x - X) * rdx;

  T_float t1 = ifthenelse(delta < -dt, one, zero);
  T_float t2 = 1 - t1;
  T_float t3 = ifthenelse(delta < +dt, one, zero);
  T_float t4 = 1 - t3;
  T_float w0 = abs(delta);
  T_float w1 = dt - delta;
  T_float w2 = dt + delta;

  T_float s0_1 = w0;
  T_float s1_1 = 1 - w0;
  T_float s2_1 = 0;
  T_float s0_2 = 0.25 * rdt * w1 * w1;
  T_float s1_2 = 0.50 * rdt * (dt * (2 - dt) - w0 * w0);
  T_float s2_2 = 0.25 * rdt * w2 * w2;
  T_float s0_3 = s2_1;
  T_float s1_3 = s1_1;
  T_float s2_3 = s0_1;

  s[0] = s0_1 * t1 + s0_2 * t2 * t3 + s0_3 * t4;
  s[1] = s1_1 * t1 + s1_2 * t2 * t3 + s1_3 * t4;
  s[2] = s2_1 * t1 + s2_2 * t2 * t3 + s2_3 * t4;
}

/// implementation of third-order particle shape function for WT scheme
template <typename T_float>
static void shape_wt3(T_float x, T_float X, T_float rdx, T_float dt, T_float rdt, T_float s[4])
{
  const T_float zero  = 0.0;
  const T_float one   = 1.0;
  const T_float a     = 1 / 96.0;
  const T_float b     = 1 / 24.0;
  const T_float c     = 1 / 12.0;
  const T_float adt   = a * rdt;
  T_float       delta = (x - X) * rdx;

  T_float t1        = ifthenelse(delta < 0.5 - dt, one, zero);
  T_float t2        = 1 - t1;
  T_float t3        = ifthenelse(delta < 0.5 + dt, one, zero);
  T_float t4        = 1 - t3;
  T_float w0        = delta;
  T_float w1        = 1 - delta;
  T_float w2        = 1 + delta;
  T_float w3        = 1 - 2 * delta;
  T_float w4        = 1 + 2 * delta;
  T_float w5        = 2 * dt + w3;
  T_float w6        = 2 * dt - w3;
  T_float w7        = 3 - 2 * delta;
  T_float w0_pow2   = w0 * w0;
  T_float w1_pow2   = w1 * w1;
  T_float w3_pow2   = w3 * w3;
  T_float w3_pow3   = w3_pow2 * w3;
  T_float w4_pow2   = w4 * w4;
  T_float w5_pow3   = w5 * w5 * w5;
  T_float w6_pow3   = w6 * w6 * w6;
  T_float w7_pow2   = w7 * w7;
  T_float dt_pow2   = dt * dt;
  T_float dt_pow3   = dt_pow2 * dt;
  T_float dt_pow2_4 = 4 * dt_pow2;
  T_float s_2_odd   = adt * (-8 * dt_pow3 - 6 * dt * w3_pow2);
  T_float s_2_even  = adt * (-36 * dt_pow2 * w3 - 3 * w3_pow3);

  T_float s0_1 = b * (dt_pow2_4 + 3 * w3_pow2);
  T_float s1_1 = c * (9 - dt_pow2_4 - 12 * w0_pow2);
  T_float s2_1 = b * (dt_pow2_4 + 3 * w4_pow2);
  T_float s3_1 = 0;
  T_float s0_2 = adt * w5_pow3;
  T_float s1_2 = s_2_odd + s_2_even + w1;
  T_float s2_2 = s_2_odd - s_2_even + w0;
  T_float s3_2 = adt * w6_pow3;
  T_float s0_3 = 0;
  T_float s1_3 = b * (dt_pow2_4 + 3 * w7_pow2);
  T_float s2_3 = c * (9 - dt_pow2_4 - 12 * w1_pow2);
  T_float s3_3 = b * (dt_pow2_4 + 3 * w3_pow2);

  s[0] = s0_1 * t1 + s0_2 * t2 * t3 + s0_3 * t4;
  s[1] = s1_1 * t1 + s1_2 * t2 * t3 + s1_3 * t4;
  s[2] = s2_1 * t1 + s2_2 * t2 * t3 + s2_3 * t4;
  s[3] = s3_1 * t1 + s3_2 * t2 * t3 + s3_3 * t4;
}

/// implementation of fourth-order particle shape function for WT scheme
template <typename T_float>
static void shape_wt4(T_float x, T_float X, T_float rdx, T_float dt, T_float rdt, T_float s[5])
{
  const T_float zero  = 0.0;
  const T_float one   = 1.0;
  const T_float a     = 1 / 48.0;
  const T_float b     = 1 / 24.0;
  const T_float c     = 1 / 12.0;
  const T_float d     = 1 / 6.0;
  const T_float adt   = a * rdt;
  const T_float bdt   = b * rdt;
  const T_float cdt   = c * rdt;
  T_float       delta = (x - X) * rdx;

  T_float t1      = ifthenelse(delta < -dt, one, zero);
  T_float t2      = 1 - t1;
  T_float t3      = ifthenelse(delta < +dt, one, zero);
  T_float t4      = 1 - t3;
  T_float w0      = abs(delta);
  T_float w1      = 1 - w0;
  T_float w2      = 1 - delta;
  T_float w3      = 1 + delta;
  T_float w4      = dt - delta;
  T_float w5      = dt + delta;
  T_float w0_pow2 = w0 * w0;
  T_float w0_pow3 = w0_pow2 * w0;
  T_float w0_pow4 = w0_pow3 * w0;
  T_float w1_pow2 = w1 * w1;
  T_float w1_pow3 = w1_pow2 * w1;
  T_float w2_pow3 = w2 * w2 * w2;
  T_float w3_pow3 = w3 * w3 * w3;
  T_float w4_pow4 = w4 * w4 * w4 * w4;
  T_float w5_pow4 = w5 * w5 * w5 * w5;
  T_float dt_pow2 = dt * dt;
  T_float dt_pow3 = dt_pow2 * dt;
  T_float dt_pow4 = dt_pow3 * dt;
  T_float ss1     = -dt_pow4 - 6 * w0_pow2 * dt_pow2 - w0_pow4;
  T_float ss2 =
      3 * dt_pow4 - 8 * dt_pow3 + 18 * w0_pow2 * dt_pow2 + (16 - 24 * w0_pow2) * dt + 3 * w0_pow4;

  T_float s0_1 = d * w0 * (w0_pow2 + dt_pow2);
  T_float s1_1 = d * (4 - 6 * w1_pow2 + 3 * w1_pow3 + (1 - 3 * w0) * dt_pow2);
  T_float s2_1 = d * (4 - 6 * w0_pow2 + 3 * w0_pow3 - (2 - 3 * w0) * dt_pow2);
  T_float s3_1 = d * w1 * (w1_pow2 + dt_pow2);
  T_float s4_1 = 0;
  T_float s0_2 = adt * w4_pow4;
  T_float s1_2 = cdt * (ss1 + 2 * dt_pow3 * w3 + 2 * dt * (-6 * delta + w3_pow3));
  T_float s2_2 = bdt * ss2;
  T_float s3_2 = cdt * (ss1 + 2 * dt_pow3 * w2 + 2 * dt * (+6 * delta + w2_pow3));
  T_float s4_2 = adt * w5_pow4;
  T_float s0_3 = s4_1;
  T_float s1_3 = s3_1;
  T_float s2_3 = s2_1;
  T_float s3_3 = s1_1;
  T_float s4_3 = s0_1;

  s[0] = s0_1 * t1 + s0_2 * t2 * t3 + s0_3 * t4;
  s[1] = s1_1 * t1 + s1_2 * t2 * t3 + s1_3 * t4;
  s[2] = s2_1 * t1 + s2_2 * t2 * t3 + s2_3 * t4;
  s[3] = s3_1 * t1 + s3_2 * t2 * t3 + s3_3 * t4;
  s[4] = s4_1 * t1 + s4_2 * t2 * t3 + s4_3 * t4;
}

///
/// @brief Generic particle shape function
///
/// This function calculate particle assignment weights at grid points using a given order of shape
/// function. For an odd order shape function, the particle position is assume to be
///     X <= x < X + dx.
/// On the other hand, for an even order shape function, the particle position is assume to be
///     X - dx/2 <= x < X + dx/2.
///
/// The weights at the following positions
///     - first-order  : (X, X + dx)
///     - second-order : (X - dx, X, X + dx)
///     - third-order  : (X - dx, X, X + dx, X + 2dx)
///     - fourth-order : (X - 2dx, X - dx, X, X + dx, X + 2dx)
/// will be assigned to s.
///
/// @param[in]  x   particle position
/// @param[in]  X   grid position
/// @param[in]  rdx 1/dx
/// @param[out] s   weights at grid points
///
template <int Order, typename T_float>
static void shape(T_float x, T_float X, T_float rdx, T_float s[Order + 1])
{
  static_assert(Order >= 1 && Order <= 4, "Order must be 1, 2, 3, or 4");

  if constexpr (Order == 1) {
    shape1(x, X, rdx, s);
  } else if constexpr (Order == 2) {
    shape2(x, X, rdx, s);
  } else if constexpr (Order == 3) {
    shape3(x, X, rdx, s);
  } else if constexpr (Order == 4) {
    shape4(x, X, rdx, s);
  }
}

///
/// @brief Generic particle shape function for WT scheme
///
/// This function provides a particle shape function used for the WT scheme, which defines the
/// assignment weights dependent on the time step.
/// Otherwise, the function is the same as the `shape`.
///
/// Reference:
/// - Y. Lu, et al., Journal of Computational Physics 413, 109388 (2020).
///
/// @param[in]  x   particle position
/// @param[in]  X   grid position
/// @param[in]  rdx 1/dx
/// @param[in]  dt  c*dt/dx
/// @param[in]  rdt 1/dt
/// @param[out] s   weights at grid points
///
template <int Order, typename T_float>
static void shape_wt(T_float x, T_float X, T_float rdx, T_float dt, T_float rdt,
                     T_float s[Order + 1])
{
  static_assert(Order >= 1 && Order <= 4, "Order must be 1, 2, 3, or 4");

  if constexpr (Order == 1) {
    shape_wt1(x, X, rdx, dt, rdt, s);
  } else if constexpr (Order == 2) {
    shape_wt2(x, X, rdx, dt, rdt, s);
  } else if constexpr (Order == 3) {
    shape_wt3(x, X, rdx, dt, rdt, s);
  } else if constexpr (Order == 4) {
    shape_wt4(x, X, rdx, dt, rdt, s);
  }
}

/// charge density calculation for Esirkepov scheme
template <int N, typename T_float>
static void esirkepov3d_rho(T_float ss[2][3][N], T_float current[N][N][N][4])
{
  for (int jz = 0; jz < N; jz++) {
    for (int jy = 0; jy < N; jy++) {
      for (int jx = 0; jx < N; jx++) {
        current[jz][jy][jx][0] += ss[1][0][jx] * ss[1][1][jy] * ss[1][2][jz];
      }
    }
  }
}

/// calculation of DS(*,*) of Esirkepov (2001)
template <int N, typename T_float>
static void esirkepov3d_ds(T_float ss[2][3][N])
{
  for (int dir = 0; dir < 3; dir++) {
    for (int l = 0; l < N; l++) {
      ss[1][dir][l] -= ss[0][dir][l];
    }
  }
}

/// calculation of Jx for Esirkepov scheme
template <int N, typename T_float>
static void esirkepov3d_jx(T_float dxdt, T_float ss[2][3][N], T_float current[N][N][N][4])
{
  const T_float A = 1.0 / 2;
  const T_float B = 1.0 / 3;

  for (int jz = 0; jz < N; jz++) {
    for (int jy = 0; jy < N; jy++) {
      T_float ww = 0;
      T_float wx = -((1 * ss[0][1][jy] + A * ss[1][1][jy]) * ss[0][2][jz] +
                     (A * ss[0][1][jy] + B * ss[1][1][jy]) * ss[1][2][jz]) *
                   dxdt;

      for (int jx = 0; jx < N - 1; jx++) {
        ww += ss[1][0][jx] * wx;
        current[jz][jy][jx + 1][1] += ww;
      }
    }
  }
}

/// calculation of Jy for Esirkepov scheme
template <int N, typename T_float>
static void esirkepov3d_jy(T_float dydt, T_float ss[2][3][N], T_float current[N][N][N][4])
{
  const T_float A = 1.0 / 2;
  const T_float B = 1.0 / 3;

  for (int jz = 0; jz < N; jz++) {
    for (int jx = 0; jx < N; jx++) {
      T_float ww = 0;
      T_float wy = -((1 * ss[0][2][jz] + A * ss[1][2][jz]) * ss[0][0][jx] +
                     (A * ss[0][2][jz] + B * ss[1][2][jz]) * ss[1][0][jx]) *
                   dydt;

      for (int jy = 0; jy < N - 1; jy++) {
        ww += ss[1][1][jy] * wy;
        current[jz][jy + 1][jx][2] += ww;
      }
    }
  }
}

/// calculation of Jz for Esirkepov scheme
template <int N, typename T_float>
static void esirkepov3d_jz(T_float dzdt, T_float ss[2][3][N], T_float current[N][N][N][4])
{
  const T_float A = 1.0 / 2;
  const T_float B = 1.0 / 3;

  for (int jy = 0; jy < N; jy++) {
    for (int jx = 0; jx < N; jx++) {
      T_float ww = 0;
      T_float wz = -((1 * ss[0][0][jx] + A * ss[1][0][jx]) * ss[0][1][jy] +
                     (A * ss[0][0][jx] + B * ss[1][0][jx]) * ss[1][1][jy]) *
                   dzdt;

      for (int jz = 0; jz < N - 1; jz++) {
        ww += ss[1][2][jz] * wz;
        current[jz + 1][jy][jx][3] += ww;
      }
    }
  }
}

/// shift weights after movement for Esirkepov scheme (needed for vectorization)
template <int Order, typename T_int, typename T_float>
static void esirkepov3d_shift_weights_after_movement(T_int shift[3], T_float ss[3][Order + 3])
{
  using value_type = typename T_float::value_type;

  for (int dir = 0; dir < 3; dir++) {
    // forward
    for (int ii = 0; ii < Order + 2; ii++) {
      auto cond   = xsimd::batch_bool_cast<value_type>(shift[dir] < 0);
      ss[dir][ii] = xsimd::select(cond, ss[dir][ii + 1], ss[dir][ii]);
    }
    // backward
    for (int ii = Order + 2; ii > 0; ii--) {
      auto cond   = xsimd::batch_bool_cast<value_type>(shift[dir] > 0);
      ss[dir][ii] = xsimd::select(cond, ss[dir][ii - 1], ss[dir][ii]);
    }
  }
}

template <typename T_int, typename T_float>
static void append_current3d_impl_1(float64* ptr, T_int offset[1], T_float cur[4], float64 q)
{
  static_assert(T_float::size == 1);

  xsimd::store_aligned(ptr + offset[0] + 0, q * cur[0]);
  xsimd::store_aligned(ptr + offset[0] + 1, q * cur[1]);
  xsimd::store_aligned(ptr + offset[0] + 2, q * cur[2]);
  xsimd::store_aligned(ptr + offset[0] + 3, q * cur[3]);
}

template <typename T_int, typename T_float>
static void append_current3d_impl_2(float64* ptr, T_int offset[2], T_float cur[4], float64 q)
{
  static_assert(T_float::size == 2);

  T_float data[4];

  // transpose
  data[0] = q * xsimd::zip_lo(cur[0], cur[1]);
  data[1] = q * xsimd::zip_lo(cur[2], cur[3]);
  data[2] = q * xsimd::zip_hi(cur[0], cur[1]);
  data[3] = q * xsimd::zip_hi(cur[2], cur[3]);

  // particle 0
  data[0] += xsimd::load_aligned(ptr + offset[0] + 0);
  xsimd::store_aligned(ptr + offset[0] + 0, data[0]);
  data[1] += xsimd::load_aligned(ptr + offset[0] + 2);
  xsimd::store_aligned(ptr + offset[0] + 2, data[1]);
  // particle 1
  data[2] += xsimd::load_aligned(ptr + offset[1] + 0);
  xsimd::store_aligned(ptr + offset[1] + 0, data[2]);
  data[3] += xsimd::load_aligned(ptr + offset[1] + 2);
  xsimd::store_aligned(ptr + offset[1] + 2, data[3]);
}

template <typename T_int, typename T_float>
static void append_current3d_impl_4(float64* ptr, T_int offset[4], T_float cur[4], float64 q)
{
  static_assert(T_float::size == 4);

  T_float data[4];

  // transpose
  auto p0 = xsimd::zip_lo(cur[0], cur[2]);
  auto p1 = xsimd::zip_hi(cur[0], cur[2]);
  auto p2 = xsimd::zip_lo(cur[1], cur[3]);
  auto p3 = xsimd::zip_hi(cur[1], cur[3]);
  data[0] = q * xsimd::zip_lo(p0, p2);
  data[1] = q * xsimd::zip_hi(p0, p2);
  data[2] = q * xsimd::zip_lo(p1, p3);
  data[3] = q * xsimd::zip_hi(p1, p3);

  // particle 0
  data[0] += xsimd::load_aligned(ptr + offset[0]);
  xsimd::store_aligned(ptr + offset[0], data[0]);
  // particle 1
  data[1] += xsimd::load_aligned(ptr + offset[1]);
  xsimd::store_aligned(ptr + offset[1], data[1]);
  // particle 2
  data[2] += xsimd::load_aligned(ptr + offset[2]);
  xsimd::store_aligned(ptr + offset[2], data[2]);
  // particle 3
  data[3] += xsimd::load_aligned(ptr + offset[3]);
  xsimd::store_aligned(ptr + offset[3], data[3]);
}

template <typename T_int, typename T_float>
static void append_current3d_impl_8(float64* ptr, T_int offset[8], T_float cur[4], float64 q)
{
  static_assert(T_float::size == 8);

  const T_float zero = 0;
  const auto    mask = (xsimd::detail::make_sequence_as_batch<T_float>() < 4);

  T_float data[8];

  // transpose
  auto p0 = xsimd::zip_lo(cur[0], cur[2]);
  auto p1 = xsimd::zip_hi(cur[0], cur[2]);
  auto p2 = xsimd::zip_lo(cur[1], cur[3]);
  auto p3 = xsimd::zip_hi(cur[1], cur[3]);
  auto q0 = q * xsimd::zip_lo(p0, p2);
  auto q1 = q * xsimd::zip_hi(p0, p2);
  auto q2 = q * xsimd::zip_lo(p1, p3);
  auto q3 = q * xsimd::zip_hi(p1, p3);
  data[0] = xsimd::select(mask, q0, zero);
  data[1] = xsimd::select(mask, xsimd::rotate_right<4>(q0), zero);
  data[2] = xsimd::select(mask, q1, zero);
  data[3] = xsimd::select(mask, xsimd::rotate_right<4>(q1), zero);
  data[4] = xsimd::select(mask, q2, zero);
  data[5] = xsimd::select(mask, xsimd::rotate_right<4>(q2), zero);
  data[6] = xsimd::select(mask, q3, zero);
  data[7] = xsimd::select(mask, xsimd::rotate_right<4>(q3), zero);

  // particle 0
  data[0] += xsimd::load_aligned(ptr + offset[0]);
  xsimd::store_aligned(ptr + offset[0], data[0]);
  // particle 1
  data[1] += xsimd::load_aligned(ptr + offset[1]);
  xsimd::store_aligned(ptr + offset[1], data[1]);
  // particle 2
  data[2] += xsimd::load_aligned(ptr + offset[2]);
  xsimd::store_aligned(ptr + offset[2], data[2]);
  // particle 3
  data[3] += xsimd::load_aligned(ptr + offset[3]);
  xsimd::store_aligned(ptr + offset[3], data[3]);
  // particle 4
  data[4] += xsimd::load_aligned(ptr + offset[4]);
  xsimd::store_aligned(ptr + offset[4], data[4]);
  // particle 5
  data[5] += xsimd::load_aligned(ptr + offset[5]);
  xsimd::store_aligned(ptr + offset[5], data[5]);
  // particle 6
  data[6] += xsimd::load_aligned(ptr + offset[6]);
  xsimd::store_aligned(ptr + offset[6], data[6]);
  // particle 7
  data[7] += xsimd::load_aligned(ptr + offset[7]);
  xsimd::store_aligned(ptr + offset[7], data[7]);
}

template <int Order, typename T_array, typename T_int, typename T_float>
static void append_current3d(T_array& uj, T_int iz0, T_int iy0, T_int ix0,
                             T_float current[Order + 3][Order + 3][Order + 3][4], float64 q)
{
  using namespace simd;
  constexpr int  size      = Order + 3;
  constexpr bool is_scalar = std::is_integral_v<T_int> && std::is_floating_point_v<T_float>;
  constexpr bool is_vector = std::is_same_v<T_float, simd_f32> || std::is_same_v<T_float, simd_f64>;
  constexpr bool is_sorted = std::is_integral_v<T_int>;
  static_assert(is_scalar || is_vector, "Only scalar or vector of floating point is allowed");

  if constexpr (is_scalar == true) {
    // naive scalar implementation
    for (int jz = 0, iz = iz0; jz < size; jz++, iz++) {
      for (int jy = 0, iy = iy0; jy < size; jy++, iy++) {
        for (int jx = 0, ix = ix0; jx < size; jx++, ix++) {
          uj(iz, iy, ix, 0) += q * current[jz][jy][jx][0];
          uj(iz, iy, ix, 1) += q * current[jz][jy][jx][1];
          uj(iz, iy, ix, 2) += q * current[jz][jy][jx][2];
          uj(iz, iy, ix, 3) += q * current[jz][jy][jx][3];
        }
      }
    }
  } else if constexpr (is_vector == true && is_sorted == true) {
    // all particle contributions are added to the same grid point
    for (int jz = 0, iz = iz0; jz < size; jz++, iz++) {
      for (int jy = 0, iy = iy0; jy < size; jy++, iy++) {
        for (int jx = 0, ix = ix0; jx < size; jx++, ix++) {
          uj(iz, iy, ix, 0) += q * xsimd::reduce_add(current[jz][jy][jx][0]);
          uj(iz, iy, ix, 1) += q * xsimd::reduce_add(current[jz][jy][jx][1]);
          uj(iz, iy, ix, 2) += q * xsimd::reduce_add(current[jz][jy][jx][2]);
          uj(iz, iy, ix, 3) += q * xsimd::reduce_add(current[jz][jy][jx][3]);
        }
      }
    }
  } else if constexpr (is_vector == true && is_sorted == false) {
    // particle contributions are added to different grid points
    auto stride_z = get_stride(uj, 0);
    auto stride_y = get_stride(uj, 1);
    auto stride_x = get_stride(uj, 2);
    auto stride_c = get_stride(uj, 3);
    auto pointer  = get_data_pointer(uj);

    alignas(64) int64 offset[T_float::size];

    for (int jz = 0; jz < size; jz++) {
      T_int iz = iz0 + jz;
      for (int jy = 0; jy < size; jy++) {
        T_int iy = iy0 + jy;
        for (int jx = 0; jx < size; jx++) {
          T_int ix = ix0 + jx;
          // call specialized function for a specific SIMD vector size
          xsimd::store_aligned(offset, iz * stride_z + iy * stride_y + ix * stride_x);
          if constexpr (T_float::size == 1) {
            append_current3d_impl_1(pointer, offset, current[jz][jy][jx], q);
          } else if constexpr (T_float::size == 2) {
            append_current3d_impl_2(pointer, offset, current[jz][jy][jx], q);
          } else if constexpr (T_float::size == 4) {
            append_current3d_impl_4(pointer, offset, current[jz][jy][jx], q);
          } else if constexpr (T_float::size == 8) {
            append_current3d_impl_8(pointer, offset, current[jz][jy][jx], q);
          } else {
            static_assert([] { return false; }(), "Invalid vector size");
          }
        }
      }
    }
  }
}

///
/// @brief calculate current via density decomposition scheme (Esirkepov 2001)
///
/// This implements the density decomposition scheme of Esirkepov (2001) in 3D with the order of
/// shape function given by "Order". The input to this routine is the first argument "ss", with
/// ss[0][*][*] and ss[1][*][*] are 1D weights before and after the movement of particle by one time
/// step, respectively. Note that the weight should be multiplied by the particle charge. The
/// current density is appended to the second argument "current", which is an array of local
/// (Order+3)x(Order+3)x(Order+3) mesh with 4 components including charge density.
///
/// @param[in]     dxdt    dx/dt
/// @param[in]     dydt    dy/dt
/// @param[in]     dzdt    dz/dt
/// @param[in]     ss      array of 1D weights
/// @param[in,out] current array of local current
///
template <int Order, typename T_float>
static void esirkepov3d(T_float dxdt, T_float dydt, T_float dzdt, T_float ss[2][3][Order + 3],
                        T_float current[Order + 3][Order + 3][Order + 3][4])
{
  // calculate rho
  esirkepov3d_rho<Order + 3>(ss, current);

  // ss[1][*][*] now represents DS(*,*) of Esirkepov (2001)
  esirkepov3d_ds<Order + 3>(ss);

  // calculate Jx, Jy, Jz
  esirkepov3d_jx<Order + 3>(dxdt, ss, current);
  esirkepov3d_jy<Order + 3>(dydt, ss, current);
  esirkepov3d_jz<Order + 3>(dzdt, ss, current);
}

template <int Order, typename T_array>
static auto interpolate3d_impl_scalar(T_array& eb, int iz0, int iy0, int ix0, int ik,
                                      float64 wz[Order + 2], float64 wy[Order + 2],
                                      float64 wx[Order + 2], float64 dt)
{
  // Notice for the loop length (Order + 1) instead of (Order + 2)
  float64 result_z = 0;
  for (int jz = 0, iz = iz0; jz < Order + 1; jz++, iz++) {
    float64 result_y = 0;
    for (int jy = 0, iy = iy0; jy < Order + 1; jy++, iy++) {
      float64 result_x = 0;
      for (int jx = 0, ix = ix0; jx < Order + 1; jx++, ix++) {
        result_x += eb(iz, iy, ix, ik) * wx[jx];
      }
      result_y += result_x * wy[jy];
    }
    result_z += result_y * wz[jz];
  }

  return result_z * dt;
}

template <int Order, typename T_array, typename T_int, typename T_float>
static auto interpolate3d_impl_vector_unsorted(T_array& eb, T_int iz0, T_int iy0, T_int ix0, int ik,
                                               T_float wz[Order + 2], T_float wy[Order + 2],
                                               T_float wx[Order + 2], float64 dt)
{
  using namespace simd;

  auto stride_z = get_stride(eb, 0);
  auto stride_y = get_stride(eb, 1);
  auto stride_x = get_stride(eb, 2);
  auto stride_c = get_stride(eb, 3);
  auto pointer  = get_data_pointer(eb);

  // Notice for the loop length (Order + 1) instead of (Order + 2)
  T_float result_z = 0;
  for (int jz = 0; jz < Order + 1; jz++) {
    T_int   index_z  = (iz0 + jz) * stride_z;
    T_float result_y = 0;
    for (int jy = 0; jy < Order + 1; jy++) {
      T_int   index_y  = (iy0 + jy) * stride_y + index_z;
      T_float result_x = 0;
      for (int jx = 0; jx < Order + 1; jx++) {
        T_int index_x = (ix0 + jx) * stride_x + index_y;
        T_int index   = index_x + ik * stride_c;
        result_x += simd_f64::gather(pointer, index) * wx[jx];
      }
      result_y += result_x * wy[jy];
    }
    result_z += result_y * wz[jz];
  }

  return result_z * dt;
}

template <int Order, typename T_array, typename T_float>
static auto interpolate3d_impl_vector_sorted(T_array& eb, int iz0, int iy0, int ix0, int ik,
                                             T_float wz[Order + 2], T_float wy[Order + 2],
                                             T_float wx[Order + 2], float64 dt)
{
  T_float result_z = 0;
  for (int jz = 0, iz = iz0; jz < Order + 2; jz++, iz++) {
    T_float result_y = 0;
    for (int jy = 0, iy = iy0; jy < Order + 2; jy++, iy++) {
      T_float result_x = 0;
      for (int jx = 0, ix = ix0; jx < Order + 2; jx++, ix++) {
        result_x += eb(iz, iy, ix, ik) * wx[jx];
      }
      result_y += result_x * wy[jy];
    }
    result_z += result_y * wz[jz];
  }

  return result_z * dt;
}

///
/// @brief calculate electromagnetic field at particle position by interpolation
///
/// @param[in] eb  electromagnetic field (4D array)
/// @param[in] iz0 first z-index of eb
/// @param[in] iy0 first y-index of eb
/// @param[in] ix0 first x-index of eb
/// @param[in] ik  index for electromagnetic field component
/// @param[in] wz  weight in z direction
/// @param[in] wy  weight in y direction
/// @param[in] wx  weight in x direction
/// @param[in] dt  time step (multiplied to the returned electromagnetic field)
///
template <int Order, typename T_array, typename T_int, typename T_float>
static auto interpolate3d(T_array& eb, T_int iz0, T_int iy0, T_int ix0, int ik,
                          T_float wz[Order + 2], T_float wy[Order + 2], T_float wx[Order + 2],
                          float64 dt)
{
  using namespace simd;
  constexpr bool is_scalar = std::is_floating_point_v<T_float>;
  constexpr bool is_vector = std::is_same_v<T_float, simd_f32> || std::is_same_v<T_float, simd_f64>;
  constexpr bool is_sorted = std::is_integral_v<T_int>;

  if constexpr (is_scalar == true) {
    return interpolate3d_impl_scalar<Order>(eb, iz0, iy0, ix0, ik, wz, wy, wx, dt);
  } else if constexpr (is_vector == true && is_sorted == false) {
    return interpolate3d_impl_vector_unsorted<Order>(eb, iz0, iy0, ix0, ik, wz, wy, wx, dt);
  } else if constexpr (is_vector == true && is_sorted == true) {
    return interpolate3d_impl_vector_sorted<Order>(eb, iz0, iy0, ix0, ik, wz, wy, wx, dt);
  } else {
    static_assert([] { return false; }(), "Invalid template parameters");
  }
}

} // namespace primitives

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
