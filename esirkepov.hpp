// -*- C++ -*-
#ifndef _ESIRKEPOV_HPP_
#define _ESIRKEPOV_HPP_

#include "nix.hpp"
#include "xsimd/xsimd.hpp"

NIX_NAMESPACE_BEGIN

namespace esirkepov
{
/// charge density calculation for Esirkepov scheme
template <int N, typename T_float>
static void rho3d(T_float qs, T_float ss[2][3][N], T_float current[N][N][N][4])
{
  for (int jz = 0; jz < N; jz++) {
    for (int jy = 0; jy < N; jy++) {
      for (int jx = 0; jx < N; jx++) {
        current[jz][jy][jx][0] += qs * ss[1][0][jx] * ss[1][1][jy] * ss[1][2][jz];
      }
    }
  }
}

/// calculation of DS(*,*) of Esirkepov (2001)
template <int N, typename T_float>
static void ds3d(T_float ss[2][3][N])
{
  for (int dir = 0; dir < 3; dir++) {
    for (int l = 0; l < N; l++) {
      ss[1][dir][l] -= ss[0][dir][l];
    }
  }
}

/// calculation of Jx for Esirkepov scheme
template <int N, typename T_float>
static void jx3d(T_float qdxdt, T_float ss[2][3][N], T_float current[N][N][N][4])
{
  const T_float A = 1.0 / 2;
  const T_float B = 1.0 / 3;

  for (int jz = 0; jz < N; jz++) {
    for (int jy = 0; jy < N; jy++) {
      T_float ww = 0;
      T_float wx = -((1 * ss[0][1][jy] + A * ss[1][1][jy]) * ss[0][2][jz] +
                     (A * ss[0][1][jy] + B * ss[1][1][jy]) * ss[1][2][jz]) *
                   qdxdt;

      for (int jx = 0; jx < N - 1; jx++) {
        ww += ss[1][0][jx] * wx;
        current[jz][jy][jx + 1][1] += ww;
      }
    }
  }
}

/// calculation of Jy for Esirkepov scheme
template <int N, typename T_float>
static void jy3d(T_float qdydt, T_float ss[2][3][N], T_float current[N][N][N][4])
{
  const T_float A = 1.0 / 2;
  const T_float B = 1.0 / 3;

  for (int jz = 0; jz < N; jz++) {
    for (int jx = 0; jx < N; jx++) {
      T_float ww = 0;
      T_float wy = -((1 * ss[0][2][jz] + A * ss[1][2][jz]) * ss[0][0][jx] +
                     (A * ss[0][2][jz] + B * ss[1][2][jz]) * ss[1][0][jx]) *
                   qdydt;

      for (int jy = 0; jy < N - 1; jy++) {
        ww += ss[1][1][jy] * wy;
        current[jz][jy + 1][jx][2] += ww;
      }
    }
  }
}

/// calculation of Jz for Esirkepov scheme
template <int N, typename T_float>
static void jz3d(T_float qdzdt, T_float ss[2][3][N], T_float current[N][N][N][4])
{
  const T_float A = 1.0 / 2;
  const T_float B = 1.0 / 3;

  for (int jy = 0; jy < N; jy++) {
    for (int jx = 0; jx < N; jx++) {
      T_float ww = 0;
      T_float wz = -((1 * ss[0][0][jx] + A * ss[1][0][jx]) * ss[0][1][jy] +
                     (A * ss[0][0][jx] + B * ss[1][0][jx]) * ss[1][1][jy]) *
                   qdzdt;

      for (int jz = 0; jz < N - 1; jz++) {
        ww += ss[1][2][jz] * wz;
        current[jz + 1][jy][jx][3] += ww;
      }
    }
  }
}

/// shift weights after movement for Esirkepov scheme (needed for vectorization)
template <int Order, typename T_int, typename T_float>
static void shift_weights3d(T_int shift[3], T_float ss[3][Order + 3])
{
  constexpr bool is_scalar = std::is_integral_v<T_int> && std::is_floating_point_v<T_float>;
  constexpr bool is_vector = std::is_same_v<T_float, simd_f32> || std::is_same_v<T_float, simd_f64>;

  if constexpr (is_scalar == true) {
    for (int dir = 0; dir < 3; dir++) {
      if (shift[dir] < 0) {
        // forward
        for (int ii = 0; ii < Order + 2; ii++) {
          ss[dir][ii] = ss[dir][ii + 1];
        }
      } else if (shift[dir] > 0) {
        // backward
        for (int ii = Order + 2; ii > 0; ii--) {
          ss[dir][ii] = ss[dir][ii - 1];
        }
      }
    }
  } else if constexpr (is_vector == true) {
    using value_type = typename T_float::value_type;

    for (int dir = 0; dir < 3; dir++) {
      // forward
      auto cond_f = xsimd::batch_bool_cast<value_type>(shift[dir] < 0);
      for (int ii = 0; ii < Order + 2; ii++) {
        ss[dir][ii] = xsimd::select(cond_f, ss[dir][ii + 1], ss[dir][ii]);
      }
      // backward
      auto cond_b = xsimd::batch_bool_cast<value_type>(shift[dir] > 0);
      for (int ii = Order + 2; ii > 0; ii--) {
        ss[dir][ii] = xsimd::select(cond_b, ss[dir][ii - 1], ss[dir][ii]);
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
static void deposit3d(float64 dxdt, float64 dydt, float64 dzdt, T_float qs,
                      T_float ss[2][3][Order + 3],
                      T_float current[Order + 3][Order + 3][Order + 3][4])
{
  // calculate rho
  rho3d<Order + 3>(qs, ss, current);

  // ss[1][*][*] now represents DS(*,*) of Esirkepov (2001)
  ds3d<Order + 3>(ss);

  // calculate Jx, Jy, Jz
  jx3d<Order + 3>(qs * dxdt, ss, current);
  jy3d<Order + 3>(qs * dydt, ss, current);
  jz3d<Order + 3>(qs * dzdt, ss, current);
}

} // namespace esirkepov

NIX_NAMESPACE_END

#endif // _ESIRKEPOV_HPP_