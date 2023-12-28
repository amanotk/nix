// -*- C++ -*-
#ifndef _PARTICLE_PRIMITIVES_HPP_
#define _PARTICLE_PRIMITIVES_HPP_

#include "nix.hpp"

NIX_NAMESPACE_BEGIN

namespace primitives
{

///
/// @brief digitize for calculating grid index for particles
///
static int digitize(float64 x, float64 xmin, float64 rdx)
{
  return static_cast<int>(floor((x - xmin) * rdx));
}

///
/// @brief return sign of argument
///
static float64 sign(float64 x)
{
  return copysign(1.0, x);
}

///
/// @brief return Lorentz factor
///
/// @param[in] ux X-component of four velocity
/// @param[in] uy Y-component of four velocity
/// @param[in] uz Z-component of four velocity
/// @param[in] rc 1/c^2
///
static float64 lorentz_factor(float64 ux, float64 uy, float64 uz, float64 rc)
{
  return sqrt(1 + (ux * ux + uy * uy + uz * uz) * rc * rc);
}

///
/// @brief Buneman-Boris pusher for Lorentz equation
///
/// @param[in,out] u  Velocity
/// @param[in]     eb Electromagnetic field multiplied by time step
///
static void push_buneman_boris(float64 u[3], float64 eb[6])
{
  float64 tt, v[3];

  u[0] += eb[0];
  u[1] += eb[1];
  u[2] += eb[2];

  tt = 2.0 / (1.0 + eb[3] * eb[3] + eb[4] * eb[4] + eb[5] * eb[5]);

  v[0] = u[0] + (u[1] * eb[5] - u[2] * eb[4]);
  v[1] = u[1] + (u[2] * eb[3] - u[0] * eb[5]);
  v[2] = u[2] + (u[0] * eb[4] - u[1] * eb[3]);

  u[0] += (v[1] * eb[5] - v[2] * eb[4]) * tt + eb[0];
  u[1] += (v[2] * eb[3] - v[0] * eb[5]) * tt + eb[1];
  u[2] += (v[0] * eb[4] - v[1] * eb[3]) * tt + eb[2];
}

///
/// @brief first-order particle shape function
///
/// This function assumes X <= x < X + dx and computes particle weights at
/// the two gird points X, X + dx, which are assigned to s[0..1].
///
/// @param[in]  x   particle position
/// @param[in]  X   grid point such that X <= x < X + dx
/// @param[in]  rdx 1/dx
/// @param[out] s   weights at grid points (X, X + dx)
/// @param[in]  w   normalization factor (1 by default)
///
static void shape1(float64 x, float64 X, float64 rdx, float64 s[2], float64 w = 1)
{
  float64 delta = (x - X) * rdx;

  s[0] = w * (1 - delta);
  s[1] = w * delta;
}

///
/// @brief second-order particle shape function
///
/// This function assumes X - dx/2 <= x < X + dx/2 and computes particle weights
/// at the three grid points X-dx, X, X + dx, which are assigned to s[0..2].
///
/// @param[in]  x   particle position
/// @param[in]  X   grid point such that X - dx/2 <= x < X + dx/2
/// @param[in]  rdx 1/dx
/// @param[out] s   weights at grid points (X - dx, X, X + dx)
/// @param[in]  w   normalization factor (1 by default)
///
static void shape2(float64 x, float64 X, float64 rdx, float64 s[3], float64 w = 1)
{
  float64 delta = (x - X) * rdx;

  s[0] = w * 0.50 * (0.50 - delta) * (0.5 - delta);
  s[1] = w * 0.75 - delta * delta;
  s[2] = w * 0.50 * (0.50 + delta) * (0.5 + delta);
}

///
/// @brief calculate electromagnetic field at particle position by first-order interpolation
///
/// This implements linear interpolation of the electromagnetic field at the particle position.
/// The indices iz, iy, ix and weights wz, wy, wx must appropriately be calculated in advance.
/// This function is just a shorthand notation of interpolation.
///
/// @param[in] eb electromagnetic field (4D array)
/// @param[in] iz z-index of particle position
/// @param[in] iy y-index of particle position
/// @param[in] ix x-index of particle position
/// @param[in] ik index for electromagnetic field component
/// @param[in] wz weight in z direction
/// @param[in] wy weight in y direction
/// @param[in] wx weight in x direction
/// @param[in] dt time step (multiplied to the returned electromagnetic field)
///
template <typename T>
static float64 interp3d1(const T& eb, int iz, int iy, int ix, int ik, const float64 wz[2],
                         const float64 wy[2], const float64 wx[2], float64 dt = 1)
{
  float64 result;

  int ix1 = ix;
  int ix2 = ix + 1;
  int iy1 = iy;
  int iy2 = iy + 1;
  int iz1 = iz;
  int iz2 = iz + 1;

  // clang-format off
  result = (
    wz[0] * (
      wy[0] * (wx[0] * eb(iz1, iy1, ix1, ik) + wx[1] * eb(iz1, iy1, ix2, ik)) +
      wy[1] * (wx[0] * eb(iz1, iy2, ix1, ik) + wx[1] * eb(iz1, iy2, ix2, ik))
      ) +
    wz[1] * (
      wy[0] * (wx[0] * eb(iz2, iy1, ix1, ik) + wx[1] * eb(iz2, iy1, ix2, ik)) +
      wy[1] * (wx[0] * eb(iz2, iy2, ix1, ik) + wx[1] * eb(iz2, iy2, ix2, ik))
      )
    ) * dt;
  // clang-format on

  return result;
}

template <typename T>
static float64 interp3d2(const T& eb, int iz, int iy, int ix, int ik, const float64 wz[3],
                         const float64 wy[3], const float64 wx[3], float64 dt = 1)
{
  float64 result;

  int ix1 = ix - 1;
  int ix2 = ix;
  int ix3 = ix + 1;
  int iy1 = iy - 1;
  int iy2 = iy;
  int iy3 = iy + 1;
  int iz1 = iz - 1;
  int iz2 = iz;
  int iz3 = iz + 1;

  // clang-format off
  result = (
    wz[0] * (
      wy[0] * (
        wx[0] * eb(iz1, iy1, ix1, ik) + wx[1] * eb(iz1, iy1, ix2, ik) + wx[2] * eb(iz1, iy1, ix3, ik)
        ) +
      wy[1] * (
        wx[0] * eb(iz1, iy2, ix1, ik) + wx[1] * eb(iz1, iy2, ix2, ik) + wx[2] * eb(iz1, iy2, ix3, ik)
        ) +
      wy[2] * (
        wx[0] * eb(iz1, iy3, ix1, ik) + wx[1] * eb(iz1, iy3, ix2, ik) + wx[2] * eb(iz1, iy3, ix3, ik)
        )
      ) +
    wz[1] * (
      wy[0] * (
        wx[0] * eb(iz2, iy1, ix1, ik) + wx[1] * eb(iz2, iy1, ix2, ik) + wx[2] * eb(iz2, iy1, ix3, ik)
        ) +
      wy[1] * (
        wx[0] * eb(iz2, iy2, ix1, ik) + wx[1] * eb(iz2, iy2, ix2, ik) + wx[2] * eb(iz2, iy2, ix3, ik)
        ) +
      wy[2] * (
        wx[0] * eb(iz2, iy3, ix1, ik) + wx[1] * eb(iz2, iy3, ix2, ik) + wx[2] * eb(iz2, iy3, ix3, ik)
        )
      ) +
    wz[2] * (
      wy[0] * (
        wx[0] * eb(iz3, iy1, ix1, ik) + wx[1] * eb(iz3, iy1, ix2, ik) + wx[2] * eb(iz3, iy1, ix3, ik)
        ) +
      wy[1] * (
        wx[0] * eb(iz3, iy2, ix1, ik) + wx[1] * eb(iz3, iy2, ix2, ik) + wx[2] * eb(iz3, iy2, ix3, ik)
        ) +
      wy[2] * (
        wx[0] * eb(iz3, iy3, ix1, ik) + wx[1] * eb(iz3, iy3, ix2, ik) + wx[2] * eb(iz3, iy3, ix3, ik)
        )
      )
    ) * dt;
  // clang-format on

  return result;
}

template <int N>
static inline void esirkepov3d_rho(float64 ss[2][3][N], float64 current[N][N][N][4])
{
  for (int jz = 0; jz < N; jz++) {
    for (int jy = 0; jy < N; jy++) {
      for (int jx = 0; jx < N; jx++) {
        current[jz][jy][jx][0] += ss[1][0][jx] * ss[1][1][jy] * ss[1][2][jz];
      }
    }
  }
}

template <int N>
static inline void esirkepov3d_ds(float64 ss[2][3][N])
{
  for (int dir = 0; dir < 3; dir++) {
    for (int l = 0; l < N; l++) {
      ss[1][dir][l] -= ss[0][dir][l];
    }
  }
}

template <int N>
static inline void esirkepov3d_jx(float64 dxdt, float64 ss[2][3][N], float64 current[N][N][N][4])
{
  constexpr float64 A = 1.0 / 2;
  constexpr float64 B = 1.0 / 3;

  for (int jz = 0; jz < N; jz++) {
    for (int jy = 0; jy < N; jy++) {
      float64 ww = 0;
      float64 wx = -((1 * ss[0][1][jy] + A * ss[1][1][jy]) * ss[0][2][jz] +
                     (A * ss[0][1][jy] + B * ss[1][1][jy]) * ss[1][2][jz]) *
                   dxdt;

      for (int jx = 0; jx < N - 1; jx++) {
        ww += ss[1][0][jx] * wx;
        current[jz][jy][jx + 1][1] += ww;
      }
    }
  }
}

template <int N>
static inline void esirkepov3d_jy(float64 dydt, float64 ss[2][3][N], float64 current[N][N][N][4])
{
  constexpr float64 A = 1.0 / 2;
  constexpr float64 B = 1.0 / 3;

  for (int jz = 0; jz < N; jz++) {
    for (int jx = 0; jx < N; jx++) {
      float64 ww = 0;
      float64 wy = -((1 * ss[0][2][jz] + A * ss[1][2][jz]) * ss[0][0][jx] +
                     (A * ss[0][2][jz] + B * ss[1][2][jz]) * ss[1][0][jx]) *
                   dydt;

      for (int jy = 0; jy < N - 1; jy++) {
        ww += ss[1][1][jy] * wy;
        current[jz][jy + 1][jx][2] += ww;
      }
    }
  }
}

template <int N>
static inline void esirkepov3d_jz(float64 dzdt, float64 ss[2][3][N], float64 current[N][N][N][4])
{
  constexpr float64 A = 1.0 / 2;
  constexpr float64 B = 1.0 / 3;

  for (int jy = 0; jy < N; jy++) {
    for (int jx = 0; jx < N; jx++) {
      float64 ww = 0;
      float64 wz = -((1 * ss[0][0][jx] + A * ss[1][0][jx]) * ss[0][1][jy] +
                     (A * ss[0][0][jx] + B * ss[1][0][jx]) * ss[1][1][jy]) *
                   dzdt;

      for (int jz = 0; jz < N - 1; jz++) {
        ww += ss[1][2][jz] * wz;
        current[jz + 1][jy][jx][3] += ww;
      }
    }
  }
}

///
/// @brief calculate current via first-order density decomposition scheme
///
/// This implements the density decomposition scheme of Esirkepov (2001) with the first-order
/// shape function in 3D. The input to this routine is the first argument "ss", with ss[0][*][*]
/// and ss[1][*][*] are 1D weights before and after the movement of particle by one time step,
/// respectively. Note that the weight should be multiplied by the particle charge. The current
/// density is appended to the second argument "current", which is an array of local 4x4x4 mesh
/// with 4 components including charge density.
///
/// @param[in]     dxdt    dx/dt
/// @param[in]     dydt    dy/dt
/// @param[in]     dzdt    dz/dt
/// @param[in]     ss      array of 1D weights
/// @param[in,out] current array of local current
///
static void esirkepov3d1(float64 dxdt, float64 dydt, float64 dzdt, float64 ss[2][3][4],
                         float64 current[4][4][4][4])
{
  // calculate rho
  esirkepov3d_rho<4>(ss, current);

  // ss[1][*][*] now represents DS(*,*) of Esirkepov (2001)
  esirkepov3d_ds<4>(ss);

  // calculate Jx, Jy, Jz
  esirkepov3d_jx<4>(dxdt, ss, current);
  esirkepov3d_jy<4>(dydt, ss, current);
  esirkepov3d_jz<4>(dzdt, ss, current);
}

///
/// @brief calculate current via second-order density decomposition scheme
///
/// The second order version update the current density of local 5x5x5 mesh.
///
static void esirkepov3d2(float64 dxdt, float64 dydt, float64 dzdt, float64 ss[2][3][5],
                         float64 current[5][5][5][4])
{
  // calculate rho
  esirkepov3d_rho<5>(ss, current);

  // ss[1][*][*] now represents DS(*,*) of Esirkepov (2001)
  esirkepov3d_ds<5>(ss);

  // calculate Jx, Jy, Jz
  esirkepov3d_jx<5>(dxdt, ss, current);
  esirkepov3d_jy<5>(dydt, ss, current);
  esirkepov3d_jz<5>(dzdt, ss, current);
}

} // namespace primitives

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
