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
/// @brief particle shape function
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
template <int Order>
static void shape(float64 x, float64 X, float64 rdx, float64 s[Order + 1]);

template <>
void shape<1>(float64 x, float64 X, float64 rdx, float64 s[2])
{
  float64 delta = (x - X) * rdx;

  s[0] = 1 - delta;
  s[1] = delta;
}

template <>
void shape<2>(float64 x, float64 X, float64 rdx, float64 s[3])
{
  float64 delta0 = (x - X) * rdx;
  float64 delta1 = 0.5 - delta0;
  float64 delta2 = 0.5 + delta0;

  s[0] = 0.50 * delta1 * delta1;
  s[1] = 0.75 - delta0 * delta0;
  s[2] = 0.50 * delta2 * delta2;
}

template <>
void shape<3>(float64 x, float64 X, float64 rdx, float64 s[4])
{
  constexpr float64 a = 1 / 6.0;

  float64 delta1         = (x - X) * rdx;
  float64 delta2         = 1 - delta1;
  float64 delta1_squared = delta1 * delta1;
  float64 delta2_squared = delta2 * delta2;
  float64 delta1_cubed   = delta1_squared * delta1;
  float64 delta2_cubed   = delta2_squared * delta2;

  s[0] = a * delta2_cubed;
  s[1] = a * (4 - 6 * delta1_squared + 3 * delta1_cubed);
  s[2] = a * (4 - 6 * delta2_squared + 3 * delta2_cubed);
  s[3] = a * delta1_cubed;
}

template <>
void shape<4>(float64 x, float64 X, float64 rdx, float64 s[5])
{
  constexpr float64 a = 1 / 384.0;
  constexpr float64 b = 1 / 96.0;
  constexpr float64 c = 115 / 192.0;
  constexpr float64 d = 1 / 8.0;

  float64 delta0         = (x - X) * rdx;
  float64 delta1         = 1 + delta0;
  float64 delta2         = 1 - delta0;
  float64 delta3         = 1 + delta0 * 2;
  float64 delta4         = 1 - delta0 * 2;
  float64 delta0_squared = delta0 * delta0;
  float64 delta1_squared = delta1 * delta1;
  float64 delta2_squared = delta2 * delta2;
  float64 delta1_cubed   = delta1_squared * delta1;
  float64 delta2_cubed   = delta2_squared * delta2;
  float64 delta1_quad    = delta1_cubed * delta1;
  float64 delta2_quad    = delta2_cubed * delta2;

  s[0] = a * delta4 * delta4 * delta4 * delta4;
  s[1] = b * (55 + 20 * delta1 - 120 * delta1_squared + 80 * delta1_cubed - 16 * delta1_quad);
  s[2] = c + d * delta0_squared * (2 * delta0_squared - 5);
  s[3] = b * (55 + 20 * delta2 - 120 * delta2_squared + 80 * delta2_cubed - 16 * delta2_quad);
  s[4] = a * delta3 * delta3 * delta3 * delta3;
}

///
/// @brief particle shape function for WT scheme
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
template <int Order>
static void shape_wt(float64 x, float64 X, float64 rdx, float64 dt, float64 rdt,
                     float64 s[Order + 1]);

template <>
void shape_wt<1>(float64 x, float64 X, float64 rdx, float64 dt, float64 rdt, float64 s[2])
{
  float64 delta = (x - X) * rdx;
  float64 ss    = std::min(1.0, std::max(0.0, 0.25 * rdt * (1 + 2 * dt - 2 * delta)));

  s[0] = ss;
  s[1] = 1 - ss;
}

template <>
void shape_wt<2>(float64 x, float64 X, float64 rdx, float64 dt, float64 rdt, float64 s[3])
{
  float64 delta  = (x - X) * rdx;
  float64 t1     = std::abs(delta) < dt ? 1 : 0;
  float64 t2     = 1 - t1;
  float64 delta0 = 1 - std::abs(delta);
  float64 delta1 = dt - delta;
  float64 delta2 = dt + delta;

  // clang-format off
  s[0] = 0.25 * rdt * delta1 * delta1                    * t1 + std::max(0.0, -delta) * t2;
  s[1] = 0.50 * rdt * (2 * dt - dt * dt - delta * delta) * t1 + delta0                * t2;
  s[2] = 0.25 * rdt * delta2 * delta2                    * t1 + std::max(0.0, +delta) * t2;
  // clang-format on
}

template <>
void shape_wt<3>(float64 x, float64 X, float64 rdx, float64 dt, float64 rdt, float64 s[4])
{
  float64 delta  = (x - X) * rdx;
  float64 delta0 = 1 - delta;
  float64 delta1 = 1 - 2 * delta;
  float64 delta2 = 1 + 2 * delta;
  float64 delta3 = 2 * dt + delta1;
  float64 delta4 = 2 * dt - delta1;
  float64 delta5 = 3 - 2 * delta;
  float64 t1     = delta < 0.5 - dt ? 1 : 0;
  float64 t2     = 1 - t1;
  float64 t3     = delta < 0.5 + dt ? 1 : 0;
  float64 t4     = 1 - t3;

  // clang-format off
  float64 s01 = (4 * dt * dt + 3 * delta1 * delta1) / 24;
  float64 s11 = (9 - 4 * dt * dt - 12 * delta * delta) / 12;
  float64 s21 = (4 * dt * dt + 3 * delta2 * delta2) / 24;
  float64 s31 = 0;
  float64 s02 = delta3 * delta3 * delta3 / (96 * dt);
  float64 s12 = (-8*dt*dt*dt - 36*dt*dt*delta1 - 6*dt*delta1*delta1 - 3*delta1*delta1*delta1) / (96*dt) + delta0;
  float64 s22 = (-8*dt*dt*dt + 36*dt*dt*delta1 - 6*dt*delta1*delta1 + 3*delta1*delta1*delta1) / (96*dt) + delta;
  float64 s32 = delta4 * delta4 * delta4 / (96 * dt);
  float64 s03 = 0;
  float64 s13 = (4 * dt * dt + 3 * delta5 * delta5) / 24;
  float64 s23 = (9 - 4 * dt * dt - 12 * delta0 * delta0) / 12;
  float64 s33 = (4 * dt * dt + 3 * delta1 * delta1) / 24;
  // clang-format on

  s[0] = s01 * t1 + s02 * t2 * t3 + s03 * t4;
  s[1] = s11 * t1 + s12 * t2 * t3 + s13 * t4;
  s[2] = s21 * t1 + s22 * t2 * t3 + s23 * t4;
  s[3] = s31 * t1 + s32 * t2 * t3 + s33 * t4;
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
/// @brief calculate current via density decomposition scheme
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
template <int Order>
static void esirkepov3d(float64 dxdt, float64 dydt, float64 dzdt, float64 ss[2][3][Order + 3],
                        float64 current[Order + 3][Order + 3][Order + 3][4])
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

} // namespace primitives

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
