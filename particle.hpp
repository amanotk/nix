// -*- C++ -*-
#ifndef _PARTICLE_HPP_
#define _PARTICLE_HPP_

#include "common.hpp"
#include "xtensorall.hpp"

class Particle;

/// List of particles
using ParticleList = std::vector<std::shared_ptr<Particle>>;

///
/// @brief Particle Container
///
/// This class is a container for particles of single species. Member variables are intentionally
/// made public for enabling access from everywhere else.
///
/// It also provides some general routines as static inline member functions, including
/// Buneman-Boris pusher, shape functions, and counting sort.
///
class Particle
{
private:
  // these constructors are disabled for safety
  Particle();
  Particle(const Particle &particle);

public:
  static const int Nc = 7; ///< # component for each particle (including ID)

  int Np_total; ///< # total particles
  int Np;       ///< # particles in active
  int Ng;       ///< # grid points (product in x, y, z dirs)

  float64                 qm;     ///< charge/mass
  xt::xtensor<float64, 2> xu;     ///< particle array
  xt::xtensor<float64, 2> xv;     ///< temporary particle array
  xt::xtensor<float64, 1> gindex; ///< index to grid for each particle
  xt::xtensor<float64, 1> pindex; ///< index to first particle for each cell
  xt::xtensor<float64, 1> count;  ///< particle count for each cell

  ///
  /// @brief Constructor
  ///
  /// @param[in] Np_total Total number of particle for memory allocation
  /// @param[in] Ng       Number of grid
  ///
  Particle(const int Np_total, const int Ng)
  {
    this->Np_total = Np_total;
    this->Ng       = Ng;
    Np             = 0;

    // allocate and initialize arrays
    {
      size_t np = Np_total;
      size_t ng = Ng;
      size_t nc = Nc;

      xu.resize({np, nc});
      xv.resize({np, nc});
      gindex.resize({np});
      pindex.resize({ng + 1});
      count.resize({ng + 1});

      xu.fill(0);
      xv.fill(0);
      gindex.fill(0);
      pindex.fill(0);
      count.fill(0);
    }
  }

  ///
  /// @brief Destructor
  ///
  virtual ~Particle()
  {
    xu.resize({0, 0});
    xv.resize({0, 0});
    gindex.resize({0});
    pindex.resize({0});
    count.resize({0});
  }

  ///
  /// @brief swap pointer of particle array with temporary array
  ///
  void swap()
  {
    xu.storage().swap(xv.storage());
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
  /// @brief sort particle array
  ///
  /// This routine assumes that count and gindex are appropriately calculated before calling it.
  ///
  void sort()
  {
    // initial setup
    pindex(0) = 0;

    // cumulative sum of particle count
    xt::view(count, xt::all()) = xt::cumsum(count);

    // first particle index for each cell
    xt::view(pindex, xt::range(1, Ng + 1)) = xt::view(count, xt::range(0, Ng));

    // particle address for rearrangement
    xt::view(count, xt::all()) = xt::view(pindex, xt::all());

    // rearrange particles
    for (int ip = 0; ip < Np; ip++) {
      int ii = gindex(ip);

      // copy particle to temporary at new address
      xt::view(xv, count(ii), xt::all()) = xt::view(xu, ip, xt::all());

      // increment address
      count(ii)++;
    }

    // swap two particle arrays
    swap();

    // particles contained in the last index are discarded
    Np = pindex(Ng);
  }

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
  static float64 sign(const float64 x)
  {
    return copysign(1.0, x);
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
  ///
  static void S1(const float64 x, const float64 X, const float64 rdx, float64 s[2])
  {
    float64 delta = (x - X) * rdx;

    s[0] = 1 - delta;
    s[1] = delta;
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
  ///
  static void S2(const float64 x, const float64 X, const float64 rdx, float64 s[3])
  {
    float64 delta = (x - X) * rdx;

    s[0] = 0.50 * (0.50 + delta) * (0.5 + delta);
    s[1] = 0.75 - delta * delta;
    s[2] = 0.50 * (0.50 - delta) * (0.5 - delta);
  }

  ///
  /// @brief return first-order weighted electromagnetic field
  ///
  template <typename T>
  static float64 weighted_eb1(const T &eb, const int iz, const int iy, const int ix, const int ik,
                              const float64 wz[2], const float64 wy[2], const float64 wx[2],
                              const float64 dt)
  {
    float64 result;

    int ix1 = ix;
    int ix2 = ix1 + 1;
    int iy1 = iy;
    int iy2 = iy1 + 1;
    int iz1 = iz;
    int iz2 = iz1 + 1;

    result = (wx[0] * wy[0] * wz[0] * eb(iz1, iy1, ix1, ik) +
              wx[1] * wy[0] * wz[0] * eb(iz1, iy1, ix2, ik) +
              wx[0] * wy[1] * wz[0] * eb(iz1, iy2, ix1, ik) +
              wx[1] * wy[1] * wz[0] * eb(iz1, iy2, ix2, ik) +
              wx[0] * wy[0] * wz[1] * eb(iz2, iy1, ix1, ik) +
              wx[1] * wy[0] * wz[1] * eb(iz2, iy1, ix2, ik) +
              wx[0] * wy[1] * wz[1] * eb(iz2, iy2, ix1, ik) +
              wx[1] * wy[1] * wz[1] * eb(iz2, iy2, ix2, ik)) *
             dt;
    return result;
  }

  ///
  /// @brief calculate current via first-order density decomposition scheme
  ///
  /// This implements the density decomposition scheme of Esirkepov (2001) with the first-order
  /// shape function in 3D. The input to this routine is the first argument "ss", with ss[0][*][*]
  /// and ss[1][*][*] are 1D weights before and after the movement of particle by one time step,
  /// respectively. The current density is appended to the second argument "current",
  /// which is an array of local 4x4x4 mesh with 4 components including charge density.
  ///
  /// @param[in]     ss      array of 1D weights
  /// @param[in,out] current array of local current
  ///
  static void esirkepov3d1(float64 ss[2][3][4], float64 current[4][4][4][4])
  {
    const float64 A           = 1.0 / 2;
    const float64 B           = 1.0 / 3;
    float64       ww[4][4][4] = {0};

    // rho
    for (int jz = 0; jz < 4; jz++) {
      for (int jy = 0; jy < 4; jy++) {
        for (int jx = 0; jx < 4; jx++) {
          current[jz][jy][jx][0] += ss[1][0][jx] * ss[1][1][jy] * ss[1][2][jz];
        }
      }
    }

    // ss[1][*][*] now represents DS(*,*) of Esirkepov (2001)
    for (int dir = 0; dir < 3; dir++) {
      for (int l = 0; l < 4; l++) {
        ss[1][dir][l] -= ss[0][dir][l];
      }
    }

    // Jx
    for (int jz = 0; jz < 4; jz++) {
      for (int jy = 0; jy < 4; jy++) {
        float64 wx = (1 * ss[0][1][jy] + A * ss[1][1][jy]) * ss[0][2][jz] +
                     (A * ss[0][1][jy] + B * ss[1][1][jy]) * ss[1][2][jz];

        ww[jz][jy][0] = ss[1][0][0] * wx;
        ww[jz][jy][1] = ss[1][0][1] * wx + ww[jz][jy][0];
        ww[jz][jy][2] = ss[1][0][2] * wx + ww[jz][jy][1];
        ww[jz][jy][3] = ss[1][0][3] * wx + ww[jz][jy][2];

        current[jz][jy][1][1] += ww[jz][jy][0];
        current[jz][jy][2][1] += ww[jz][jy][1];
        current[jz][jy][3][1] += ww[jz][jy][2];
      }
    }

    // Jy
    for (int jz = 0; jz < 4; jz++) {
      for (int jx = 0; jx < 4; jx++) {
        float64 wy = (1 * ss[0][2][jz] + A * ss[1][2][jz]) * ss[0][0][jx] +
                     (A * ss[0][2][jz] + B * ss[1][2][jz]) * ss[1][0][jx];

        ww[jz][0][jx] = ss[1][1][0] * wy;
        ww[jz][1][jx] = ss[1][1][1] * wy + ww[jz][0][jx];
        ww[jz][2][jx] = ss[1][1][2] * wy + ww[jz][1][jx];
        ww[jz][3][jx] = ss[1][1][3] * wy + ww[jz][2][jx];

        current[jz][1][jx][2] += ww[jz][0][jx];
        current[jz][2][jx][2] += ww[jz][1][jx];
        current[jz][3][jx][2] += ww[jz][2][jx];
      }
    }

    // Jz
    for (int jy = 0; jy < 4; jy++) {
      for (int jx = 0; jx < 4; jx++) {
        float64 wz = (1 * ss[0][0][jx] + A * ss[1][0][jx]) * ss[0][1][jy] +
                     (A * ss[0][0][jx] + B * ss[1][0][jx]) * ss[1][1][jy];

        ww[0][jy][jx] = ss[1][2][0] * wz;
        ww[1][jy][jx] = ss[1][2][1] * wz + ww[0][jy][jx];
        ww[2][jy][jx] = ss[1][2][2] * wz + ww[1][jy][jx];
        ww[3][jy][jx] = ss[1][2][3] * wz + ww[2][jy][jx];

        current[1][jy][jx][3] += ww[0][jy][jx];
        current[2][jy][jx][3] += ww[1][jy][jx];
        current[3][jy][jx][3] += ww[2][jy][jx];
      }
    }
  }
};

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
