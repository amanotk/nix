// -*- C++ -*-
#ifndef _PARTICLE_HPP_
#define _PARTICLE_HPP_

#include "common.hpp"
#include "xtensorall.hpp"

class Particle;

/// List of particles
using ParticleList = std::vector<std::unique_ptr<Particle>>;

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

  xt::xtensor<float64, 2> xp;     ///< particle array
  xt::xtensor<float64, 2> xq;     ///< temporary particle array (for sorting)
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

      xp.resize({np, nc});
      xq.resize({np, nc});
      gindex.resize({np});
      pindex.resize({ng + 1});
      count.resize({ng + 1});

      xp.fill(0);
      xq.fill(0);
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
    xp.resize({0, 0});
    xq.resize({0, 0});
    gindex.resize({0});
    pindex.resize({0});
    count.resize({0});
  }

  ///
  /// @brief swap pointer of particle array with temporary array
  ///
  void swap()
  {
    xp.storage().swap(xq.storage());
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
      xt::view(xq, count(ii), xt::all()) = xt::view(xp, ip, xt::all());

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
  /// @brief template for shape function
  ///
  template <int order>
  static float64 S(float64 xp, float64 xg, float64 rdx);

  ///
  /// @brief template for derivative of shape function
  ///
  template <int order>
  static float64 D(float64 xp, float64 xg, float64 rdx);
};

// first-order shape function
template <>
inline float64 Particle::S<1>(float64 xp, float64 xg, float64 rdx)
{
  float64 xx = std::abs((xp - xg) * rdx);
  float64 s1 = 0.5 * (Particle::sign(1.0 - xx) + 1.0);
  return s1 * (1.0 - xx);
}

// derivative of first-order shape function
template <>
inline float64 Particle::D<1>(float64 xp, float64 xg, float64 rdx)
{
  float64 xx = std::abs((xp - xg) * rdx);
  float64 pm = Particle::sign(xp - xg);
  return 0.5 * (Particle::sign(1.0 - xx) + 1.0) * pm;
}

// second-order shape function
template <>
inline float64 Particle::S<2>(float64 xp, float64 xg, float64 rdx)
{
  float64 xx = std::abs((xp - xg) * rdx);
  float64 yy = 1.5 - xx;
  float64 s1 = 0.5 * (Particle::sign(0.5 - xx) + 1.0);
  float64 s2 = 0.5 * (Particle::sign(1.5 - xx) + 1.0) - s1;

  return s1 * (0.75 - xx * xx) + s2 * 0.5 * yy * yy;
}

// derivative of second-order shape function
template <>
inline float64 Particle::D<2>(float64 xp, float64 xg, float64 rdx)
{
  float64 xx = std::abs((xp - xg) * rdx);
  float64 yy = 1.5 - xx;
  float64 pm = Particle::sign(xp - xg);
  float64 s1 = 0.5 * (Particle::sign(0.5 - xx) + 1.0);
  float64 s2 = 0.5 * (Particle::sign(1.5 - xx) + 1.0) - s1;

  return (s1 * (2.0 * xx) + s2 * yy) * pm;
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
