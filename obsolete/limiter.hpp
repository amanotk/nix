// -*- C++ -*-
#ifndef _LIMITER_HPP_
#define _LIMITER_HPP_

///
/// Implementation of Nonlinear Limiter Functions
///
/// $Id: limiter.hpp,v b821754eb425 2015/03/26 03:35:33 amano $
///

namespace limiter
{
inline
float64 sign(const float64 x)
{
  return copysign(1.0, x);
}

//
// -*- maximum / minimum -*-
//
template <class T> inline
T max(const T a, const T b)
{
  return std::max(a, b);
}

template <class T> inline
T min(const T a, const T b)
{
  return std::min(a, b);
}

template <class T> inline
T max(const T a, const T b, const T c)
{
  return max(max(a, b), c);
}

template <class T> inline
T min(const T a, const T b, const T c)
{
  return min(min(a, b), c);
}

template <class T> inline
T max(const T a, const T b, const T c, const T d)
{
  return max(max(a, b), max(c, d));
}

template <class T> inline
T min(const T a, const T b, const T c, const T d)
{
  return min(min(a, b), min(c, d));
}

//
// -*- minnmod -*-
//
inline
float64 minmod(const float64 a, const float64 b)
{
  return 0.5*(sign(a) + sign(b)) * min(std::abs(a), std::abs(b));
}

inline
float64 minmod(const float64 a, const float64 b, const float64 c)
{
  return 1.0/3.0*(sign(a) + sign(b))*std::abs(sign(a) + sign(c)) *
    min(std::abs(a),std::abs(b), std::abs(c));
}

inline
float64 minmod(const float64 a, const float64 b,
               const float64 c, const float64 d)
{
  return 0.125*(sign(a) + sign(b)) *
    std::abs((sign(a) + sign(c))*(sign(a) + sign(d))) *
    min(std::abs(a), std::abs(b), std::abs(c), std::abs(d));
}

//
// -*- MC2 -*-
//
inline
float64 mc2(const float64 a, const float64 b)
{
  return 0.5*(sign(a) + sign(b)) *
    min(2*std::abs(a), 2*std::abs(b), 0.5*std::abs(a+b));
}

//
// -*- MP5 -*-
//
inline
void mp5(const float64 fm2, const float64 fm1, const float64 fc0,
         const float64 fp1, const float64 fp2,
         float64 &ffp, float64 & ffm)
{
  static const float64 alpha = 4.0;
  static const float64 beta  = 4.0/3.0;

  float64 dm, dc, dp, ddp, ddm;
  float64 fulp, fmdp, flcp, fminp, fmaxp;
  float64 fulm, fmdm, flcm, fminm, fmaxm;

  dm = fm2 - 2*fm1 + fc0;
  dc = fm1 - 2*fc0 + fp1;
  dp = fc0 - 2*fp1 + fp2;

  ddp = minmod(4*dc-dp, 4*dp-dc, dc, dp);
  ddm = minmod(4*dm-dc, 4*dc-dm, dc, dm);

  // i+1/2
  fulp = fc0 + alpha*(fc0 - fm1);
  fmdp = 0.5*(fc0 + fp1 - ddp);
  flcp = 1.5*fc0 - 0.5*fm1 + beta*ddm;

  fminp = max(min(fc0, fp1, fmdp), min(fc0, fulp, flcp));
  fmaxp = min(max(fc0, fp1, fmdp), max(fc0, fulp, flcp));
  ffp = ffp + minmod(fminp-ffp, fmaxp-ffp);

  // i-1/2
  fulm = fc0 + alpha*(fc0 - fp1);
  fmdm = 0.5*(fc0 + fm1 - ddm);
  flcm = 1.5*fc0 - 0.5*fp1 + beta*ddp;

  fminm = max(min(fc0, fm1, fmdm), min(fc0, fulm, flcm));
  fmaxm = min(max(fc0, fm1, fmdm), max(fc0, fulm, flcm));
  ffm = ffm + minmod(fminm-ffm, fmaxm-ffm);
}

//
// Calculate correction for reconstruction of:
// 1) cell-average to point-value
// 2) point-value to derivative
//

inline
float64 correct_c2p_full(const float64 fm2, const float64 fm1,
                         const float64 fc0, const float64 fp1, const float64 fp2)
{
  static const float64 d2 = -1.0/24.0;
  static const float64 d4 = +3.0/640.0;

  return d2*(fm1 - 2*fc0 + fp1) + d4*(fm2 - 4*fm1 + 6*fc0 - 4*fp1 + fp2);
}

inline
float64 correct_c2p(const float64 fm2, const float64 fm1,
                    const float64 fc0, const float64 fp1, const float64 fp2)
{
  static const float64 d2 = -1.0/24.0;

  float64 dm = fm2 - 2*fm1 + fc0;
  float64 dc = fm1 - 2*fc0 + fp1;
  float64 dp = fc0 - 2*fp1 + fp2;

  return d2*minmod(dm, dc, dp);
}

}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
