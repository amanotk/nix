// -*- C++ -*-
#ifndef _XTENSOR_PACKER3D_HPP_
#define _XTENSOR_PACKER3D_HPP_

#include "nix.hpp"
#include "xtensorall.hpp"

NIX_NAMESPACE_BEGIN

///
/// @brief Data Packer3D for data output
///
class XtensorPacker3D
{
public:
  template <typename Vector>
  int pack_load(Vector& load, uint8_t* buffer, int address)
  {
    int count = sizeof(float64) * load.size() + address;

    if (buffer == nullptr) {
      return count;
    }

    // packing
    float64* ptr = reinterpret_cast<float64*>(buffer + address);
    std::copy(load.begin(), load.end(), ptr);

    return count;
  }

  /// pack coordinate
  template <typename Array>
  int pack_coordinate(int& Lb, int& Ub, Array& x, uint8_t* buffer, int address)
  {
    size_t size  = Ub - Lb + 1;
    int    count = sizeof(float64) * size + address;

    if (buffer == nullptr) {
      return count;
    }

    // packing
    float64* ptr   = reinterpret_cast<float64*>(buffer + address);
    auto     coord = xt::view(x, xt::range(Lb, Ub + 1));
    std::copy(coord.begin(), coord.end(), ptr);

    return count;
  }

  /// pack field
  template <typename Array, typename Data>
  int pack_field(Array& x, Data data, uint8_t* buffer, int address)
  {
    // calculate number of elements
    int size = (data.Ubz - data.Lbz + 1) * (data.Uby - data.Lby + 1) * (data.Ubx - data.Lbx + 1);
    for (int i = 3; i < x.dimension(); i++) {
      size *= x.shape(i);
    }

    int count = sizeof(float64) * size + address;

    if (buffer == nullptr) {
      return count;
    }

    auto Iz = xt::range(data.Lbz, data.Ubz + 1);
    auto Iy = xt::range(data.Lby, data.Uby + 1);
    auto Ix = xt::range(data.Lbx, data.Ubx + 1);
    auto vv = xt::strided_view(x, {Iz, Iy, Ix, xt::ellipsis()});

    // packing
    float64* ptr = reinterpret_cast<float64*>(buffer + address);
    std::copy(vv.begin(), vv.end(), ptr);

    return count;
  }

  /// pack particle
  template <typename ParticlePtr, typename Data>
  int pack_particle(ParticlePtr& p, Data data, uint8_t* buffer, int address)
  {
    int count = address;

    count += memcpy_count(buffer, p->xu.data(), p->Np * Particle::Nc * sizeof(float64), count, 0);

    return count;
  }
};

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
