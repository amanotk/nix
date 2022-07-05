// -*- C++ -*-
#ifndef _JSONIO_HPP_
#define _JSONIO_HPP_

///
/// I/O module with JSON and MPI-IO
///
/// $Id$
///
#include "json.hpp"
#include "tinyformat.hpp"
#include <cstddef>
#include <iostream>
#include <mpi.h>

namespace jsonio
{
using float32 = float;
using float64 = double;
using std::string;
using std::cerr;
using std::endl;
using tfm::format;
using json = nlohmann::ordered_json;

void hindexed_readwrite(MPI_File *fh, size_t *disp, void *data,
                        const size_t offset, const size_t size,
                        const int32_t elembyte, const int32_t packbyte,
                        const int mode);

void subarray_readwrite(MPI_File *fh, size_t *disp, void *data,
                        const int32_t ndim, const int32_t gshape[],
                        const int32_t lshape[], const int32_t offset[],
                        const int32_t elembyte, const int mode, const int order);

void open_file(const char *filename, MPI_File *fh, size_t *disp,
               const char *mode);

void close_file(MPI_File *fh);

void read_single(MPI_File *fh, size_t *disp, void *data, const size_t size);

void write_single(MPI_File *fh, size_t *disp, void *data, const size_t size);

void read_collective(MPI_File *fh, size_t *disp, void *data, const size_t size,
                     const int32_t elembyte, const int32_t packbyte = -1);

void write_collective(MPI_File *fh, size_t *disp, void *data, const size_t size,
                      const int32_t elembyte, const int32_t packbyte = -1);

template <typename T1, typename T2, typename T3, typename T4, typename T5>
void read_collective(MPI_File *fh, size_t *disp, void *data, const T1 ndim,
                     const T2 gshape[], const T3 lshape[], const T4 offset[],
                     const T5 elembyte, const int order = MPI_ORDER_C)
{
  if (order != MPI_ORDER_C && order != MPI_ORDER_FORTRAN) {
    cerr << format("Error: No such order available\n");
  }

  // convert to int32_t for MPI call
  int32_t nd = static_cast<int32_t>(ndim);
  int32_t eb = static_cast<int32_t>(elembyte);
  int32_t gs[nd], ls[nd], os[nd];

  size_t size = 1;
  for (int i = 0; i < nd; i++) {
    gs[i] = static_cast<int32_t>(gshape[i]);
    ls[i] = static_cast<int32_t>(lshape[i]);
    os[i] = static_cast<int32_t>(offset[i]);
    size *= gshape[i];
  }

  subarray_readwrite(fh, disp, data, nd, gs, ls, os, eb, +1, order);
  *disp += size * elembyte;
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
void write_collective(MPI_File *fh, size_t *disp, void *data, const T1 ndim,
                      const T2 gshape[], const T3 lshape[], const T4 offset[],
                      const T5 elembyte, const int order = MPI_ORDER_C)
{
  if (order != MPI_ORDER_C && order != MPI_ORDER_FORTRAN) {
    cerr << format("Error: No such order available\n");
  }

  // convert to int32_t for MPI call
  int32_t nd = static_cast<int32_t>(ndim);
  int32_t eb = static_cast<int32_t>(elembyte);
  int32_t gs[nd], ls[nd], os[nd];

  size_t size = 1;
  for (int i = 0; i < nd; i++) {
    gs[i] = static_cast<int32_t>(gshape[i]);
    ls[i] = static_cast<int32_t>(lshape[i]);
    os[i] = static_cast<int32_t>(offset[i]);
    size *= gshape[i];
  }

  subarray_readwrite(fh, disp, data, nd, gs, ls, os, eb, -1, order);
  *disp += size * elembyte;
}

inline void put_metadata(json &obj, string name, string dtype, string desc,
                         const size_t disp, const size_t size,
                         const int32_t ndim, const int32_t dims[])
{
  obj[name]["datatype"] = dtype;
  obj[name]["description"] = desc;
  obj[name]["offset"] = disp;
  obj[name]["size"] = size;
  obj[name]["ndim"] = ndim;
  obj[name]["shape"] = json::array();

  for (int i = 0; i < ndim; i++) {
    obj[name]["shape"].push_back(dims[i]);
  }
}

inline void put_metadata(json &obj, string name, string dtype, string desc,
                         const size_t disp, const size_t size)
{
  const int ndim = 1;
  const int dims[1] = {1};
  put_metadata(obj, name, dtype, desc, disp, size, ndim, dims);
}

inline void get_metadata(json &obj, string name, string &dtype, string &desc,
                         size_t &disp, size_t &size, int32_t &ndim,
                         int32_t dims[])
{
  dtype = obj[name]["datatype"].get<string>();
  desc = obj[name]["description"].get<string>();
  disp = obj[name]["offset"].get<size_t>();
  size = obj[name]["size"].get<size_t>();
  ndim = obj[name]["ndim"].get<int>();

  auto v = obj[name]["shape"];
  for (int i = 0; i < ndim; i++) {
    dims[i] = v[i].get<int>();
  }
}

inline void get_metadata(json &obj, string name, string &dtype, string &desc,
                         size_t &disp, size_t &size)
{
  int32_t ndim;
  int32_t dims[1];
  get_metadata(obj, name, dtype, desc, disp, size, ndim, dims);
}

template <typename T>
inline void get_attribute(json &obj, string name, size_t &disp, T &data)
{
  string dtype;
  string desc;
  size_t size;
  get_metadata(obj, name, dtype, desc, disp, size);
  data = obj[name]["data"].get<T>();
}

template <typename T>
inline void get_attribute(json &obj, string name, size_t &disp, int32_t length,
                          T *data)
{
  string dtype;
  string desc;
  size_t size;
  get_metadata(obj, name, dtype, desc, disp, size);

  std::vector<T> vec = obj[name]["data"].get<std::vector<T>>();
  for (int i = 0; i < length; i++) {
    data[i] = vec[i];
  }
}

inline void put_attribute(json &obj, string name, const size_t disp,
                          const int32_t data)
{
  put_metadata(obj, name, "i4", "", disp, sizeof(int32_t));
  obj[name]["data"] = data;
}

inline void put_attribute(json &obj, string name, const size_t disp,
                          const int64_t data)
{
  put_metadata(obj, name, "i8", "", disp, sizeof(size_t));
  obj[name]["data"] = data;
}

inline void put_attribute(json &obj, string name, const size_t disp,
                          const float32 data)
{
  put_metadata(obj, name, "f4", "", disp, sizeof(float32));
  obj[name]["data"] = data;
}

inline void put_attribute(json &obj, string name, const size_t disp,
                          const float64 data)
{
  put_metadata(obj, name, "f8", "", disp, sizeof(float64));
  obj[name]["data"] = data;
}

inline void put_attribute(json &obj, string name, const size_t disp,
                          const int32_t length, const int32_t *data)
{
  int32_t dims[1] = {length};
  put_metadata(obj, name, "i4", "", disp, sizeof(int32_t) * length, 1, dims);
  obj[name]["data"] = std::vector<int32_t>(&data[0], &data[length - 1]);
}

inline void put_attribute(json &obj, string name, const size_t disp,
                          const int32_t length, const int64_t *data)
{
  int32_t dims[1] = {length};
  put_metadata(obj, name, "i8", "", disp, sizeof(size_t) * length, 1, dims);
  obj[name]["data"] = std::vector<size_t>(&data[0], &data[length - 1]);
}

inline void put_attribute(json &obj, string name, const size_t disp,
                          const int32_t length, const float32 *data)
{
  int32_t dims[1] = {length};
  put_metadata(obj, name, "f4", "", disp, sizeof(float32) * length, 1, dims);
  obj[name]["data"] = std::vector<float32>(&data[0], &data[length - 1]);
}

inline void put_attribute(json &obj, string name, const size_t disp,
                          const int32_t length, const float64 *data)
{
  int32_t dims[1] = {length};
  put_metadata(obj, name, "f8", "", disp, sizeof(float64) * length, 1, dims);
  obj[name]["data"] = std::vector<float64>(&data[0], &data[length - 1]);
}

} // namespace jsonio

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
