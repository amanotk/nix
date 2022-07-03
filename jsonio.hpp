// -*- C++ -*-
#ifndef _JSONIO_HPP_
#define _JSONIO_HPP_

///
/// I/O module with JSON and MPI-IO
///
/// $Id$
///
#include "json.hpp"
#include <cstddef>
#include <iostream>
#include <mpi.h>

namespace jsonio
{
using float32 = float;
using float64 = double;
using string = std::string;
using json = nlohmann::ordered_json;

void open_file(const char *filename, MPI_File *fh, int64_t *disp,
               const char *mode);

void close_file(MPI_File *fh);

void read_single(MPI_File *fh, int64_t *disp, void *data, const size_t size);

void write_single(MPI_File *fh, int64_t *disp, void *data, const size_t size);

void read_collective(MPI_File *fh, int64_t *disp, void *data,
                     const int64_t size, const int32_t elembyte,
                     const int32_t packbyte = -1);

void write_collective(MPI_File *fh, int64_t *disp, void *data,
                      const int64_t size, const int32_t elembyte,
                      const int32_t packbyte = -1);

void read_collective(MPI_File *fh, int64_t *disp, void *data,
                     const int32_t ndim, const int32_t gshape[],
                     const int32_t lshape[], const int32_t offset[],
                     const int32_t elembyte, const int order = MPI_ORDER_C);

void write_collective(MPI_File *fh, int64_t *disp, void *data,
                      const int32_t ndim, const int32_t gshape[],
                      const int32_t lshape[], const int32_t offset[],
                      const int32_t elembyte, const int order = MPI_ORDER_C);

inline void put_metadata(json &obj, string name, string dtype, string desc,
                         const int64_t disp, const int64_t size,
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
                         const int64_t disp, const int64_t size)
{
  const int ndim = 1;
  const int dims[1] = {1};
  put_metadata(obj, name, dtype, desc, disp, size, ndim, dims);
}

inline void get_metadata(json &obj, string name, string &dtype, string &desc,
                         int64_t &disp, int64_t &size, int32_t &ndim,
                         int32_t dims[])
{
  dtype = obj[name]["datatype"].get<string>();
  desc = obj[name]["description"].get<string>();
  disp = obj[name]["offset"].get<int64_t>();
  size = obj[name]["size"].get<int64_t>();
  ndim = obj[name]["ndim"].get<int>();

  auto v = obj[name]["shape"];
  for (int i = 0; i < ndim; i++) {
    dims[i] = v[i].get<int>();
  }
}

inline void get_metadata(json &obj, string name, string &dtype, string &desc,
                         int64_t &disp, int64_t &size)
{
  int32_t ndim;
  int32_t dims[1];
  get_metadata(obj, name, dtype, desc, disp, size, ndim, dims);
}

template <typename T>
inline void get_attribute(json &obj, string name, int64_t &disp, T &data)
{
  string dtype;
  string desc;
  int64_t size;
  get_metadata(obj, name, dtype, desc, disp, size);
  data = obj[name]["data"].get<T>();
}

template <typename T>
inline void get_attribute(json &obj, string name, int64_t &disp, int32_t length,
                          T *data)
{
  string dtype;
  string desc;
  int64_t size;
  get_metadata(obj, name, dtype, desc, disp, size);

  std::vector<T> vec = obj[name]["data"].get<std::vector<T>>();
  for (int i = 0; i < length; i++) {
    data[i] = vec[i];
  }
}

inline void put_attribute(json &obj, string name, const int64_t disp,
                          const int32_t data)
{
  put_metadata(obj, name, "i4", "", disp, sizeof(int32_t));
  obj[name]["data"] = data;
}

inline void put_attribute(json &obj, string name, const int64_t disp,
                          const int64_t data)
{
  put_metadata(obj, name, "i8", "", disp, sizeof(int64_t));
  obj[name]["data"] = data;
}

inline void put_attribute(json &obj, string name, const int64_t disp,
                          const float32 data)
{
  put_metadata(obj, name, "f4", "", disp, sizeof(float32));
  obj[name]["data"] = data;
}

inline void put_attribute(json &obj, string name, const int64_t disp,
                          const float64 data)
{
  put_metadata(obj, name, "f8", "", disp, sizeof(float64));
  obj[name]["data"] = data;
}

inline void put_attribute(json &obj, string name, const int64_t disp,
                          const int32_t length, const int32_t *data)
{
  int32_t dims[1] = {length};
  put_metadata(obj, name, "i4", "", disp, sizeof(int32_t) * length, 1, dims);
  obj[name]["data"] = std::vector<int32_t>(&data[0], &data[length - 1]);
}

inline void put_attribute(json &obj, string name, const int64_t disp,
                          const int32_t length, const int64_t *data)
{
  int32_t dims[1] = {length};
  put_metadata(obj, name, "i8", "", disp, sizeof(int64_t) * length, 1, dims);
  obj[name]["data"] = std::vector<int64_t>(&data[0], &data[length - 1]);
}

inline void put_attribute(json &obj, string name, const int64_t disp,
                          const int32_t length, const float32 *data)
{
  int32_t dims[1] = {length};
  put_metadata(obj, name, "f4", "", disp, sizeof(float32) * length, 1, dims);
  obj[name]["data"] = std::vector<float32>(&data[0], &data[length - 1]);
}

inline void put_attribute(json &obj, string name, const int64_t disp,
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
