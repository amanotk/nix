// -*- C++ -*-
#ifndef _IOCORE_HPP_
#define _IOCORE_HPP_

///
/// Parallel I/O module with JSON metadata and MPI-IO binary
///
/// $Id$
///
#include <iostream>
#include <mpi.h>
#include "json.hpp"


namespace iocore
{
using json = nlohmann::ordered_json;
using string = std::string;

//
// meata data
//

void write_metadata(json &obj, string name, string dtype, string desc,
                    const int64_t disp, const int64_t size,
                    const int ndim, const int dims[])
{
  obj[name]["datatype"] = dtype;
  obj[name]["desc"]     = desc;
  obj[name]["offset"]   = disp;
  obj[name]["size"]     = size;
  obj[name]["ndim"]     = ndim;
  obj[name]["shape"]    = {};

  for(int i=0; i < ndim ;i++ ) {
    obj[name]["shape"].push_back(dims[i]);
  }
}


//
// scalar attribute
//

template <typename T>
void write_attribute(json &obj, string name, string dtype, string desc,
                     const int64_t disp, const T data)
{
  const int ndim = 1;
  const int dims[1] = {1};
  write_metadata(obj, name, dtype, desc, disp, sizeof(T), ndim, dims);

  obj[name]["data"] = data;
}

void write_attribute(json &obj, string name, string desc,
                     const int64_t disp, const int32_t data)
{
  write_attribute(obj, name, "i4", desc, disp, data);
}

void write_attribute(json &obj, string name, string desc,
                     const int64_t disp, const int64_t data)
{
  write_attribute(obj, name, "i8", desc, disp, data);
}

void write_attribute(json &obj, string name, string desc,
                     const int64_t disp, const float data)
{
  write_attribute(obj, name, "f4", desc, disp, data);
}

void write_attribute(json &obj, string name, string desc,
                     const int64_t disp, const double data)
{
  write_attribute(obj, name, "f8", desc, disp, data);
}


//
// array attribute
//

template <typename T>
void write_attribute(json &obj, string name, string dtype, string desc,
                     const int64_t disp, const int length, const T *data)
{
  const int ndim = 1;
  const int dims[1] = {length};
  write_metadata(obj, name, dtype, desc, disp, sizeof(T), ndim, dims);

  obj[name]["data"] = json::array();
  for(int i=0; i < length ;i++) {
    obj[name]["data"].push_back(data[i]);
  }
}

void write_attribute(json &obj, string name, string dtype, string desc,
                     const int64_t disp, const int length, const int32_t *data)
{
  write_attribute(obj, name, "i4", desc, disp, length, data);
}

void write_attribute(json &obj, string name, string dtype, string desc,
                     const int64_t disp, const int length, const int64_t *data)
{
  write_attribute(obj, name, "i8", desc, disp, length, data);
}

void write_attribute(json &obj, string name, string dtype, string desc,
                     const int64_t disp, const int length, const float *data)
{
  write_attribute(obj, name, "f4", desc, disp, length, data);
}

void write_attribute(json &obj, string name, string dtype, string desc,
                     const int64_t disp, const int length, const double *data)
{
  write_attribute(obj, name, "f8", desc, disp, length, data);
}

int get_endian_flag();

void open_file(const char *filename, MPI_File *fh, int64_t *disp, const char *mode);

void close_file(MPI_File *fh);

void read_single(MPI_File *fh, int64_t *disp, void *data, const size_t size);

void write_single(MPI_File *fh, int64_t *disp, void *data, const size_t size);

void read_collective(MPI_File *fh, int64_t *disp, void *data,
                     const int64_t size,
                     const int32_t elembyte,
                     const int32_t packbyte=-1);

void write_collective(MPI_File *fh, int64_t *disp, void *data,
                      const int64_t size,
                      const int32_t elembyte,
                      const int32_t packbyte=-1);

void read_collective(MPI_File *fh, int64_t *disp, void *data,
                     const int32_t ndim,
                     const int32_t gdims[],
                     const int32_t ldims[],
                     const int32_t offset[],
                     const int32_t elembyte,
                     const int order=MPI_ORDER_C);

void write_collective(MPI_File *fh, int64_t *disp, void *data,
                      const int32_t ndim,
                      const int32_t gdims[],
                      const int32_t ldims[],
                      const int32_t offset[],
                      const int32_t elembyte,
                      const int order=MPI_ORDER_C);

}

#include "iocore.cpp"

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
