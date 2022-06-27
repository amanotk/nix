// -*- C++ -*-
#ifndef _JSONIO_HPP_
#define _JSONIO_HPP_

///
/// I/O module with JSON and MPI-IO
///
/// $Id$
///
#include <iostream>
#include <cstddef>
#include <mpi.h>
#include "json.hpp"
#include "tinyformat.hpp"


namespace jsonio
{
using float32 = float;
using float64 = double;
using json = nlohmann::ordered_json;
using string = std::string;

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

void put_metadata(json &obj, string name, string dtype, string desc,
                  const int64_t disp, const int64_t size,
                  const int ndim, const int dims[]);

void put_metadata(json &obj, string name, string dtype, string desc,
                  const int64_t disp, const int64_t size);

void put_attribute(json &obj, string name, const int64_t disp, const int32_t data);

void put_attribute(json &obj, string name, const int64_t disp, const int64_t data);

void put_attribute(json &obj, string name, const int64_t disp, const float32 data);

void put_attribute(json &obj, string name, const int64_t disp, const float64 data);

void put_attribute(json &obj, string name, const int64_t disp, const int length, const int32_t *data);

void put_attribute(json &obj, string name, const int64_t disp, const int length, const int64_t *data);

void put_attribute(json &obj, string name, const int64_t disp, const int length, const float32 *data);

void put_attribute(json &obj, string name, const int64_t disp, const int length, const float64 *data);

void get_metadata(json &obj, string name, string &dtype, string &desc,
                  int64_t &disp, int64_t &size, int &ndim, int dims[]);

void get_metadata(json &obj, string name, string &dtype, string &desc,
                  int64_t &disp, int64_t &size);

template <typename T>
void get_attribute(json &obj, string name, int64_t &disp, T &data);

template <typename T>
void get_attribute(json &obj, string name, int64_t &disp, int length, T *data);

}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
