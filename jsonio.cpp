// -*- C++ -*-

#include "jsonio.hpp"
#include "tinyformat.hpp"

namespace jsonio
{
using std::cerr;
using std::endl;
using tfm::format;

template <typename T_int>
T_int get_size(const int32_t ndim, const T_int shape[])
{
  T_int size = 1;
  for (int i = 0; i < ndim; i++) {
    size *= shape[i];
  }
  return size;
}

// calculate offset
void calculate_global_offset(int64_t lsize, int64_t *offset, int64_t *gsize)
{
  int nprocess;
  int thisrank;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocess);
  MPI_Comm_rank(MPI_COMM_WORLD, &thisrank);

  int64_t *buffer = new int64_t[nprocess];

  MPI_Allgather(&lsize, 1, MPI_INT64_T, buffer, 1, MPI_INT64_T, MPI_COMM_WORLD);

  *offset = 0;
  for (int i = 1; i <= thisrank; i++) {
    *offset += buffer[i - 1];
  }

  *gsize = 0;
  for (int i = 0; i < nprocess; i++) {
    *gsize += buffer[i];
  }

  delete[] buffer;
}

// collective raed/write with hindexed type
void hindexed_readwrite(MPI_File *fh, int64_t *disp, void *data,
                        const int64_t offset, const int64_t size,
                        const int32_t elembyte, const int32_t packbyte,
                        const int mode)
{
  MPI_Status status;
  MPI_Datatype ptype, ftype;
  MPI_Aint packed_offset[1];
  int32_t packed_size[1];

  packed_offset[0] = elembyte * offset;
  packed_size[0] = static_cast<int32_t>(size * elembyte / packbyte);

  MPI_Type_contiguous(packbyte, MPI_BYTE, &ptype);
  MPI_Type_commit(&ptype);

  MPI_Type_create_hindexed(1, packed_size, packed_offset, ptype, &ftype);
  MPI_Type_commit(&ftype);

  MPI_File_set_view(*fh, *disp, ptype, ftype, "native", MPI_INFO_NULL);

  switch (mode) {
  case +1:
    // read
    MPI_File_read_all(*fh, data, packed_size[0], ptype, &status);
    break;
  case -1:
    // write
    MPI_File_write_all(*fh, data, packed_size[0], ptype, &status);
    break;
  default:
    cerr << format("Error: No such mode available\n");
  }

  MPI_Type_free(&ptype);
  MPI_Type_free(&ftype);
}

// collective raed/write with subarray type
void subarray_readwrite(MPI_File *fh, int64_t *disp, void *data,
                        const int32_t ndim, const int32_t gshape[],
                        const int32_t lshape[], const int32_t offset[],
                        const int32_t elembyte, const int mode, const int order)
{
  MPI_Datatype ptype, ftype;
  MPI_Status status;
  int count = get_size(ndim, lshape);

  MPI_Type_contiguous(elembyte, MPI_BYTE, &ptype);
  MPI_Type_commit(&ptype);

  MPI_Type_create_subarray(ndim, gshape, lshape, offset, order, ptype, &ftype);
  MPI_Type_commit(&ftype);

  MPI_File_set_view(*fh, *disp, ptype, ftype, "native", MPI_INFO_NULL);

  switch (mode) {
  case +1:
    // read
    MPI_File_read_all(*fh, data, count, ptype, &status);
    break;
  case -1:
    // write
    MPI_File_write_all(*fh, data, count, ptype, &status);
    break;
  default:
    cerr << format("Error: No such mode available\n");
  }

  MPI_Type_free(&ptype);
  MPI_Type_free(&ftype);
}

void open_file(const char *filename, MPI_File *fh, int64_t *disp,
               const char *mode)
{
  int status;

  switch (mode[0]) {
  case 'r':
    // read only
    status = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY,
                           MPI_INFO_NULL, fh);
    if (status != MPI_SUCCESS) {
      cerr << format("Error: failed to open file: %s\n", filename);
    }

    // set pointer to the beggining
    *disp = 0;
    MPI_File_seek(*fh, *disp, MPI_SEEK_SET);

    break;
  case 'w':
    // write only
    status = MPI_File_delete(filename, MPI_INFO_NULL);
    status =
        MPI_File_open(MPI_COMM_WORLD, filename,
                      MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, fh);
    if (status != MPI_SUCCESS) {
      cerr << format("Error: failed to open file: %s\n", filename);
    }

    // set pointer to the beggining
    *disp = 0;
    MPI_File_seek(*fh, *disp, MPI_SEEK_SET);

    break;
  case 'a':
    // append
    status =
        MPI_File_open(MPI_COMM_WORLD, filename,
                      MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, fh);
    if (status != MPI_SUCCESS) {
      cerr << format("Error: failed to open file: %s\n", filename);
    }

    // set pointer to the end
    *disp = 0;
    MPI_File_seek(*fh, *disp, MPI_SEEK_END);
    MPI_File_get_position(*fh, disp);

    break;
  default:
    cerr << format("Error: No such mode available\n");
  }
}

void close_file(MPI_File *fh)
{
  MPI_File_close(fh);
}

void write_single(MPI_File *fh, int64_t *disp, void *data, const size_t size)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    MPI_Status status;
    MPI_File_write_at(*fh, *disp, data, size, MPI_BYTE, &status);
  }

  *disp += size;
}

void read_single(MPI_File *fh, int64_t *disp, void *data, const size_t size)
{
  MPI_Status status;
  MPI_File_read_at(*fh, *disp, data, size, MPI_BYTE, &status);

  *disp += size;
}

void write_collective(MPI_File *fh, int64_t *disp, void *data,
                      const int64_t size, const int32_t elembyte,
                      const int32_t packbyte)
{
  int64_t gsize, offset;
  int32_t pbyte;

  if (packbyte < 0) {
    pbyte = elembyte;
  } else {
    pbyte = packbyte;
  }

  // calculate offset
  calculate_global_offset(size, &offset, &gsize);

  // write to disk
  hindexed_readwrite(fh, disp, data, offset, size, elembyte, pbyte, -1);

  *disp += gsize * elembyte;
}

void read_collective(MPI_File *fh, int64_t *disp, void *data,
                     const int64_t size, const int32_t elembyte,
                     const int32_t packbyte)
{
  int64_t gsize, offset;
  int32_t pbyte;

  if (packbyte < 0) {
    pbyte = elembyte;
  } else {
    pbyte = packbyte;
  }

  // calculate offset
  calculate_global_offset(size, &offset, &gsize);

  // read from disk
  hindexed_readwrite(fh, disp, data, offset, size, elembyte, pbyte, +1);

  *disp += gsize * elembyte;
}

void write_collective(MPI_File *fh, int64_t *disp, void *data,
                      const int32_t ndim, const int32_t gshape[],
                      const int32_t lshape[], const int32_t offset[],
                      const int32_t elembyte, const int order)
{
  if (order != MPI_ORDER_C && order != MPI_ORDER_FORTRAN) {
    cerr << format("Error: No such order available\n");
  }

  subarray_readwrite(fh, disp, data, ndim, gshape, lshape, offset, elembyte, -1,
                     order);

  int64_t gsize = get_size(ndim, gshape);
  *disp += gsize * elembyte;
}

void read_collective(MPI_File *fh, int64_t *disp, void *data,
                     const int32_t ndim, const int32_t gshape[],
                     const int32_t lshape[], const int32_t offset[],
                     const int32_t elembyte, const int order)
{
  if (order != MPI_ORDER_C && order != MPI_ORDER_FORTRAN) {
    cerr << format("Error: No such order available\n");
  }

  subarray_readwrite(fh, disp, data, ndim, gshape, lshape, offset, elembyte, +1,
                     order);

  int64_t gsize = get_size(ndim, gshape);
  *disp += gsize * elembyte;
}

} // namespace jsonio

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
