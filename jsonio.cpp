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
  for(int i=0; i < ndim ;i++) {
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

  int64_t *buffer = new int64_t [nprocess];

  MPI_Allgather(&lsize, 1, MPI_INT64_T, buffer, 1, MPI_INT64_T, MPI_COMM_WORLD);

  *offset = 0;
  for(int i=1; i <= thisrank; i++) {
    *offset += buffer[i-1];
  }

  *gsize = 0;
  for(int i=0; i < nprocess; i++) {
    *gsize += buffer[i];
  }

  delete [] buffer;
}


// collective raed/write with hindexed type
void hindexed_readwrite(MPI_File *fh, int64_t *disp, void *data,
                        const int64_t offset,
                        const int64_t size,
                        const int32_t elembyte,
                        const int32_t packbyte,
                        const int mode)
{
  MPI_Status status;
  MPI_Datatype ptype, ftype;
  MPI_Aint packed_offset[1];
  int32_t packed_size[1];

  packed_offset[0] = elembyte*offset;
  packed_size[0]   = static_cast<int32_t>(size*elembyte / packbyte);

  MPI_Type_contiguous(packbyte, MPI_BYTE, &ptype);
  MPI_Type_commit(&ptype);

  MPI_Type_create_hindexed(1, packed_size, packed_offset, ptype, &ftype);
  MPI_Type_commit(&ftype);

  MPI_File_set_view(*fh, *disp, ptype, ftype, "native", MPI_INFO_NULL);

  switch(mode) {
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
                        const int32_t ndim,
                        const int32_t gshape[],
                        const int32_t lshape[],
                        const int32_t offset[],
                        const int32_t elembyte,
                        const int mode,
                        const int order)
{
  MPI_Datatype ptype, ftype;
  MPI_Status status;
  int count = get_size(ndim, lshape);

  MPI_Type_contiguous(elembyte, MPI_BYTE, &ptype);
  MPI_Type_commit(&ptype);

  MPI_Type_create_subarray(ndim, gshape, lshape, offset, order, ptype, &ftype);
  MPI_Type_commit(&ftype);

  MPI_File_set_view(*fh, *disp, ptype, ftype, "native", MPI_INFO_NULL);

  switch(mode) {
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


// return 1 on little endian and 16777216 on big endian
int32_t get_endian_flag()
{
  union {
    int  flag;
    char byte[4] = {1, 0, 0, 0};
  } endian_flag;

  return endian_flag.flag;
}


void open_file(const char *filename, MPI_File *fh, int64_t *disp,
               const char *mode)
{
  int status;

  switch (mode[0]) {
  case 'r':
    // read only
    status = MPI_File_open(MPI_COMM_WORLD, filename,
                           MPI_MODE_RDONLY,
                           MPI_INFO_NULL, fh);
    if( status != MPI_SUCCESS ) {
      cerr << format("Error: failed to open file: %s\n", filename);
    }

    // set pointer to the beggining
    *disp = 0;
    MPI_File_seek(*fh, *disp, MPI_SEEK_SET);

    break;
  case 'w':
    // write only
    status = MPI_File_delete(filename, MPI_INFO_NULL);
    status = MPI_File_open(MPI_COMM_WORLD, filename,
                           MPI_MODE_WRONLY | MPI_MODE_CREATE,
                           MPI_INFO_NULL, fh);
    if( status != MPI_SUCCESS ) {
      cerr << format("Error: failed to open file: %s\n", filename);
    }


    // set pointer to the beggining
    *disp = 0;
    MPI_File_seek(*fh, *disp, MPI_SEEK_SET);

    break;
  case 'a':
    // append
    status = MPI_File_open(MPI_COMM_WORLD, filename,
                           MPI_MODE_WRONLY | MPI_MODE_CREATE,
                           MPI_INFO_NULL, fh);
    if( status != MPI_SUCCESS ) {
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


void write_single(MPI_File *fh, int64_t *disp, void *data,
                  const size_t size)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if( rank == 0 ) {
    MPI_Status status;
    MPI_File_write_at(*fh, *disp, data, size, MPI_BYTE, &status);
  }

  *disp += size;
}


void read_single(MPI_File *fh, int64_t *disp, void *data,
                 const size_t size)
{
  MPI_Status status;
  MPI_File_read_at(*fh, *disp, data, size, MPI_BYTE, &status);

  *disp += size;
}


void write_collective(MPI_File *fh, int64_t *disp, void *data,
                      const int64_t size,
                      const int32_t elembyte,
                      const int32_t packbyte)
{
  int64_t gsize, offset;
  int32_t pbyte;

  if( packbyte < 0 ) {
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
                     const int64_t size,
                     const int32_t elembyte,
                     const int32_t packbyte)
{
  int64_t gsize, offset;
  int32_t pbyte;

  if( packbyte < 0 ) {
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
                      const int32_t ndim,
                      const int32_t gshape[],
                      const int32_t lshape[],
                      const int32_t offset[],
                      const int32_t elembyte,
                      const int order)
{
  if( order != MPI_ORDER_C && order != MPI_ORDER_FORTRAN ) {
    cerr << format("Error: No such order available\n");
  }

  subarray_readwrite(fh, disp, data, ndim, gshape, lshape, offset, elembyte, -1, order);

  int64_t gsize = get_size(ndim, gshape);
  *disp += gsize * elembyte;
}


void read_collective(MPI_File *fh, int64_t *disp, void *data,
                     const int32_t ndim,
                     const int32_t gshape[],
                     const int32_t lshape[],
                     const int32_t offset[],
                     const int32_t elembyte,
                     const int order)
{
  if( order != MPI_ORDER_C && order != MPI_ORDER_FORTRAN ) {
    cerr << format("Error: No such order available\n");
  }

  subarray_readwrite(fh, disp, data, ndim, gshape, lshape, offset, elembyte, +1, order);

  int64_t gsize = get_size(ndim, gshape);
  *disp += gsize * elembyte;
}

void put_metadata(json &obj, string name, string dtype, string desc,
                  const int64_t disp, const int64_t size,
                  const int32_t ndim, const int32_t dims[])
{
  obj[name]["datatype"]    = dtype;
  obj[name]["description"] = desc;
  obj[name]["offset"]      = disp;
  obj[name]["size"]        = size;
  obj[name]["ndim"]        = ndim;
  obj[name]["shape"]       = json::array();

  for(int i=0; i < ndim ;i++ ) {
    obj[name]["shape"].push_back(dims[i]);
  }
}

void put_metadata(json &obj, string name, string dtype, string desc,
                  const int64_t disp, const int64_t size)
{
  const int ndim = 1;
  const int dims[1] = {1};
  put_metadata(obj, name, dtype, desc, disp, size, ndim, dims);
}

void get_metadata(json &obj, string name, string &dtype, string &desc,
                  int64_t &disp, int64_t &size, int32_t &ndim, int32_t dims[])
{
  dtype = obj[name]["datatype"].get<string>();
  desc  = obj[name]["description"].get<string>();
  disp  = obj[name]["offset"].get<int64_t>();
  size  = obj[name]["size"].get<int64_t>();
  ndim  = obj[name]["ndim"].get<int>();

  auto v = obj[name]["shape"];
  for(int i=0; i < ndim ;i++ ) {
    dims[i] = v[i].get<int>();
  }
}

void get_metadata(json &obj, string name, string &dtype, string &desc,
                  int64_t &disp, int64_t &size)
{
  int32_t ndim;
  int32_t dims[1];
  get_metadata(obj, name, dtype, desc, disp, size, ndim, dims);
}

template <typename T>
void get_attribute(json &obj, string name, int64_t &disp, T &data)
{
  string dtype;
  string desc;
  int64_t size;
  get_metadata(obj, name, dtype, desc, disp, size);
  data = obj[name]["data"].get<T>();
}

template <typename T>
void get_attribute(json &obj, string name, int64_t &disp, int32_t length, T *data)
{
  string dtype;
  string desc;
  int64_t size;
  get_metadata(obj, name, dtype, desc, disp, size);
  for(int i=0; i < length ;i++) {
    data[i] = obj[name]["data"].get<T>();
  }
}

template <> void get_attribute<int32_t>(json &obj, string name, int64_t &disp, int32_t &data);
template <> void get_attribute<int64_t>(json &obj, string name, int64_t &disp, int64_t &data);
template <> void get_attribute<float32>(json &obj, string name, int64_t &disp, float32 &data);
template <> void get_attribute<float64>(json &obj, string name, int64_t &disp, float64 &data);
template <> void get_attribute<int32_t>(json &obj, string name, int64_t &disp, int32_t length, int32_t *data);
template <> void get_attribute<int64_t>(json &obj, string name, int64_t &disp, int32_t length, int64_t *data);
template <> void get_attribute<float32>(json &obj, string name, int64_t &disp, int32_t length, float32 *data);
template <> void get_attribute<float64>(json &obj, string name, int64_t &disp, int32_t length, float64 *data);


void put_attribute(json &obj, string name, const int64_t disp, const int32_t data)
{
  put_metadata(obj, name, "i4", "", disp, sizeof(int32_t));
  obj[name]["data"] = data;
}

void put_attribute(json &obj, string name, const int64_t disp, const int64_t data)
{
  put_metadata(obj, name, "i8", "", disp, sizeof(int64_t));
  obj[name]["data"] = data;
}

void put_attribute(json &obj, string name, const int64_t disp, const float32 data)
{
  put_metadata(obj, name, "f4", "", disp, sizeof(float32));
  obj[name]["data"] = data;
}

void put_attribute(json &obj, string name, const int64_t disp, const float64 data)
{
  put_metadata(obj, name, "f8", "", disp, sizeof(float64));
  obj[name]["data"] = data;
}

void put_attribute(json &obj, string name, const int64_t disp, const int32_t length, const int32_t *data)
{
  int32_t dims[1] = {length};
  put_metadata(obj, name, "i4", "", disp, sizeof(int32_t)*length, 1, dims);
  obj[name]["data"] = std::vector<int32_t>(&data[0], &data[length-1]);
}

void put_attribute(json &obj, string name, const int64_t disp, const int32_t length, const int64_t *data)
{
  int32_t dims[1] = {length};
  put_metadata(obj, name, "i8", "", disp, sizeof(int64_t)*length, 1, dims);
  obj[name]["data"] = std::vector<int64_t>(&data[0], &data[length-1]);
}

void put_attribute(json &obj, string name, const int64_t disp, const int32_t length, const float32 *data)
{
  int32_t dims[1] = {length};
  put_metadata(obj, name, "f4", "", disp, sizeof(float32)*length, 1, dims);
  obj[name]["data"] = std::vector<float32>(&data[0], &data[length-1]);
}

void put_attribute(json &obj, string name, const int64_t disp, const int32_t length, const float64 *data)
{
  int32_t dims[1] = {length};
  put_metadata(obj, name, "f8", "", disp, sizeof(float64)*length, 1, dims);
  obj[name]["data"] = std::vector<float64>(&data[0], &data[length-1]);
}

}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
