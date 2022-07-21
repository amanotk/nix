// -*- C++ -*-
#include "maxwell.hpp"

#define DEFINE_MEMBER(type, name) type Maxwell::name

DEFINE_MEMBER(void, initialize)(int argc, char **argv)
{
  using std::placeholders::_1;
  using std::placeholders::_2;
  using std::placeholders::_3;
  using std::placeholders::_4;
  FDTD::T_function f = std::bind(&Maxwell::initializer, this, _1, _2, _3, _4);

  // default initialize()
  Base::initialize(argc, argv);

  // additional parameters
  interval = cfg_json["interval"].get<int>();
  prefix   = cfg_json["prefix"].get<std::string>();
  cc       = cfg_json["cc"].get<float64>();
  kdir     = cfg_json["kdir"].get<int>();

  // set initial condition
  for (int i = 0; i < numchunk; i++) {
    int ix, iy, iz;
    int offset[3];

    chunkmap->get_coordinate(chunkvec[i]->get_id(), iz, iy, ix);
    offset[0] = iz * ndims[0] / cdims[0];
    offset[1] = iy * ndims[1] / cdims[1];
    offset[2] = ix * ndims[2] / cdims[2];
    chunkvec[i]->setup(cc, delh, offset, f);
  }
}

DEFINE_MEMBER(void, push)()
{
  for (int i = 0; i < numchunk; i++) {
    chunkvec[i]->push(delt);
  }

  curtime += delt;
  curstep++;
}

DEFINE_MEMBER(void, diagnostic)()
{
  if (curstep % interval != 0) {
    return;
  }

  // filename
  std::string filename = prefix + tfm::format("%05d", curstep);
  std::string fn_json  = filename + ".json";
  std::string fn_data  = filename + ".data";

  json     root;
  json     dataset;
  MPI_File fh;
  size_t   disp;
  int      bufsize;
  int      ndim    = 5;
  int      dims[5] = {cdims[3], ndims[0]/cdims[0], ndims[1]/cdims[1], ndims[2]/cdims[2], 6};
  int      size    = dims[0] * dims[1] * dims[2] * dims[3] * dims[4] * sizeof(float64);

  // open file
  jsonio::open_file(fn_data.c_str(), &fh, &disp, "w");

  // json metadata
  jsonio::put_metadata(dataset, "uf", "f8", "", disp, size, ndim, dims);

  // assume buffer size for each chunk is equal
  bufsize = chunkvec[0]->pack(FDTD::PackEmfQuery, nullptr);
  for (int i = 0; i < numchunk; i++) {
    assert(bufsize == chunkvec[i]->pack(FDTD::PackEmfQuery, nullptr));
  }
  sendbuf.resize(bufsize);
  disp += bufsize * chunkvec[0]->get_id();

  // write data for each chunk
  for (int i = 0; i < numchunk; i++) {
    int         byte;
    MPI_Request req;

    byte = chunkvec[i]->pack(FDTD::PackEmf, sendbuf.get());

    jsonio::write_contiguous_at(&fh, &disp, sendbuf.get(), byte, 1, &req);
    disp += byte;

    MPI_Wait(&req, MPI_STATUS_IGNORE);
  }

  jsonio::close_file(&fh);

  // meta data
  root["meta"] = {{"endian", common::get_endian_flag()},
                  {"rawfile", fn_data},
                  {"order", 1},
                  {"time", curtime},
                  {"step", curstep}};
  // dataset
  root["dataset"] = dataset;

  std::ofstream ofs(fn_json);
  ofs << std::setw(2) << root;
  ofs.close();
}

DEFINE_MEMBER(void, initializer)(float64 z, float64 y, float64 x, float64 *eb)
{
  switch (kdir) {
  case 0: {
    // propagation in z dir
    float64 kk = common::pi2 / zlim[2];
    float64 ff = cos(kk * z);
    float64 gg = sin(kk * z);
    eb[0]      = ff;
    eb[1]      = gg;
    eb[2]      = 0;
    eb[3]      = gg;
    eb[4]      = ff;
    eb[5]      = 0;
  } break;
  case 1: {
    // propagation in y dir
    float64 kk = common::pi2 / ylim[2];
    float64 ff = cos(kk * y);
    float64 gg = sin(kk * y);
    eb[0]      = gg;
    eb[1]      = 0;
    eb[2]      = ff;
    eb[3]      = ff;
    eb[4]      = 0;
    eb[5]      = gg;
  } break;
  case 2: {
    // propagation in x dir
    float64 kk = common::pi2 / xlim[2];
    float64 ff = cos(kk * x);
    float64 gg = sin(kk * x);
    eb[0]      = 0;
    eb[1]      = ff;
    eb[2]      = gg;
    eb[3]      = 0;
    eb[4]      = gg;
    eb[5]      = ff;
  } break;
  default:
    break;
  }
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
