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

  // set initial condition
  for (int i = 0; i < numchunk; i++) {
    int offset[3] = {0, 0, 0};
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
  int      dims[5] = {cdims[3], ndims[0], ndims[1], ndims[2], 6};
  int      size    = dims[0] * dims[1] * dims[2] * dims[3] * dims[4] * sizeof(float64);

  // open file
  jsonio::open_file(fn_data.c_str(), &fh, &disp, "w");

  // json metadata
  jsonio::put_metadata(dataset, "uf", "f8", "", disp, size, ndim, dims);

  // check buffer size
  bufsize = 0;
  for (int i = 0; i < numchunk; i++) {
    bufsize = std::max(bufsize, chunkvec[i]->pack(FDTD::PackEmfQuery, nullptr));
  }
  sendbuf.resize(bufsize);

  // write data for each chunk
  for (int i = 0; i < numchunk; i++) {
    int         byte;
    MPI_Request req;

    byte = chunkvec[i]->pack(FDTD::PackEmf, sendbuf.get());
    assert(byte == bufsize);

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

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
