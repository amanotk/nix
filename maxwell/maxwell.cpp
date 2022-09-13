// -*- C++ -*-
#include "maxwell.hpp"

#define DEFINE_MEMBER(type, name) type Maxwell::name

DEFINE_MEMBER(void, initialize)(int argc, char **argv)
{
  // default initialize()
  BaseApp::initialize(argc, argv);

  // additional parameters
  interval = cfg_json["interval"].get<int>();
  prefix   = cfg_json["prefix"].get<std::string>();
  cc       = cfg_json["cc"].get<float64>();
  kdir     = cfg_json["kdir"].get<int>();

  // set auxiliary information for chunk
  for (int i = 0; i < numchunk; i++) {
    int ix, iy, iz;
    int offset[3];

    chunkmap->get_coordinate(chunkvec[i]->get_id(), iz, iy, ix);
    offset[0] = iz * ndims[0] / cdims[0];
    offset[1] = iy * ndims[1] / cdims[1];
    offset[2] = ix * ndims[2] / cdims[2];
    chunkvec[i]->set_global_context(offset, ndims);
  }
}

DEFINE_MEMBER(void, setup)()
{
  // set initial condition
  for (int i = 0; i < numchunk; i++) {
    chunkvec[i]->setup(cfg_json);
    chunkvec[i]->set_boundary_begin();
  }

  for (int i = 0; i < numchunk; i++) {
    chunkvec[i]->set_boundary_end();
  }
}

DEFINE_MEMBER(void, push)()
{
  int recvmode = Chunk::RecvMode;

  std::set<int> waiting;

  for (int i = 0; i < numchunk; i++) {
    chunkvec[i]->push(delt);
    chunkvec[i]->set_boundary_begin();
    waiting.insert(i);
  }

  while (waiting.empty() == false) {
    // try to find a chunk ready for unpacking
    auto iter = std::find_if(waiting.begin(), waiting.end(),
                             [&](int i) { return chunkvec[i]->set_boundary_query(recvmode); });
    if (iter == waiting.end()) {
      continue;
    }

    // unpack and remove from the waiting queue
    chunkvec[*iter]->set_boundary_end();
    waiting.erase(*iter);
  }

  curtime += delt;
  curstep++;
}

DEFINE_MEMBER(void, diagnostic)(std::ostream &out)
{
  if (curstep % interval != 0) {
    return;
  }

  // filename
  std::string filename = prefix + tfm::format("%05d", curstep);
  std::string fn_json  = filename + ".json";
  std::string fn_data  = filename + ".data";

  json     root;
  json     obj_chunkmap;
  json     obj_dataset;
  MPI_File fh;
  size_t   disp;
  int      bufsize;
  int      ndim    = 5;
  int      dims[5] = {cdims[3], ndims[0] / cdims[0], ndims[1] / cdims[1], ndims[2] / cdims[2], 6};
  int      size    = dims[0] * dims[1] * dims[2] * dims[3] * dims[4] * sizeof(float64);

  // open file
  jsonio::open_file(fn_data.c_str(), &fh, &disp, "w");

  // save chunkmap
  chunkmap->save_json(obj_chunkmap);

  // json metadata
  jsonio::put_metadata(obj_dataset, "uf", "f8", "", disp, size, ndim, dims);

  // buffer size (assuming constant)
  bufsize = chunkvec[0]->pack_diagnostic(nullptr, 0);
  sendbuf.resize(bufsize);

  for (int i = 0; i < numchunk; i++) {
    MPI_Request req;

    // pack
    assert(bufsize == chunkvec[i]->pack_diagnostic(sendbuf.get(), 0));

    // write
    size_t chunkdisp = disp + bufsize * chunkvec[i]->get_id();
    jsonio::write_contiguous_at(&fh, &disp, sendbuf.get(), bufsize, 1, &req);

    MPI_Wait(&req, MPI_STATUS_IGNORE);
  }

  disp += size;

  // close file
  jsonio::close_file(&fh);

  //
  // output json file
  //

  // meta data
  root["meta"] = {{"endian", common::get_endian_flag()},
                  {"rawfile", fn_data},
                  {"order", 1},
                  {"time", curtime},
                  {"step", curstep}};
  // chunkmap
  root["chunkmap"] = obj_chunkmap;
  // dataset
  root["dataset"] = obj_dataset;

  if (thisrank == 0) {
    std::ofstream ofs(fn_json);
    ofs << std::setw(2) << root;
    ofs.close();
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
