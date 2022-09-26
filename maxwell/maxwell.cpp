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
  int recvmode = RecvMode;

  std::set<int> queue;

  for (int i = 0; i < numchunk; i++) {
    chunkvec[i]->push(delt);
    chunkvec[i]->set_boundary_begin();
    queue.insert(i);
  }

  // wait for boundary exchange
  this->wait_bc_exchange(queue, recvmode);

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

  MPI_File fh;
  size_t   disp;
  json     dataset;

  // open file
  jsonio::open_file(fn_data.c_str(), &fh, &disp, "w");

  {
    const char name[]  = "uf";
    const char desc[]  = "";
    const int  nc      = cdims[3];
    const int  nz      = ndims[0] / cdims[0];
    const int  ny      = ndims[1] / cdims[1];
    const int  nx      = ndims[2] / cdims[2];
    const int  ndim    = 5;
    const int  dims[5] = {nc, nz, ny, nx, 6};
    const int  size    = nc * nz * ny * nx * 6 * sizeof(float64);

    jsonio::put_metadata(dataset, name, "f8", desc, disp, size, ndim, dims);
    this->write_chunk_all(fh, disp, 0);
  }

  // close file
  jsonio::close_file(&fh);

  //
  // output json file
  //
  {
    json root;
    json cmap;

    // convert chunkmap into json
    chunkmap->save_json(cmap);

    // meta data
    root["meta"] = {{"endian", get_endian_flag()},
                    {"rawfile", fn_data},
                    {"order", 1},
                    {"time", curtime},
                    {"step", curstep}};
    // chunkmap
    root["chunkmap"] = cmap;
    // dataset
    root["dataset"] = dataset;

    if (thisrank == 0) {
      std::ofstream ofs(fn_json);
      ofs << std::setw(2) << root;
      ofs.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
}

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
