// -*- C++ -*-
#ifndef _STATEHANDLER_HPP_
#define _STATEHANDLER_HPP_

#include "buffer.hpp"
#include "mpistream.hpp"
#include "nix.hpp"

NIX_NAMESPACE_BEGIN

static constexpr int default_max_file_per_dir = 1000;

class StateHandler
{
protected:
  using Vector = std::vector<int64>;

  int         max_file_per_dir; ///< maximum number of files per directory
  std::string basedir;          ///< base directory

public:
  StateHandler(std::string basedir = "", int max_file_per_dir = default_max_file_per_dir)
      : basedir(basedir), max_file_per_dir(max_file_per_dir)
  {
  }

  template <typename App, typename Data>
  bool save(App&& app, Data&& data, std::string prefix)
  {
    save_application(app, data, prefix);
    save_chunkvec(app, data, prefix);

    return true;
  }

  template <typename App, typename Data>
  bool save_application(App&& app, Data&& data, std::string prefix)
  {
    if (data.thisrank == 0) {
      json                      state  = app.to_json();
      std::vector<std::uint8_t> buffer = json::to_msgpack(state);

      std::string   filename = get_path_with_basedir(prefix) + ".msgpack";
      std::ofstream ofs(filename, std::ios::binary);
      ofs.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
      ofs.close();
    }

    return true;
  }

  template <typename App, typename Data>
  bool save_chunkvec(App&& app, Data&& data, std::string prefix)
  {
    int thisrank = data.thisrank;
    int nprocess = data.nprocess;

    DEBUG2 << tfm::format("start saving chunkvec with prefix %s", prefix);

    std::string path = get_path_with_basedir(prefix);

    MpiStream::create_directory_tree(path, thisrank, nprocess, max_file_per_dir);
    std::string filename =
        MpiStream::get_filename(path, ".data", thisrank, nprocess, max_file_per_dir);

    {
      Vector id;
      Vector size;
      Vector offset;

      save_chunkvec_header(app, data, filename, id, size, offset);
      save_chunkvec_content(app, data, filename, id, size, offset);
    }

    DEBUG2 << tfm::format("finish saving chunkvec with prefix %s", prefix);

    return true;
  }

  template <typename App, typename Data>
  bool load(App&& app, Data&& data, std::string prefix)
  {
    load_application(app, data, prefix);
    load_chunkvec(app, data, prefix);

    return true;
  }

  template <typename App, typename Data>
  bool load_application(App&& app, Data&& data, std::string prefix)
  {
    std::string   filename = get_path_with_basedir(prefix, true) + ".msgpack";
    std::ifstream ifs(filename, std::ios::binary);

    json state  = json::from_msgpack(ifs);
    bool status = app.from_json(state);

    ifs.close();

    return status;
  }

  template <typename App, typename Data>
  bool load_chunkvec(App&& app, Data&& data, std::string prefix)
  {
    int thisrank = data.thisrank;
    int nprocess = data.nprocess;

    DEBUG2 << tfm::format("start loading chunkvec with prefix %s", prefix);

    std::string path = get_path_with_basedir(prefix, true);

    std::string filename =
        MpiStream::get_filename(path, ".data", thisrank, nprocess, max_file_per_dir);

    {
      Vector id;
      Vector size;
      Vector offset;

      load_chunkvec_header(app, data, filename, id, size, offset);
      load_chunkvec_content(app, data, filename, id, size, offset);
    }

    DEBUG2 << tfm::format("finish loading chunkvec with prefix %s", prefix);

    return true;
  }

protected:
  std::string get_path_with_basedir(std::string name, bool require_existence = false)
  {
    namespace fs = std::filesystem;

    fs::path base_path = fs::path(basedir);
    fs::path full_path = base_path / fs::path(name);

    if (require_existence == false) {
      return full_path.string();
    }

    if (fs::exists(full_path) == false) {
      // full_path should exist for loading
      // otherwise try to find it in the last directory of basedir
      fs::path last_dir = base_path;
      while (last_dir.filename() == "" || last_dir.filename() == ".") {
        last_dir = last_dir.parent_path();
      }

      full_path = last_dir.filename() / fs::path(name);
    }

    return full_path.string();
  }

  template <typename App, typename Data>
  bool save_chunkvec_header(App&& app, Data&& data, std::string filename, Vector& id, Vector& size,
                            Vector& offset)
  {
    const int element_size = sizeof(Vector::value_type);

    int64 numchunk    = data.chunkvec.size();
    int64 header_size = (1 + numchunk * 3) * element_size;

    id.resize(numchunk);
    size.resize(numchunk);
    offset.resize(numchunk + 1, 0);

    for (int i = 0; i < data.chunkvec.size(); i++) {
      id[i]   = data.chunkvec[i]->get_id();
      size[i] = data.chunkvec[i]->pack(nullptr, 0);
    }

    // calculate offset for each chunk
    std::partial_sum(size.begin(), size.end(), offset.begin() + 1);
    for (int i = 0; i < offset.size(); i++) {
      offset[i] += header_size;
    }

    // write to disk
    std::ofstream ofs(filename, std::ios::binary);

    ofs.write(reinterpret_cast<const char*>(&numchunk), element_size);
    ofs.write(reinterpret_cast<const char*>(id.data()), element_size * numchunk);
    ofs.write(reinterpret_cast<const char*>(size.data()), element_size * numchunk);
    ofs.write(reinterpret_cast<const char*>(offset.data()), element_size * numchunk);

    ofs.close();

    return true;
  }

  template <typename App, typename Data>
  bool save_chunkvec_content(App&& app, Data&& data, std::string filename, Vector& id, Vector& size,
                             Vector& offset)
  {
    Buffer buffer;
    buffer.resize(*std::max_element(size.begin(), size.end()));

    std::ofstream ofs(filename, std::ios::binary | std::ios::app);

    for (int i = 0; i < data.chunkvec.size(); i++) {
      auto& chunk = data.chunkvec[i];

      if (size[i] == chunk->pack(buffer.get(), 0) && id[i] == chunk->get_id()) {
        ofs.seekp(offset[i], std::ios::beg);
        ofs.write(reinterpret_cast<const char*>(buffer.get()), size[i]);
      } else {
        ERROR << tfm::format("Error in writing Chunk ID %08d", id[i]);
      }
    }

    ofs.close();

    return true;
  }

  template <typename App, typename Data>
  bool load_chunkvec_header(App&& app, Data&& data, std::string filename, Vector& id, Vector& size,
                            Vector& offset)
  {
    const int element_size = sizeof(Vector::value_type);

    int64 numchunk = 0;

    std::ifstream ifs(filename, std::ios::binary);

    ifs.read(reinterpret_cast<char*>(&numchunk), element_size);

    id.resize(numchunk);
    size.resize(numchunk);
    offset.resize(numchunk);

    ifs.read(reinterpret_cast<char*>(id.data()), element_size * numchunk);
    ifs.read(reinterpret_cast<char*>(size.data()), element_size * numchunk);
    ifs.read(reinterpret_cast<char*>(offset.data()), element_size * numchunk);

    ifs.close();

    return true;
  }

  template <typename App, typename Data>
  bool load_chunkvec_content(App&& app, Data&& data, std::string filename, Vector& id, Vector& size,
                             Vector& offset)
  {
    Buffer buffer;
    buffer.resize(*std::max_element(size.begin(), size.end()));

    // local dimensions
    int dims[3];
    dims[0] = data.ndims[0] / data.cdims[0];
    dims[1] = data.ndims[1] / data.cdims[1];
    dims[2] = data.ndims[2] / data.cdims[2];

    // clear
    data.chunkvec.resize(0);
    data.chunkvec.shrink_to_fit();

    std::ifstream ifs(filename, std::ios::binary);

    // read data
    for (int i = 0; i < id.size(); i++) {
      auto chunk = app.create_chunk(dims, 0);

      ifs.seekg(offset[i], std::ios::beg);
      ifs.read(reinterpret_cast<char*>(buffer.get()), size[i]);

      // restore
      if (size[i] == chunk->unpack(buffer.get(), 0) && id[i] == chunk->get_id()) {
        data.chunkvec.push_back(std::move(chunk));
      } else {
        ERROR << tfm::format("Error in reading Chunk ID %08d", id[i]);
      }
    }

    ifs.close();

    return true;
  }
};

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
