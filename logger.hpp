// -*- C++ -*-
#ifndef _LOGGER_HPP_
#define _LOGGER_HPP_

#include "nix.hpp"
#include "tinyformat.hpp"
#include <nlohmann/json.hpp>

NIX_NAMESPACE_BEGIN

static constexpr int logger_flush_interval = 10;

static const json default_config = R"(
{
  "prefix": "log",
  "path": ".",
  "interval": 100,
  "flush" : 10.0
}
)"_json;

///
/// @brief Simple Logger class
///
class Logger
{
protected:
  int               thisrank;     ///< MPI rank
  float64           last_flushed; ///< last flushed time
  json              config;       ///< configuration
  std::vector<json> content;      ///< log content

public:
  Logger(int rank, const json& object, bool append = false) : thisrank(rank)
  {
    initialize_content();

    // set configuration (use default if not specified)
    if (object.is_null() == true) {
      config = default_config;
    } else {
      for (auto& element : default_config.items()) {
        config[element.key()] = object.value(element.key(), element.value());
      }
    }

    if (append == false) {
      std::filesystem::remove(get_filename());
    }

    last_flushed = wall_clock();
  }

  virtual int get_interval()
  {
    return config["interval"];
  }

  virtual std::string get_filename()
  {
    std::string path   = config["path"];
    std::string prefix = config["prefix"];

    return tfm::format("%s/%s.msgpack", path, prefix);
  }

  virtual bool is_flush_required()
  {
    return (wall_clock() - last_flushed > config["flush"]);
  }

  virtual void initialize_content()
  {
    content.resize(0);
    content.push_back({});
  }

  virtual void log(int curstep)
  {
    int  interval      = get_interval();
    bool is_final_step = (curstep + 1) % interval == 0;

    // save file
    bool status = this->save(curstep, is_final_step);

    // clear
    if (status)
      initialize_content();
  }

  virtual void append(int curstep, const char* name, json& obj)
  {
    json& last = content.back();

    // check the last element and append new element if needed
    if (last.is_null() == true) {
      last = {{"rank", thisrank}, {"step", curstep}};
    } else if (last.contains("step") == true && last.value("step", -1) != curstep) {
      content.push_back({{"rank", thisrank}, {"step", curstep}});
    }

    // data
    json& element = content.back();

    element[name] = {};
    for (auto it = obj.begin(); it != obj.end(); ++it) {
      element[name][it.key()] = it.value();
    }
  }

  virtual bool save(int curstep, bool force = false)
  {
    std::string filename = get_filename();
    int         interval = get_interval();
    int         step     = interval * (curstep / interval);

    bool status = (force == true) || is_flush_required();

    if (thisrank == 0 && status) {
      // output
      std::ofstream ofs(filename, std::ios::binary | std::ios::app);

      for (auto it = content.begin(); it != content.end(); ++it) {
        std::vector<std::uint8_t> buffer = json::to_msgpack(*it);
        ofs.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
      }
      ofs.close();

      last_flushed = wall_clock();
    }

    return status;
  }
};

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
