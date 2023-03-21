// -*- C++ -*-
#ifndef _LOGGER_HPP_
#define _LOGGER_HPP_

#include "nix.hpp"
#include "tinyformat.hpp"
#include <nlohmann/json.hpp>

NIX_NAMESPACE_BEGIN

static constexpr int logger_flush_interval = 10;

///
/// @brief Simple Logger class
///
class Logger
{
private:
  int     thisrank; ///< MPI rank
  float64 flushed;  ///< last flushed time
  json    config;   ///< configuration
  json    content;  ///< log content

public:
  Logger(json& cfg)
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &thisrank);
    initialize_content();

    config  = cfg;
    flushed = wall_clock();
  }

  virtual void initialize_content()
  {
    content = json::array();
    content.push_back({});
  }

  virtual void log(int curstep)
  {
    int  interval      = config.value("interval", 1);
    bool is_final_step = (curstep + 1) % interval == 0;

    // save file
    this->save(curstep, is_final_step);

    // clear
    if (is_final_step)
      initialize_content();
  }

  virtual void append(int curstep, const char* name, json& obj)
  {
    // add new element if the last element already contains the given entry
    json last = content.back();
    if (last.contains(name) == true) {
      content.push_back({});
    }

    json& element = content.back();

    // data
    element[name] = {};
    for (auto it = obj.begin(); it != obj.end(); ++it) {
      element[name][it.key()] = it.value();
    }

    // step
    element[name]["step"] = curstep;
  }

  virtual void save(int curstep, bool force = false)
  {
    int         interval = config.value("interval", 1);
    int         step     = interval * (curstep / interval);
    std::string path     = config.value("path", ".");
    std::string prefix   = config.value("prefix", "log");
    std::string filename = tfm::format("%s/%s_%s.msgpack", path, prefix, format_step(step));
    float64     wclock   = wall_clock();

    if (thisrank == 0 && (force == true || (wclock - flushed > logger_flush_interval))) {
      // serialize to msgpack
      std::vector<std::uint8_t> buffer = json::to_msgpack(content);

      // output
      std::ofstream ofs(filename, std::ios::binary);
      ofs.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
      ofs.close();

      flushed = wclock;
    }
  }
};

NIX_NAMESPACE_END

// Local Variables:
// c-file-style   : "gnu"
// c-file-offsets : ((innamespace . 0) (inline-open . 0))
// End:
#endif
