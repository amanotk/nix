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
  int               thisrank; ///< MPI rank
  std::string       filename; ///< log filename
  float64           flushed;  ///< last flushed time
  json              config;   ///< configuration
  std::vector<json> content;  ///< log content

public:
  Logger(json& cfg, bool append = false)
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &thisrank);
    initialize_content();

    config  = cfg;
    flushed = wall_clock();

    // log filename
    std::string path   = config.value("path", ".");
    std::string prefix = config.value("prefix", "log");
    filename           = tfm::format("%s/%s.msgpack", path, prefix);
    if (append == false) {
      std::remove(filename.c_str());
    }
  }

  virtual void initialize_content()
  {
    content.resize(0);
    content.push_back({});
  }

  virtual void log(int curstep)
  {
    int  interval      = config.value("interval", 1);
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
    int         interval = config.value("interval", 1);
    int         step     = interval * (curstep / interval);
    float64     wclock   = wall_clock();

    bool status = (force == true) || (wclock - flushed > logger_flush_interval);

    if (thisrank == 0 && status) {
      // output
      std::ofstream ofs(filename, std::ios::binary | std::ios::app);

      for (auto it = content.begin(); it != content.end(); ++it) {
        std::vector<std::uint8_t> buffer = json::to_msgpack(*it);
        ofs.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
      }
      ofs.close();

      flushed = wclock;
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
